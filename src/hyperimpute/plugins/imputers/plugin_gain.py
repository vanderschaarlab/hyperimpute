# stdlib
from typing import Any, List, Tuple, Union

# third party
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
import torch
from torch import nn

# hyperimpute absolute
import hyperimpute.logger as log
import hyperimpute.plugins.core.params as params
import hyperimpute.plugins.imputers.base as base
import hyperimpute.plugins.utils.decorators as decorators
from hyperimpute.utils.distributions import enable_reproducible_results

EPS = 1e-8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_Z(m: int, n: int) -> np.ndarray:
    """Random sample generator for Z.

    Args:
        m: number of rows
        n: number of columns

    Returns:
        np.ndarray: generated random values
    """
    res = np.random.uniform(0.0, 0.01, size=[m, n])
    return torch.from_numpy(res).to(DEVICE)


def sample_M(m: int, n: int, p: float) -> np.ndarray:
    """Hint Vector Generation

    Args:
        m: number of rows
        n: number of columns
        p: hint rate

    Returns:
        np.ndarray: generated random values
    """
    unif_prob = np.random.uniform(0.0, 1.0, size=[m, n])
    M = unif_prob > p
    M = 1.0 * M

    return torch.from_numpy(M).to(DEVICE)


def sample_idx(m: int, n: int) -> np.ndarray:
    """Mini-batch generation

    Args:
        m: number of rows
        n: number of columns

    Returns:
        np.ndarray: generated random indices
    """
    idx = np.random.permutation(m)
    idx = idx[:n]
    return idx


class GainModel:
    """The core model for GAIN Imputation.

    Args:
        dim: float
            Number of features.
        h_dim: float
            Size of the hidden layer.
        loss_alpha: int
            Hyperparameter for the generator loss.
    """

    def __init__(
        self,
        dim: int,
        h_dim: int,
        loss_alpha: float = 10,
    ) -> None:
        self.generator_layer = nn.Sequential(
            nn.Linear(dim * 2, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, dim),
            nn.Sigmoid(),
        ).to(DEVICE)
        self.discriminator_layer = nn.Sequential(
            nn.Linear(dim * 2, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, dim),
            nn.Sigmoid(),
        ).to(DEVICE)
        self.loss_alpha = loss_alpha

    def discriminator(self, X: torch.Tensor, hints: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([X, hints], dim=1).float()
        return self.discriminator_layer(inputs)

    def generator(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([X, mask], dim=1).float()
        return self.generator_layer(inputs)

    def discr_loss(
        self, X: torch.Tensor, M: torch.Tensor, H: torch.Tensor
    ) -> torch.Tensor:
        G_sample = self.generator(X, M)
        X_hat = X * M + G_sample * (1 - M)
        D_prob = self.discriminator(X_hat, H)
        return -torch.mean(
            M * torch.log(D_prob + EPS) + (1 - M) * torch.log(1.0 - D_prob + EPS)
        )

    def gen_loss(
        self, X: torch.Tensor, M: torch.Tensor, H: torch.Tensor
    ) -> torch.Tensor:
        G_sample = self.generator(X, M)
        X_hat = X * M + G_sample * (1 - M)
        D_prob = self.discriminator(X_hat, H)

        G_loss1 = -torch.mean((1 - M) * torch.log(D_prob + EPS))
        MSE_train_loss = torch.mean((M * X - M * G_sample) ** 2) / torch.mean(M)

        return G_loss1 + self.loss_alpha * MSE_train_loss


class GainImputation(TransformerMixin):
    """GAIN Imputation for static data using Generative Adversarial Nets.
    The training steps are:
     - The generato imputes the missing components conditioned on what is actually observed, and outputs a completed vector.
     - The discriminator takes a completed vector and attempts to determine which components were actually observed and which were imputed.

    Args:
        batch_size: int
            The batch size for the training steps.
        n_epochs: int
            Number of epochs for training.
        hint_rate: float
            Percentage of additional information for the discriminator.
        loss_alpha: int
            Hyperparameter for the generator loss.

    Paper: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data Imputation using Generative Adversarial Nets," ICML, 2018.
    Original code: https://github.com/jsyoon0823/GAIN
    """

    def __init__(
        self,
        batch_size: int = 256,
        n_epochs: int = 1000,
        hint_rate: float = 0.9,
        loss_alpha: float = 10,
    ) -> None:
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.hint_rate = hint_rate
        self.loss_alpha = loss_alpha
        self.norm_parameters: Union[dict, None] = None
        self.model: Union[GainModel, None] = None

    def fit(self, X: torch.Tensor) -> "GainImputation":
        """Train the GAIN model.

        Args:
            X: incomplete dataset.

        Returns:
            self: the updated model.
        """
        X = X.clone()

        # Parameters
        no = len(X)
        dim = len(X[0, :])

        # Hidden state dimensions
        h_dim = dim

        # MinMaxScaler normalization
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)

        X = X.cpu()

        for i in range(dim):
            min_val[i] = np.nanmin(X[:, i])
            X[:, i] = X[:, i] - np.nanmin(X[:, i])
            max_val[i] = np.nanmax(X[:, i])
            X[:, i] = X[:, i] / (np.nanmax(X[:, i]) + EPS)

        # Set missing
        mask = 1 - (1 * (np.isnan(X)))
        mask = mask.float().to(DEVICE)

        X = torch.nan_to_num(X)

        X = X.to(DEVICE)

        self.model = GainModel(dim, h_dim)

        D_solver = torch.optim.Adam(self.model.discriminator_layer.parameters())
        G_solver = torch.optim.Adam(self.model.generator_layer.parameters())

        def sample() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            mb_size = min(self.batch_size, no)

            mb_idx = sample_idx(no, mb_size)
            x_mb = X[mb_idx, :].clone()
            m_mb = mask[mb_idx, :].clone()

            z_mb = sample_Z(mb_size, dim)
            h_mb = sample_M(mb_size, dim, 1 - self.hint_rate)
            h_mb = m_mb * h_mb

            x_mb = m_mb * x_mb + (1 - m_mb) * z_mb

            return x_mb, h_mb, m_mb

        for it in range(self.n_epochs):
            D_solver.zero_grad()

            x_mb, h_mb, m_mb = sample()

            D_loss = self.model.discr_loss(x_mb, m_mb, h_mb)
            D_loss.backward()
            D_solver.step()

            G_solver.zero_grad()
            x_mb, h_mb, m_mb = sample()
            G_loss = self.model.gen_loss(x_mb, m_mb, h_mb)
            G_loss.backward()
            G_solver.step()

        self.norm_parameters = {"min": min_val, "max": max_val}

        return self

    def transform(self, Xmiss: torch.Tensor) -> torch.Tensor:
        """Return imputed data by trained GAIN model.

        Args:
            Xmiss: the array with missing data

        Returns:
            torch.Tensor: the array without missing data

        Raises:
            RuntimeError: if the result contains np.nans.
        """
        if self.norm_parameters is None:
            raise RuntimeError("invalid norm_parameters")
        if self.model is None:
            raise RuntimeError("Fit the model first")

        X = Xmiss.clone()

        min_val = self.norm_parameters["min"]
        max_val = self.norm_parameters["max"]

        no, dim = X.shape

        X = X.cpu()
        # MinMaxScaler normalization
        for i in range(dim):
            X[:, i] = X[:, i] - min_val[i]
            X[:, i] = X[:, i] / (max_val[i] + EPS)

        # Set missing
        mask = 1 - (1 * (np.isnan(X)))
        x = np.nan_to_num(X)

        x = torch.from_numpy(x).to(DEVICE)
        mask = mask.to(DEVICE)

        # Imputed data
        z = sample_Z(no, dim)
        x = mask * x + (1 - mask) * z

        imputed_data = self.model.generator(x, mask)

        # Renormalize
        for i in range(dim):
            imputed_data[:, i] = imputed_data[:, i] * (max_val[i] + EPS)
            imputed_data[:, i] = imputed_data[:, i] + min_val[i]

        if np.all(np.isnan(imputed_data.detach().cpu().numpy())):
            err = "The imputed result contains nan. This is a bug. Please report it on the issue tracker."
            log.critical(err)
            raise RuntimeError(err)

        mask = mask.cpu()
        imputed_data = imputed_data.detach().cpu()

        return mask * np.nan_to_num(Xmiss.cpu()) + (1 - mask) * imputed_data

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Imputes the provided dataset using the GAIN strategy.

        Args:
            X: np.ndarray
                A dataset with missing values.

        Returns:
            Xhat: The imputed dataset.
        """
        return self.fit(X).transform(X)


class GainPlugin(base.ImputerPlugin):
    """Imputation plugin for completing missing values using the GAIN strategy.

    Method:
        Details in the GainImputation class implementation.

    Example:
        >>> import numpy as np
        >>> from hyperimpute.plugins.imputers import Imputers
        >>> plugin = Imputers().get("gain")
        >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
    """

    def __init__(
        self,
        batch_size: int = 128,
        n_epochs: int = 100,
        hint_rate: float = 0.8,
        loss_alpha: int = 10,
        random_state: int = 0,
    ) -> None:
        super().__init__(random_state=random_state)

        enable_reproducible_results(random_state)

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.hint_rate = hint_rate
        self.loss_alpha = loss_alpha

        self._model = GainImputation()

    @staticmethod
    def name() -> str:
        return "gain"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Categorical("batch_size", [64, 128, 256, 512]),
            params.Integer("n_epochs", 100, 1000, 100),
            params.Float("hint_rate", 0.8, 0.99),
            params.Integer("loss_alpha", 10, 100, 10),
        ]

    @decorators.benchmark
    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "GainPlugin":
        return self

    @decorators.benchmark
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = torch.tensor(X.values).to(DEVICE)
        self._model.fit(X)
        return self._model.transform(X).detach().cpu().numpy()


plugin = GainPlugin
