# stdlib
from typing import Any, List

# third party
from geomloss import SamplesLoss
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
import torch

# hyperimpute absolute
from hyperimpute.plugins.core.device import DEVICE
import hyperimpute.plugins.core.params as params
import hyperimpute.plugins.imputers.base as base
import hyperimpute.plugins.utils.decorators as decorators
from hyperimpute.utils.distributions import enable_reproducible_results


class SinkhornImputation(TransformerMixin):
    """Sinkhorn imputation can be used to impute quantitative data and it relies on the idea that two batches extracted randomly from the same dataset should share the same distribution and consists in minimizing optimal transport distances between batches.

    Args:
        eps: float, default=0.01
            Sinkhorn regularization parameter.
        lr : float, default = 0.01
            Learning rate.
        opt: torch.nn.optim.Optimizer, default=torch.optim.Adam
            Optimizer class to use for fitting.
        n_epochs : int, default=15
            Number of gradient updates for each model within a cycle.
        batch_size : int, defatul=256
            Size of the batches on which the sinkhorn divergence is evaluated.
        n_pairs : int, default=10
            Number of batch pairs used per gradient update.
        noise : float, default = 0.1
            Noise used for the missing values initialization.
        scaling: float, default=0.9
            Scaling parameter in Sinkhorn iterations

    Example:
        >>> import numpy as np
        >>> from hyperimpute.plugins.imputers import Imputers
        >>> plugin = Imputers().get("sinkhorn")
        >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])


    Reference: "Missing Data Imputation using Optimal Transport", Boris Muzellec, Julie Josse, Claire Boyer, Marco Cuturi
    Original code: https://github.com/BorisMuzellec/MissingDataOT
    """

    def __init__(
        self,
        eps: float = 0.01,
        lr: float = 1e-3,
        opt: Any = torch.optim.Adam,
        n_epochs: int = 500,
        batch_size: int = 256,
        n_pairs: int = 1,
        noise: float = 1e-2,
        scaling: float = 0.9,
    ):
        self.eps = eps
        self.lr = lr
        self.opt = opt
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_pairs = n_pairs
        self.noise = noise
        self.sk = SamplesLoss(
            "sinkhorn", p=2, blur=eps, scaling=scaling, backend="tensorized"
        )

    def fit_transform(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        X = torch.tensor(X.values).to(DEVICE)
        X = X.clone()
        n, d = X.shape

        if self.batch_size > n // 2:
            e = int(np.log2(n // 2))
            self.batch_size = 2**e

        mask = torch.isnan(X).double().cpu()
        imps = (self.noise * torch.randn(mask.shape).double() + np.nanmean(X.cpu(), 0))[
            mask.bool()
        ]
        imps = imps.to(DEVICE)
        mask = mask.to(DEVICE)
        imps.requires_grad = True

        optimizer = self.opt([imps], lr=self.lr)

        for i in range(self.n_epochs):
            X_filled = X.detach().clone()
            X_filled[mask.bool()] = imps
            loss: SamplesLoss = 0

            for _ in range(self.n_pairs):

                idx1 = np.random.choice(n, self.batch_size, replace=False)
                idx2 = np.random.choice(n, self.batch_size, replace=False)

                X1 = X_filled[idx1]
                X2 = X_filled[idx2]

                loss = loss + self.sk(X1, X2)

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                # Catch numerical errors/overflows (should not happen)
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        X_filled = X.detach().clone()
        X_filled[mask.bool()] = imps

        return X_filled.detach().cpu().numpy()


class SinkhornPlugin(base.ImputerPlugin):
    """Imputation plugin for completing missing values using the Sinkhorn strategy.

    Method:
        Details in the SinkhornImputation class implementation.

    Example:
        >>> import numpy as np
        >>> from hyperimpute.plugins.imputers import Imputers
        >>> plugin = Imputers().get("sinkhorn")
        >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
                  0         1         2         3
        0  1.000000  1.000000  1.000000  1.000000
        1  1.404637  1.651113  1.651093  1.404638
        2  1.000000  2.000000  2.000000  1.000000
        3  2.000000  2.000000  2.000000  2.000000
    """

    def __init__(
        self,
        eps: float = 0.01,
        lr: float = 1e-2,
        opt: Any = torch.optim.Adam,
        n_epochs: int = 500,
        batch_size: int = 512,
        n_pairs: int = 1,
        noise: float = 1e-2,
        scaling: float = 0.9,
        random_state: int = 0,
    ) -> None:
        super().__init__(random_state=random_state)

        enable_reproducible_results(random_state)

        self.eps = eps
        self.lr = lr
        self.opt = opt
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_pairs = n_pairs
        self.noise = noise
        self.scaling = scaling

        self._model = SinkhornImputation(
            eps=eps,
            lr=lr,
            opt=opt,
            n_epochs=n_epochs,
            batch_size=batch_size,
            n_pairs=n_pairs,
            noise=noise,
            scaling=scaling,
        )

    @staticmethod
    def name() -> str:
        return "sinkhorn"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Float("eps", 1e-3, 1e-2),
            params.Categorical("lr", [1e-2, 1e-3]),
            params.Integer("n_epochs", 100, 500, 100),
            params.Integer("batch_size", 100, 200, 100),
            params.Categorical("noise", [1e-2, 1e-3, 1e-4]),
            params.Float("scaling", 0.8, 0.99),
        ]

    @decorators.benchmark
    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "SinkhornPlugin":
        return self

    @decorators.benchmark
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._model.fit_transform(X)


plugin = SinkhornPlugin
