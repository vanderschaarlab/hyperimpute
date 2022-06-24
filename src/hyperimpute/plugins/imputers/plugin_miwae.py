# stdlib
from typing import Any, List

# third party
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.distributions as td

# hyperimpute absolute
import hyperimpute.logger as log
import hyperimpute.plugins.core.params as params
import hyperimpute.plugins.imputers.base as base
from hyperimpute.utils.distributions import enable_reproducible_results

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weights_init(layer: Any) -> None:
    if type(layer) == nn.Linear:
        torch.nn.init.orthogonal_(layer.weight)


class MIWAEPlugin(base.ImputerPlugin):
    """MIWAE imputation plugin

    Args:
        n_epochs: int
            Number of training iterations
        batch_size: int
            Batch size
        latent_size: int
            dimension of the latent space
        n_hidden: int
            number of hidden units
        K: int
            number of IS during training
        random_state: int
            random seed


    Example:
        >>> import numpy as np
        >>> from hyperimpute.plugins.imputers import Imputers
        >>> plugin = Imputers().get("miwae")
        >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])


    Reference: "MIWAE: Deep Generative Modelling and Imputation of Incomplete Data", Pierre-Alexandre Mattei, Jes Frellsen
    Original code: https://github.com/pamattei/miwae
    """

    def __init__(
        self,
        n_epochs: int = 500,
        batch_size: int = 256,
        latent_size: int = 1,
        n_hidden: int = 1,
        random_state: int = 0,
        K: int = 20,
    ) -> None:
        super().__init__(random_state=random_state)

        enable_reproducible_results(random_state)

        self.n_epochs = n_epochs
        self.batch_size = batch_size  # batch size
        self.n_hidden = n_hidden  # number of hidden units in (same for all MLPs)
        self.latent_size = latent_size  # dimension of the latent space
        self.K = K  # number of IS during training

    @staticmethod
    def name() -> str:
        return "miwae"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return []

    def _miwae_loss(self, iota_x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size = iota_x.shape[0]
        p = iota_x.shape[1]

        out_encoder = self.encoder(iota_x)
        q_zgivenxobs = td.Independent(
            td.Normal(
                loc=out_encoder[..., : self.latent_size],
                scale=torch.nn.Softplus()(
                    out_encoder[..., self.latent_size : (2 * self.latent_size)]
                ),
            ),
            1,
        )

        zgivenx = q_zgivenxobs.rsample([self.K])
        zgivenx_flat = zgivenx.reshape([self.K * batch_size, self.latent_size])

        out_decoder = self.latent_sizeecoder(zgivenx_flat)
        all_means_obs_model = out_decoder[..., :p]
        all_scales_obs_model = (
            torch.nn.Softplus()(out_decoder[..., p : (2 * p)]) + 0.001
        )
        all_degfreedom_obs_model = (
            torch.nn.Softplus()(out_decoder[..., (2 * p) : (3 * p)]) + 3
        )

        data_flat = torch.Tensor.repeat(iota_x, [self.K, 1]).reshape([-1, 1]).to(DEVICE)
        tiledmask = torch.Tensor.repeat(mask, [self.K, 1]).to(DEVICE)

        all_log_pxgivenz_flat = torch.distributions.StudentT(
            loc=all_means_obs_model.reshape([-1, 1]),
            scale=all_scales_obs_model.reshape([-1, 1]),
            df=all_degfreedom_obs_model.reshape([-1, 1]),
        ).log_prob(data_flat)
        all_log_pxgivenz = all_log_pxgivenz_flat.reshape([self.K * batch_size, p])

        logpxobsgivenz = torch.sum(all_log_pxgivenz * tiledmask, 1).reshape(
            [self.K, batch_size]
        )
        logpz = self.p_z.log_prob(zgivenx.to(DEVICE))
        logq = q_zgivenxobs.log_prob(zgivenx)

        neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq, 0))

        return neg_bound

    def _miwae_impute(
        self, iota_x: torch.Tensor, mask: torch.Tensor, L: int
    ) -> torch.Tensor:
        batch_size = iota_x.shape[0]
        p = iota_x.shape[1]

        out_encoder = self.encoder(iota_x)
        q_zgivenxobs = td.Independent(
            td.Normal(
                loc=out_encoder[..., : self.latent_size],
                scale=torch.nn.Softplus()(
                    out_encoder[..., self.latent_size : (2 * self.latent_size)]
                ),
            ),
            1,
        )

        zgivenx = q_zgivenxobs.rsample([L])
        zgivenx_flat = zgivenx.reshape([L * batch_size, self.latent_size])

        out_decoder = self.latent_sizeecoder(zgivenx_flat)
        all_means_obs_model = out_decoder[..., :p]
        all_scales_obs_model = (
            torch.nn.Softplus()(out_decoder[..., p : (2 * p)]) + 0.001
        )
        all_degfreedom_obs_model = (
            torch.nn.Softplus()(out_decoder[..., (2 * p) : (3 * p)]) + 3
        )

        data_flat = torch.Tensor.repeat(iota_x, [L, 1]).reshape([-1, 1]).to(DEVICE)
        tiledmask = torch.Tensor.repeat(mask, [L, 1]).to(DEVICE)

        all_log_pxgivenz_flat = torch.distributions.StudentT(
            loc=all_means_obs_model.reshape([-1, 1]),
            scale=all_scales_obs_model.reshape([-1, 1]),
            df=all_degfreedom_obs_model.reshape([-1, 1]),
        ).log_prob(data_flat)
        all_log_pxgivenz = all_log_pxgivenz_flat.reshape([L * batch_size, p])

        logpxobsgivenz = torch.sum(all_log_pxgivenz * tiledmask, 1).reshape(
            [L, batch_size]
        )
        logpz = self.p_z.log_prob(zgivenx)
        logq = q_zgivenxobs.log_prob(zgivenx)

        xgivenz = td.Independent(
            td.StudentT(
                loc=all_means_obs_model,
                scale=all_scales_obs_model,
                df=all_degfreedom_obs_model,
            ),
            1,
        )

        imp_weights = torch.nn.functional.softmax(
            logpxobsgivenz + logpz - logq, 0
        )  # these are w_1,....,w_L for all observations in the batch
        xms = xgivenz.sample().reshape([L, batch_size, p])
        xm = torch.einsum("ki,kij->ij", imp_weights, xms)

        return xm

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "MIWAEPlugin":
        X = torch.from_numpy(np.asarray(X)).float().to(DEVICE)
        mask = np.isfinite(X.cpu()).bool().to(DEVICE)

        xhat_0 = torch.clone(X)

        xhat_0[np.isnan(X.cpu()).bool()] = 0

        n = X.shape[0]  # number of observations
        p = X.shape[1]  # number of features

        self.p_z = td.Independent(
            td.Normal(
                loc=torch.zeros(self.latent_size).to(DEVICE),
                scale=torch.ones(self.latent_size).to(DEVICE),
            ),
            1,
        )

        self.latent_sizeecoder = nn.Sequential(
            torch.nn.Linear(self.latent_size, self.n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_hidden, self.n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(
                self.n_hidden, 3 * p
            ),  # the decoder will output both the mean, the scale, and the number of degrees of freedoms (hence the 3*p)
        ).to(DEVICE)

        self.encoder = nn.Sequential(
            torch.nn.Linear(p, self.n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_hidden, self.n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(
                self.n_hidden, 2 * self.latent_size
            ),  # the encoder will output both the mean and the diagonal covariance
        ).to(DEVICE)

        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.latent_sizeecoder.parameters()),
            lr=1e-3,
        )

        xhat = torch.clone(xhat_0)  # This will be out imputed data matrix

        self.encoder.apply(weights_init)
        self.latent_sizeecoder.apply(weights_init)

        bs = min(self.batch_size, n)

        for ep in range(1, self.n_epochs):
            perm = np.random.permutation(
                n
            )  # We use the "random reshuffling" version of SGD
            batches_data = np.array_split(
                xhat_0[
                    perm,
                ],
                int(n / bs),
            )
            batches_mask = np.array_split(
                mask[
                    perm,
                ],
                int(n / bs),
            )
            for it in range(len(batches_data)):
                optimizer.zero_grad()
                self.encoder.zero_grad()
                self.latent_sizeecoder.zero_grad()
                b_data = batches_data[it]
                b_mask = batches_mask[it].float()
                loss = self._miwae_loss(iota_x=b_data, mask=b_mask)
                loss.backward()
                optimizer.step()
            if ep % 100 == 1:
                log.debug(f"Epoch {ep}")
                log.debug(
                    "MIWAE likelihood bound  %g"
                    % (
                        -np.log(self.K)
                        - self._miwae_loss(iota_x=xhat_0, mask=mask).cpu().data.numpy()
                    )
                )  # Gradient step

                # Now we do the imputation

                xhat[~mask] = self._miwae_impute(
                    iota_x=xhat_0,
                    mask=mask,
                    L=10,
                )[~mask]

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = torch.from_numpy(np.asarray(X)).float().to(DEVICE)
        mask = np.isfinite(X.cpu()).bool().to(DEVICE)

        xhat = torch.clone(X)
        xhat[np.isnan(X.cpu()).bool()] = 0

        xhat[~mask] = self._miwae_impute(
            iota_x=xhat,
            mask=mask,
            L=10,
        )[~mask]

        return xhat.detach().cpu().numpy()


plugin = MIWAEPlugin
