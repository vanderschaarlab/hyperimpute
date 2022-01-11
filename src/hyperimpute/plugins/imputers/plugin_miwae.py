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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weights_init(layer: Any) -> None:
    if type(layer) == nn.Linear:
        torch.nn.init.orthogonal_(layer.weight)


class MIWAEPlugin(base.ImputerPlugin):
    def __init__(self, n_epochs: int = 2000, **kwargs: Any) -> None:
        super().__init__()

        self.n_epochs = n_epochs
        self.bs = 64  # batch size
        self.h = 128  # number of hidden units in (same for all MLPs)
        self.d = 1  # dimension of the latent space
        self.K = 20  # number of IS during training

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
                loc=out_encoder[..., : self.d],
                scale=torch.nn.Softplus()(out_encoder[..., self.d : (2 * self.d)]),
            ),
            1,
        )

        zgivenx = q_zgivenxobs.rsample([self.K])
        zgivenx_flat = zgivenx.reshape([self.K * batch_size, self.d])

        out_decoder = self.decoder(zgivenx_flat)
        all_means_obs_model = out_decoder[..., :p]
        all_scales_obs_model = (
            torch.nn.Softplus()(out_decoder[..., p : (2 * p)]) + 0.001
        )
        all_degfreedom_obs_model = (
            torch.nn.Softplus()(out_decoder[..., (2 * p) : (3 * p)]) + 3
        )

        data_flat = torch.Tensor.repeat(iota_x, [self.K, 1]).reshape([-1, 1])
        tiledmask = torch.Tensor.repeat(mask, [self.K, 1])

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
                loc=out_encoder[..., : self.d],
                scale=torch.nn.Softplus()(out_encoder[..., self.d : (2 * self.d)]),
            ),
            1,
        )

        zgivenx = q_zgivenxobs.rsample([L])
        zgivenx_flat = zgivenx.reshape([L * batch_size, self.d])

        out_decoder = self.decoder(zgivenx_flat)
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
        X = np.asarray(X)

        mask = np.isfinite(X)

        xhat_0 = np.copy(X)
        xhat_0[np.isnan(X)] = 0

        n = X.shape[0]  # number of observations
        p = X.shape[1]  # number of features

        self.p_z = td.Independent(
            td.Normal(
                loc=torch.zeros(self.d).to(DEVICE), scale=torch.ones(self.d).to(DEVICE)
            ),
            1,
        )

        self.decoder = nn.Sequential(
            torch.nn.Linear(self.d, self.h),
            torch.nn.ReLU(),
            torch.nn.Linear(self.h, self.h),
            torch.nn.ReLU(),
            torch.nn.Linear(
                self.h, 3 * p
            ),  # the decoder will output both the mean, the scale, and the number of degrees of freedoms (hence the 3*p)
        ).to(DEVICE)

        self.encoder = nn.Sequential(
            torch.nn.Linear(p, self.h),
            torch.nn.ReLU(),
            torch.nn.Linear(self.h, self.h),
            torch.nn.ReLU(),
            torch.nn.Linear(
                self.h, 2 * self.d
            ),  # the encoder will output both the mean and the diagonal covariance
        ).to(DEVICE)

        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=1e-3
        )

        xhat = np.copy(xhat_0)  # This will be out imputed data matrix

        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)

        bs = min(self.bs, n)

        for ep in range(1, self.n_epochs):
            perm = np.random.permutation(
                n
            )  # We use the "random reshuffling" version of SGD
            batches_data = np.array_split(
                xhat_0[
                    perm,
                ],
                n / bs,
            )
            batches_mask = np.array_split(
                mask[
                    perm,
                ],
                n / bs,
            )
            for it in range(len(batches_data)):
                optimizer.zero_grad()
                self.encoder.zero_grad()
                self.decoder.zero_grad()
                b_data = torch.from_numpy(batches_data[it]).float().to(DEVICE)
                b_mask = torch.from_numpy(batches_mask[it]).float().to(DEVICE)
                loss = self._miwae_loss(iota_x=b_data, mask=b_mask)
                loss.backward()
                optimizer.step()
            if ep % 100 == 1:
                log.debug(f"Epoch {ep}")
                log.debug(
                    "MIWAE likelihood bound  %g"
                    % (
                        -np.log(self.K)
                        - self._miwae_loss(
                            iota_x=torch.from_numpy(xhat_0).float().to(DEVICE),
                            mask=torch.from_numpy(mask).float().to(DEVICE),
                        )
                        .cpu()
                        .data.numpy()
                    )
                )  # Gradient step

                # Now we do the imputation

                xhat[~mask] = (
                    self._miwae_impute(
                        iota_x=torch.from_numpy(xhat_0).float().to(DEVICE),
                        mask=torch.from_numpy(mask).float().to(DEVICE),
                        L=10,
                    )
                    .cpu()
                    .data.numpy()[~mask]
                )

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = np.asarray(X)
        mask = np.isfinite(X)

        xhat = np.copy(X)
        xhat[np.isnan(X)] = 0

        xhat[~mask] = (
            self._miwae_impute(
                iota_x=torch.from_numpy(xhat).float().to(DEVICE),
                mask=torch.from_numpy(mask).float().to(DEVICE),
                L=10,
            )
            .cpu()
            .data.numpy()[~mask]
        )

        return xhat


plugin = MIWAEPlugin
