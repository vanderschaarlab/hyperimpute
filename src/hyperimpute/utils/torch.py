# third party
import numpy as np
import torch


def one_hot_encoder(arr: np.ndarray) -> torch.Tensor:
    arr = np.asarray(arr)
    n_values = np.max(arr) + 1

    result = np.eye(n_values)[arr]
    return torch.from_numpy(result).long()
