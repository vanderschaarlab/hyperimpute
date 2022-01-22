"""
Original code: https://rmisstastic.netlify.app/how-to/python/generate_html/how%20to%20generate%20missing%20values
"""

# stdlib
from typing import List

# third party
import numpy as np
from scipy import optimize
from scipy.special import expit


def pick_coeffs(
    X: np.ndarray,
    idxs_obs: List[int] = [],
    idxs_nas: List[int] = [],
    self_mask: bool = False,
) -> np.ndarray:
    n, d = X.shape
    if self_mask:
        coeffs = np.random.rand(d)
        Wx = X * coeffs
        coeffs /= np.std(Wx, 0)
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = np.random.rand(d_obs, d_na)
        Wx = X[:, idxs_obs] @ coeffs
        coeffs /= np.std(Wx, 0, keepdims=True)
    return coeffs


def fit_intercepts(
    X: np.ndarray, coeffs: np.ndarray, p: float, self_mask: bool = False
) -> np.ndarray:
    if self_mask:
        d = len(coeffs)
        intercepts = np.zeros(d)
        for j in range(d):

            def f(x: np.ndarray) -> np.ndarray:
                return expit(X * coeffs[j] + x).mean().item() - p

            intercepts[j] = optimize.bisect(f, -50, 50)
    else:
        d_obs, d_na = coeffs.shape
        intercepts = np.zeros(d_na)
        for j in range(d_na):

            def f(x: np.ndarray) -> np.ndarray:
                return expit(np.dot(X, coeffs[:, j]) + x).mean().item() - p

            intercepts[j] = optimize.bisect(f, -50, 50)
    return intercepts


def MAR_mask(
    X: np.ndarray,
    p: float,
    p_obs: float,
    sample_columns: bool = True,
) -> np.ndarray:
    """
    Missing at random mechanism with a logistic masking model. First, a subset of variables with *no* missing values is
    randomly selected. The remaining variables have missing values according to a logistic model with random weights,
    re-scaled so as to attain the desired proportion of missing values on those variables.

    Args:
        X : Data for which missing values will be simulated.
        p : Proportion of missing values to generate for variables which will have missing values.
        p_obs : Proportion of variables with *no* missing values that will be used for the logistic masking model.

    Returns:
        mask : Mask of generated missing values (True if the value is missing).

    """

    n, d = X.shape

    mask = np.zeros((n, d)).astype(bool)

    d_obs = max(
        int(p_obs * d), 1
    )  # number of variables that will have no missing values (at least one variable)
    d_na = d - d_obs  # number of variables that will have missing values

    # Sample variables that will all be observed, and those with missing values:
    if sample_columns:
        idxs_obs = np.random.choice(d, d_obs, replace=False)
    else:
        idxs_obs = list(range(d_obs))

    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    # Other variables will have NA proportions that depend on those observed variables, through a logistic model
    # The parameters of this logistic model are random.

    # Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_obs, idxs_nas)
    # Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_obs], coeffs, p)

    ps = expit(X[:, idxs_obs] @ coeffs + intercepts)

    ber = np.random.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    return mask


def MNAR_mask_logistic(
    X: np.ndarray, p: float, p_params: float = 0.3, exclude_inputs: bool = True
) -> np.ndarray:
    """
    Missing not at random mechanism with a logistic masking model. It implements two mechanisms:
    (i) Missing probabilities are selected with a logistic model, taking all variables as inputs. Hence, values that are
    inputs can also be missing.
    (ii) Variables are split into a set of intputs for a logistic model, and a set whose missing probabilities are
    determined by the logistic model. Then inputs are then masked MCAR (hence, missing values from the second set will
    depend on masked values.
    In either case, weights are random and the intercept is selected to attain the desired proportion of missing values.

    Args:
        X : Data for which missing values will be simulated.
        p : Proportion of missing values to generate for variables which will have missing values.
        p_params : Proportion of variables that will be used for the logistic masking model (only if exclude_inputs).
        exclude_inputs : True: mechanism (ii) is used, False: (i)

    Returns:
        mask : Mask of generated missing values (True if the value is missing).

    """

    n, d = X.shape

    mask = np.zeros((n, d)).astype(bool)

    d_params = (
        max(int(p_params * d), 1) if exclude_inputs else d
    )  # number of variables used as inputs (at least 1)
    d_na = (
        d - d_params if exclude_inputs else d
    )  # number of variables masked with the logistic model

    # Sample variables that will be parameters for the logistic regression:
    idxs_params = (
        np.random.choice(d, d_params, replace=False) if exclude_inputs else np.arange(d)
    )
    idxs_nas = (
        np.array([i for i in range(d) if i not in idxs_params])
        if exclude_inputs
        else np.arange(d)
    )

    # Other variables will have NA proportions selected by a logistic model
    # The parameters of this logistic model are random.

    # Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_params, idxs_nas)
    # Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_params], coeffs, p)

    ps = expit(X[:, idxs_params] @ coeffs + intercepts)

    ber = np.random.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    # If the inputs of the logistic model are excluded from MNAR missingness,
    # mask some values used in the logistic model at random.
    # This makes the missingness of other variables potentially dependent on masked values

    if exclude_inputs:
        mask[:, idxs_params] = np.random.rand(n, d_params) < p

    return mask


def MNAR_self_mask_logistic(X: np.ndarray, p: float) -> np.ndarray:
    """
    Missing not at random mechanism with a logistic self-masking model. Variables have missing values probabilities
    given by a logistic model, taking the same variable as input (hence, missingness is independent from one variable
    to another). The intercepts are selected to attain the desired missing rate.

    Args:
        X : Data for which missing values will be simulated.
        p : Proportion of missing values to generate for variables which will have missing values.

    Returns:
        mask : Mask of generated missing values (True if the value is missing).

    """

    n, d = X.shape

    # Variables will have NA proportions that depend on those observed variables, through a logistic model
    # The parameters of this logistic model are random.

    # Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, self_mask=True)
    # Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X, coeffs, p, self_mask=True)

    ps = expit(X * coeffs + intercepts)

    ber = np.random.rand(n, d)
    mask = ber < ps

    return mask


def MNAR_mask_quantiles(
    X: np.ndarray,
    p: float,
    q: float,
    p_params: float,
    cut: str = "both",
    MCAR: bool = False,
) -> np.ndarray:
    """
    Missing not at random mechanism with quantile censorship. First, a subset of variables which will have missing
    variables is randomly selected. Then, missing values are generated on the q-quantiles at random. Since
    missingness depends on quantile information, it depends on masked values, hence this is a MNAR mechanism.

    Args:
        X : Data for which missing values will be simulated.
        p : Proportion of missing values to generate for variables which will have missing values.
        q : Quantile level at which the cuts should occur
        p_params : Proportion of variables that will have missing values
        cut : 'both', 'upper' or 'lower'. Where the cut should be applied. For instance, if q=0.25 and cut='upper', then missing values will be generated in the upper quartiles of selected variables.
        MCAR : If true, masks variables that were not selected for quantile censorship with a MCAR mechanism.

    Returns:
        mask : Mask of generated missing values (True if the value is missing).

    """
    n, d = X.shape

    mask = np.zeros((n, d)).astype(bool)

    d_na = max(int(p_params * d), 1)  # number of variables that will have NMAR values

    # Sample variables that will have imps at the extremes
    idxs_na = np.random.choice(
        d, d_na, replace=False
    )  # select at least one variable with missing values

    # check if values are greater/smaller that corresponding quantiles
    if cut == "upper":
        quants = np.quantile(X[:, idxs_na], 1 - q, dim=0)
        m = X[:, idxs_na] >= quants
    elif cut == "lower":
        quants = np.quantile(X[:, idxs_na], q, dim=0)
        m = X[:, idxs_na] <= quants
    elif cut == "both":
        u_quants = np.quantile(X[:, idxs_na], 1 - q, axis=0)
        l_quants = np.quantile(X[:, idxs_na], q, axis=0)
        m = (X[:, idxs_na] <= l_quants) | (X[:, idxs_na] >= u_quants)

    # Hide some values exceeding quantiles
    ber = np.random.rand(n, d_na)
    mask[:, idxs_na] = (ber < p) & m

    if MCAR:
        # Add a mcar mecanism on top
        mask = mask | (np.random.rand(n, d) < p)

    return mask


def simulate_nan(
    X: np.ndarray,
    p_miss: float,
    mecha: str = "MCAR",
    opt: str = "logistic",
    p_obs: float = 0.5,
    q: float = 0,
    sample_columns: bool = True,
) -> dict:
    """
    Generate missing values for specifics missing-data mechanism and proportion of missing values.

    Args:
        X : Data for which missing values will be simulated.
        p_miss : Proportion of missing values to generate for variables which will have missing values.
        mecha : Indicates the missing-data mechanism to be used. "MCAR" by default, "MAR", "MNAR" or "MNARsmask"
        opt: For mecha = "MNAR", it indicates how the missing-data mechanism is generated: using a logistic regression ("logistic"), a quantile censorship ("quantile")  or logistic regression for generating a self-masked MNAR mechanism ("selfmasked").
        p_obs : If mecha = "MAR", or mecha = "MNAR" with opt = "logistic" or "quantile", proportion of variables with *no* missing values that will be used for the logistic masking model.
        q : If mecha = "MNAR" and opt = "quanti", quantile level at which the cuts should occur.

    Returns:
        A dictionnary containing:
            - 'X_init': the initial data matrix.
            - 'X_incomp': the data with the generated missing values.
            - 'mask': a matrix indexing the generated missing values.
    """

    if mecha == "MAR":
        mask = MAR_mask(X, p_miss, p_obs, sample_columns=sample_columns).astype(float)
    elif mecha == "MNAR" and opt == "logistic":
        mask = MNAR_mask_logistic(X, p_miss, p_obs).astype(float)
    elif mecha == "MNAR" and opt == "quantile":
        mask = MNAR_mask_quantiles(X, p_miss, q, 1 - p_obs).astype(float)
    elif mecha == "MNAR" and opt == "selfmasked":
        mask = MNAR_self_mask_logistic(X, p_miss).astype(float)
    else:
        mask = (np.random.rand(*X.shape) < p_miss).astype(float)

    X_nas = X.copy()
    X_nas[mask.astype(bool)] = np.nan

    return {"X_init": X.astype(float), "X_incomp": X_nas.astype(float), "mask": mask}
