# stdlib
from typing import Any

# third party
from sklearn.calibration import CalibratedClassifierCV

# hyperimpute absolute
from hyperimpute.utils.parallel import cpu_count

calibrations = ["none", "sigmoid", "isotonic"]


def calibrated_model(model: Any, calibration: int = 1, **kwargs: Any) -> Any:
    assert calibration < len(calibrations), "invalid calibration value"

    if not hasattr(model, "predict_proba"):
        return CalibratedClassifierCV(base_estimator=model, n_jobs=cpu_count())

    if calibration != 0:
        return CalibratedClassifierCV(
            base_estimator=model,
            method=calibrations[calibration],
            n_jobs=cpu_count(),
        )

    return model
