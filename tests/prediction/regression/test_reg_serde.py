# third party
import pytest
from sklearn.datasets import load_iris

# hyperimpute absolute
from hyperimpute.plugins import Predictions
from hyperimpute.utils.serialization import load, save


@pytest.mark.parametrize("plugin", Predictions(category="regression").list())
def test_pickle(plugin: str) -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)

    estimator = Predictions(category="regression").get(plugin)

    buff = save(estimator)
    estimator_new = load(buff)

    estimator.fit(X, y)

    buff = save(estimator)
    estimator_new = load(buff)

    estimator_new.predict(X)
