import pytest
from sklearn.datasets import load_iris

from src.iris_data import Iris, Preprocess


@pytest.fixture
def iris_data():
    X, y = load_iris(return_X_y=True)
    return Iris(X=X, y=y)

def test_iris_data(iris_data):
    assert iris_data.X is not None
    assert iris_data.y is not None
    assert len(iris_data.X) == len(iris_data.y)

def test_preprocess(iris_data):
    preprocess = Preprocess(dataset=iris_data)
    processed_data = preprocess.preprocess
    assert processed_data is not None
    assert processed_data.shape == iris_data.X.shape