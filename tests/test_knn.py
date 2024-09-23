import pytest
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

from src.iris_data import Iris
from src.KNN import KNN


@pytest.fixture
def iris_data():
    X, y = load_iris(return_X_y=True)
    return Iris(X=X, y=y)

@pytest.fixture
def knn_model(iris_data):
    model = KNeighborsClassifier(n_neighbors=3)
    return KNN(model=model, dataset=iris_data)

def test_knn_train(knn_model):
    knn_model.train()
    assert knn_model.model is not None

def test_knn_predict(knn_model):
    knn_model.train()
    predictions = knn_model.predict([[5.1, 3.5, 1.4, 0.2]])
    assert len(predictions) == 1
