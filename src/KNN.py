from dataclasses import dataclass

from numpy import ndarray
from sklearn.neighbors import KNeighborsClassifier

from src.iris_data import Iris
from src.model import Model


@dataclass
class KNN(Model):
    model : KNeighborsClassifier
    dataset : Iris
    
    def train(self) -> None:
        self.model.fit(self.dataset.X, self.dataset.y)

    def predict(self, data : ndarray):
        return self.model.predict(data)
    
