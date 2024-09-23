from dataclasses import dataclass

from numpy import ndarray
from sklearn.neighbors import KNeighborsClassifier

from src.iris_data import Iris
from src.model import Model


@dataclass
class KNN(Model):
    """
    A class used to represent a K-Nearest Neighbors model.

    Attributes
    ----------
    model : KNeighborsClassifier
        The K-Nearest Neighbors classifier.
    dataset : Iris
        The Iris dataset to be used for training and prediction.

    Methods
    -------
    train() -> None
        Trains the K-Nearest Neighbors model using the Iris dataset.
    predict(data : ndarray)
        Predicts the class labels for the given data using the trained model.
    """
    model : KNeighborsClassifier
    dataset : Iris
    
    def train(self) -> None:
        """
        Trains the K-Nearest Neighbors model using the Iris dataset.
        """
        self.model.fit(self.dataset.X, self.dataset.y)

    def predict(self, data : ndarray):
        """
        Predicts the class labels for the given data using the trained model.

        Parameters
        ----------
        data : ndarray
            The input data for which the class labels need to be predicted.

        Returns
        -------
        ndarray
            The predicted class labels for the input data.
        """
        return self.model.predict(data)
