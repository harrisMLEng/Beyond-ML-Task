from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Tuple

from numpy import ndarray
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch


@dataclass
class Iris:
    X : ndarray
    y : ndarray

class Preprocess(StandardScaler):
    """
    A class used to preprocess the Iris dataset using StandardScaler.

    Attributes
    ----------
    dataset : Iris
        The Iris dataset to be preprocessed.

    Methods
    -------
    preprocess:
        Preprocesses the dataset by fitting and transforming it using StandardScaler.
    """
    dataset : Iris

    def __init__(self, dataset : Iris):
        """
        Parameters
        ----------
        dataset : Iris
            The Iris dataset to be preprocessed.
        """
        self.dataset = dataset
        super().__init__()

    @cached_property
    def preprocess(self):
        """
        Preprocesses the dataset by fitting and transforming it using StandardScaler.

        Returns
        -------
        ndarray
            The preprocessed dataset.
        """
        return super().fit(self.dataset.X).transform(self.dataset.X)
