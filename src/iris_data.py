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
    dataset : Iris

    def __init__(self, dataset : Iris):
        self.dataset = dataset
        super().__init__()

    @cached_property
    def preprocess(self):
        return super().fit(self.dataset.X).transform(self.dataset.X)
