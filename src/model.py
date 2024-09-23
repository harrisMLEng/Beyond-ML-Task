from abc import ABC, abstractmethod
from dataclasses import dataclass

from sklearn import datasets, neighbors


class Model(ABC):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass