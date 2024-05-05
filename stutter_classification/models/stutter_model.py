from abc import ABC, abstractmethod
from typing import NamedTuple

from sklearn.base import BaseEstimator
import numpy as np

from 

Dataset = NamedTuple(
    "Dataset",
    [
        ("X_train", np.ndarray),
        ("X_test", np.ndarray),
        ("y_train", np.ndarray),
        ("y_test", np.ndarray),
    ],
)

class StutterModel(ABC):
    RANDOM_STATE = 42
    
    model: BaseEstimator
    
    def __init__(self, model: BaseEstimator):
        self.model = model
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)
    
    @abstractmethod
    def train():
        pass
    
    def get_sep28k_dataset():
        