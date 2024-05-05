from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator
import numpy as np

class StutterModel(ABC):
    model: BaseEstimator
    
    def __init__(self, model: BaseEstimator):
        self.model = model
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)
    
    @abstractmethod
    def train():
        pass
    
    def get_sep28k_df():
        