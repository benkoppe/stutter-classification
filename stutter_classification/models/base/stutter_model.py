from abc import ABC, abstractmethod
from typing import NamedTuple, Type

from sklearn.base import BaseEstimator
import numpy as np

TYPE_LABELS = [
    "NaturalPause",
    "Interjection",
    "Prolongation",
    "WordRep",
    "SoundRep",
    "Block",
]

RANDOM_STATE_DEFAULT = 42

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
    RANDOM_STATE: int = RANDOM_STATE_DEFAULT
    TEST_SIZE = 0.4

    model: BaseEstimator
    dataset: Dataset = None

    def __init__(
        self,
        model: Type[BaseEstimator],
        random_state: int = None,
    ):
        if random_state is not None:
            self.RANDOM_STATE = random_state
        self.model = model(random_state=self.RANDOM_STATE)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)

    def train(self):
        dataset = self.get_dataset()
        self.model.fit(dataset.X_train, dataset.y_train)

    def score(self):
        dataset = self.get_dataset()
        return self.model.score(dataset.X_test, dataset.y_test)

    def get_dataset(self) -> Dataset:
        if self.dataset is None:
            self.dataset = self._get_dataset()
        return self.dataset

    @abstractmethod
    def _get_dataset(self) -> Dataset:
        pass
