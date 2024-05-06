from typing import Type
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator

from stutter_classification.models.base.stutter_model import (
    Dataset,
    StutterModel,
    TYPE_LABELS,
)
from stutter_classification.data.sep28k_data import get_sep28k_mfcc_df


class AllFeaturesModel(StutterModel):
    def __init__(
        self, model: Type[BaseEstimator], random_state: int = None, n_mfccs=13
    ):
        super().__init__(model, random_state=random_state, n_mfccs=n_mfccs)

    def _get_dataset(self) -> Dataset:
        df = get_sep28k_mfcc_df(n_mfccs=self.n_mfccs)

        # get feature columns
        X = df.iloc[:, (-1 * self.n_mfccs) :]

        # get target columns
        y = df[TYPE_LABELS]

        # ensure that only one target column is selected
        y = y.idxmax(axis=1)

        # split into train and test data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.TEST_SIZE, random_state=self.RANDOM_STATE
        )

        return Dataset(X_train, X_test, y_train, y_test)
