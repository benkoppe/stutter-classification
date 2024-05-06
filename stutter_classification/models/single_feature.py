from typing import Type
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator

from stutter_classification.models.base.stutter_model import (
    Dataset,
    StutterModel,
    TYPE_LABELS,
)
from stutter_classification.data.sep28k_data import get_sep28k_mfcc_df


class SingleFeatureModel(StutterModel):
    def __init__(
        self,
        model: Type[BaseEstimator],
        target_column: str,
        filter: bool = True,
        filter_extreme_cases: bool = False,
        random_state: int = None,
        n_mfccs: int = 13,
    ):
        super().__init__(model, random_state=random_state, n_mfccs=n_mfccs)
        self.target_column = target_column
        self.filter_extreme_cases = filter_extreme_cases
        self.filter = filter

    def _get_dataset(self) -> Dataset:
        df = get_sep28k_mfcc_df(n_mfccs=self.n_mfccs)
        if self.filter:
            df = self.filter_columns_except_target(df, self.target_column)

        # ensure target column is binary
        df.loc[df[self.target_column] >= 1.0, self.target_column] = 1.0

        # get feature columns
        X = df.iloc[:, (-1 * self.n_mfccs) :]

        # get target column
        y = df[self.target_column]

        # split into train and test data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.TEST_SIZE, random_state=self.RANDOM_STATE
        )

        return Dataset(X_train, X_test, y_train, y_test)

    def filter_columns_except_target(self, df, target_column):
        if self.filter_extreme_cases:
            df = df[df["NoStutteredWords"] != 0]

        for column in TYPE_LABELS:
            if column != target_column:
                df = df[df[column] == 0]

        return df
