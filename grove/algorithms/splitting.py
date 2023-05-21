from typing import Any, Iterable, Tuple
from collections import deque
from dataclasses import dataclass


import pandas as pd
import numpy as np


@dataclass
class SplitResult:
    """Placeholder for the results values of a split"""

    gain: float
    value: float
    feature: str


class SplittingMixin:
    GINI = "gini"
    ENTROPY = "entropy"

    def _get_gain_function(self, criteria: str):
        if criteria == self.GINI:
            return self.gini_gain

        if criteria == self.ENTROPY:
            return self.entropy

    def binary_split(
        self,
        dataset: pd.DataFrame,
        split_col: str,
        split_val: Any,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        left_ds = dataset[dataset[split_col] < split_val]
        right_ds = dataset[dataset[split_col] >= split_val]
        return left_ds, right_ds

    def gini(self, series: pd.Series):
        count = series.size
        return 1 - sum(map(lambda value: (value / count) ** 2, series.value_counts()))

    def gini_gain(self, current_gini, left_branch: pd.Series, right_branch: pd.Series):
        # higher gini gain == better split
        llen = len(left_branch)
        rlen = len(right_branch)
        total = llen + rlen
        return current_gini - (llen / total) * self.gini(left_branch) - (rlen / total) * self.gini(right_branch)
        pass

    def entropy(self, series: pd.Series):
        count = series.size
        return -1 * sum(
            map(
                lambda value: (value / count) * np.log2(value / count),
                series.value_counts(),
            )
        )

    def calculate_best_split(self, dataset: pd.DataFrame, features: Iterable, target: str, criteria: str):
        info_gain = self._get_gain_function(criteria)
        remaining_features = deque(features)

        split_gain = 0
        split_value = 0
        split_feature = ""

        current_gini = self.gini(dataset[target])

        if current_gini == 0:
            return SplitResult(
                gain=0,
                value=None,
                feature=None,
            )

        while remaining_features:
            feature = remaining_features.popleft()

            # iterating over only unique values
            # TODO: research non gready (more efficient) methods to do this
            for value in dataset[feature].unique():
                left, right = self.binary_split(dataset, feature, value)

                new_gain = info_gain(current_gini, left[target], right[target])

                if new_gain > split_gain:
                    split_gain = new_gain
                    split_value = value
                    split_feature = feature

        return SplitResult(
            gain=split_gain,
            value=split_value,
            feature=split_feature,
        )
