from typing import Any, Iterable, Tuple
from collections import deque
from dataclasses import dataclass


import pandas as pd
import numpy as np


@dataclass
class SplitResult:
    """Placeholder for the results values of a split"""
    score: float
    value: float
    feature: str


class SplittingMixin:
    GINI = 'gini'
    ENTROPY = 'entropy'


    def _get_criteria_algo(self, criteria: str):
        if criteria == self.GINI:
            return self.gini

        if criteria == self.ENTROPY:
            return self.entropy

    def binary_split(
        self,
        dataset: pd.DataFrame,
        split_col: str,
        split_val: Any
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        left_ds = dataset[dataset[split_col] < split_val]
        right_ds = dataset[dataset[split_col] >= split_val]
        return left_ds, right_ds

    def gini(self, series: pd.Series):
        count = series.size
        return 1 - sum(
            map(
                lambda value: (value / count) ** 2,
                series.value_counts()
            )
        )

    def gini_index(self):
        # TODO: implement
        pass

    def entropy(self, series: pd.Series):
        count = series.size
        return -1 * sum(
            map(
                lambda value: (value/count) * np.log2(value/count),
                series.value_counts()
            )
        )

    def calculate_best_split(self, dataset: pd.DataFrame, features: Iterable, target: str, criteria: str):
        evaluate_purity = self._get_criteria_algo(criteria)
        remaining_features = deque(features)

        split_score = 999
        split_value = 0
        split_feature = ''

        if self.gini(dataset[target]) == 0:
            return SplitResult(
                score=0,
                value=None,
                feature=None,
            )


        while remaining_features:
            feature = remaining_features.popleft()

            # iterating over only unique values
            # TODO: research non gready (more efficient) methods to do this
            for value in dataset[feature].unique():
                left, right = self.binary_split(dataset, feature, value)

                left_score = evaluate_purity(left[target])
                right_score = evaluate_purity(right[target])

                new_score = min(split_score, left_score, right_score)

                if new_score < split_score:
                    split_score = new_score
                    split_value = value
                    split_feature = feature

        return SplitResult(
            score=split_score,
            value=split_value,
            feature=split_feature
        )
