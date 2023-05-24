import pandas as pd
import numpy as np

GINI = "gini"
ENTROPY = "entropy"


class GainMixin:
    def _get_gain_function(self, criteria: str):
        if criteria == GINI:
            return self.gini_gain

        if criteria == ENTROPY:
            return self.entropy

    def gini(self, series: pd.Series):
        count = series.size
        return 1 - sum(map(lambda value: (value / count) ** 2, series.value_counts()))

    def gini_gain(self, current_gini, left_branch: pd.Series, right_branch: pd.Series):
        # higher gini gain == better split
        llen = len(left_branch)
        rlen = len(right_branch)
        total = llen + rlen
        return (
            current_gini
            - (llen / total) * self.gini(left_branch)
            - (rlen / total) * self.gini(right_branch)
        )

    def entropy(self, series: pd.Series):
        count = series.size
        return -1 * sum(
            map(
                lambda value: (value / count) * np.log2(value / count),
                series.value_counts(),
            )
        )
