from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass
class EncodedData:
    """Placeholder for the encoded dataset"""

    x: pd.DataFrame
    xtp: pd.Series
    y: pd.DataFrame
    ytp: str
    features: Iterable[str]
    vtp: pd.Series  # ?


@dataclass
class SplitResult:
    """Placeholder for the results values of a split"""

    gain: float
    value: float
    feature: str


@dataclass
class TestResults:
    labeled_data: pd.DataFrame
    misclassification_error: float
    misclassified_indexes: pd.Series

    def __str__(self) -> str:
        return (
            f"Test Results:\n"
            f"  Misclassification rate: {self.misclassification_error_perc:.2f}%\n"
            f"  Misclassified indexes: {', '.join(str(v) for v in self.misclassified_indexes.values)}\n"
            f"  Accuracy: {100 - self.misclassification_error_perc:.2f}%\n"
        )

    @property
    def misclassification_error_perc(self):
        return self.misclassification_error * 100
