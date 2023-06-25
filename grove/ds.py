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
    missclassification_error: float
    missclassified_indexes: pd.Series

    def __str__(self) -> str:
        return (
            f"Test Results:\n"
            f"  Missclassification error: {self.missclassification_error_perc:.2f}%\n"
            f"  Missclassified indexes: {', '.join(str(v) for v in self.missclassified_indexes.values)}\n"
        )

    @property
    def missclassification_error_perc(self):
        return self.missclassification_error * 100
