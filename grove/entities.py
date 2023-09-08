from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass
class EncodedData:
    """Placeholder for the encoded dataset"""

    x: pd.DataFrame
    xtp: pd.Series
    y: pd.Series
    ytp: str
    features: Iterable[str]


@dataclass
class SplitResult:
    """Placeholder for the results values of a split"""

    gain: float
    value: float
    feature: str


@dataclass
class Coordinates:
    """Placeholder for the coordinates of a node"""

    depth: int
    index: int

    def __str__(self):
        return f"[{self.depth},{self.index}]"

    @property
    def as_tuple(self):
        return self.depth, self.index
