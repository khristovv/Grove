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
