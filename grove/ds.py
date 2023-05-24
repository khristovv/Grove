from dataclasses import dataclass


@dataclass
class SplitResult:
    """Placeholder for the results values of a split"""

    gain: float
    value: float
    feature: str
