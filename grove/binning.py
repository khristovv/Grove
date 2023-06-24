from dataclasses import dataclass
from typing import Literal

import numpy.typing as npt

from aislab import dp_feng


unsupervised_binning = dp_feng.ubng
supervised_binning = dp_feng.sbng


@dataclass
class Bin:
    """Placeholder for a bin"""

    left_bound: npt.ArrayLike
    right_bound: npt.ArrayLike
    type: Literal["Normal", "Special Values", "Missing", "Other"]
    size: int

    @property
    def bounds(self) -> list[float, float]:
        if self.is_categorical:
            return self.left_bound

        return [self.left_bound[0], self.right_bound[0]]

    @property
    def is_categorical(self) -> bool:
        return len(self.left_bound) > 1


@dataclass
class BinnedFeature:
    label: str
    bins: list[Bin]
    stats: dict[str, npt.ArrayLike]

    def get_criterion_value(self, criterion: str) -> float | None:
        return self.stats.get(criterion, [None])[0]


def parse_supervised_binning_results(binned_features: list[dict]) -> list[BinnedFeature]:
    """Parse the result of supervised binning into a list of BinnedFeature"""
    return [
        BinnedFeature(
            label=binned_feature["cname"],
            bins=[
                Bin(
                    left_bound=bin["lb"],
                    right_bound=bin["rb"],
                    type=bin["type"],
                    size=bin["n"],
                )
                for bin in binned_feature["bns"].values()
            ],
            stats=binned_feature["st"][0],
        )
        for binned_feature in binned_features
    ]
