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


@dataclass
class BinnedFeature:
    feature: str
    bins: list[Bin]
    stats: dict[str, npt.ArrayLike]


def parse_supervised_binning_results(binned_features: list[dict]) -> list[BinnedFeature]:
    """Parse the result of supervised binning into a list of BinnedFeature"""
    return [
        BinnedFeature(
            feature=binned_feature["cname"],
            bins=[
                Bin(
                    left_bound=bin["lb"],
                    right_bound=bin["rb"],
                    type=bin["type"],
                    size=bin["n"],
                )
                for bin in binned_feature['bns'].values()
            ],
            stats=binned_feature["st"],
        )
        for binned_feature in binned_features
    ]
