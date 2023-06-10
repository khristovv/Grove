from typing import Iterable

# import numpy as np
import pandas as pd
from aislab import dp_feng

from grove.binning import parse_supervised_binning_results, BinnedFeature
from grove.ds import EncodedData
from grove.nodes import Node

from grove.trees.base_tree import BaseTree


class NTree(BaseTree):
    def __init__(
        self,
        dataset: pd.DataFrame,  # X
        target: pd.DataFrame,  # y
        features: Iterable[str],
        config: pd.DataFrame,
        max_children: int,
        min_samples_per_node: int,
        max_depth: int = None,  # TODO: make required
        logging_enabled: bool = False,
        config_values_delimiter: str = "|",
    ):
        self.validate_init(max_children=max_children, min_samples_per_node=min_samples_per_node)

        super().__init__(dataset, target, features, max_depth)
        self.max_children = max_children
        self.config = config
        self.min_samples_per_node = min_samples_per_node
        self.logging_enabled = logging_enabled
        self.config_values_delimiter = config_values_delimiter

        self.root = Node(data=self.dataset, label="Root")

    def validate_init(self, *, max_children: int, min_samples_per_node: int):
        if max_children < 2:
            raise ValueError("max_children must be greater than 1")

        if min_samples_per_node < 1:
            raise ValueError("min_samples_per_node must be greater than 0")

    def encode(self) -> EncodedData:
        cname = self.config["cname"].tolist()
        x = self.dataset[cname]
        xtp = self.config["xtp"]
        vtp = self.config["vtp"]
        order = self.config["order"]
        dlm = self.config_values_delimiter

        encoded_x = dp_feng.enc_int(
            x=x,
            cname=cname,
            xtp=xtp,
            vtp=vtp,
            order=order,
            dlm=dlm,
            dsp=self.logging_enabled,
        )

        return EncodedData(
            x=encoded_x,
            xtp=xtp,
            y=self.target,
            ytp="bin",
            features=cname,
            vtp=vtp,
        )

    def bin(self, encoded_data: EncodedData) -> list[BinnedFeature]:
        # run unsupervised binning
        unsupervised_binning_results = dp_feng.ubng(
            x=encoded_data.x,
            xtp=encoded_data.xtp,
            y=encoded_data.y.values,
            ytp=encoded_data.ytp,
            cnames=encoded_data.features,
            md=self.max_children,
            nmin=self.min_samples_per_node,
            dsp=self.logging_enabled,
        )

        # run supervised binning
        supervised_binning_results = dp_feng.sbng(
            bng=unsupervised_binning_results, md=self.max_children, dsp=self.logging_enabled
        )

        return parse_supervised_binning_results(binned_features=supervised_binning_results)

    def build(self):
        encoded_data = self.encode()

        def _build(
            dataset: pd.DataFrame,
            target: pd.Series,
            features: Iterable[str],
            encoded_data: EncodedData,
            node: Node = None,
        ):
            binned_features = self.bin(encoded_data=encoded_data)
            print("ubng", binned_features)

        _build(
            dataset=self.dataset,
            target=self.features,
            features=self.features,
            encoded_data=encoded_data,
            node=self.root,
        )

    def classify(self):
        """Classify a new dataset."""
        # TODO: implement
        pass

    def test(self):
        """Test the model on a test dataset."""
        # TODO: implement
        pass
