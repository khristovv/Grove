from typing import Iterable

import numpy as np
import pandas as pd
from aislab import dp_feng

from grove.binning import Bin, parse_supervised_binning_results, BinnedFeature
from grove.constants import Criteria
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
        criterion: str = Criteria.GINI,
        criterion_threshold: float = 1.0,
        max_depth: int = None,
        logging_enabled: bool = False,
        config_values_delimiter: str = "|",
    ):
        self.validate_init(
            max_children=max_children,
            min_samples_per_node=min_samples_per_node,
            criterion=criterion.capitalize(),
        )

        super().__init__(dataset, target, features, max_depth)
        self.criterion = criterion.capitalize()
        self.criterion_threshold = criterion_threshold
        self.max_children = max_children
        self.config = config
        self.min_samples_per_node = min_samples_per_node
        self.logging_enabled = logging_enabled
        self.config_values_delimiter = config_values_delimiter

    def validate_init(self, *, max_children: int, min_samples_per_node: int, criterion: int):
        if max_children < 2:
            raise ValueError("'max_children' must be greater than 1")

        if min_samples_per_node < 1:
            raise ValueError("'min_samples_per_node' must be greater than 0")

        if criterion not in Criteria.ALL:
            raise ValueError(f"'criterion' must be one of {Criteria.ALL}")

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

    def bin(self, encoded_data: EncodedData, curr_node: Node) -> list[BinnedFeature]:
        rows_to_include = curr_node.data.index

        # run unsupervised binning
        unsupervised_binning_results = dp_feng.ubng(
            x=encoded_data.x.iloc[rows_to_include],
            xtp=encoded_data.xtp,
            y=encoded_data.y.iloc[rows_to_include].values,
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

    def calculate_best_split(self, binned_features: list[BinnedFeature]) -> tuple[str, list[Bin]]:
        feature_with_highest_gain: BinnedFeature = max(
            binned_features, key=lambda feature: feature.get_criterion_value(criterion=self.criterion)
        )

        if feature_with_highest_gain.get_criterion_value(criterion=self.criterion) < self.criterion_threshold:
            return ("", [])

        valid_bins = [bin for bin in feature_with_highest_gain.bins if bin.size]

        return feature_with_highest_gain.label, valid_bins

    def build(self):
        encoded_data = self.encode()

        self.root = Node(data=encoded_data.x, label="Root")

        def _build(node: Node, curr_depth: int):
            if self.max_depth and curr_depth == self.max_depth:
                # node.leafify()
                return

            binned_features = self.bin(encoded_data=encoded_data, curr_node=node)
            feature, bins = self.calculate_best_split(binned_features=binned_features)

            if not bins:
                # node.leafify()
                return

            for bin in bins:
                child_node = self._build_node(bin=bin, data=node.data, feature=feature)
                node.add_child(child_node)

            for child_node in node.children:
                _build(node=child_node, curr_depth=curr_depth + 1)

        _build(node=self.root, curr_depth=0)

    def _build_node_label(self, feature: str, bin: Bin) -> str:
        if bin.is_discrete:
            return f"{feature} \u2208  [{', '.join(str(v) for v in bin.values)}]"

        lb, rb = bin.bounds
        lb = lb if not np.isinf(lb) else "-∞"
        rb = rb if not np.isinf(rb) else "∞"

        return f"({lb} <= {feature} < {rb})"

    def _build_node(self, bin: Bin, data: pd.DataFrame, feature: str):
        if bin.is_discrete:
            records = data[data[feature].isin(bin.values)]
        else:
            records = data[data[feature].between(*bin.bounds, inclusive="left")]

        return Node(
            data=records,
            label=self._build_node_label(feature=feature, bin=bin),
        )

    def classify(self):
        """Classify a new dataset."""
        # TODO: implement
        pass

    def test(self):
        """Test the model on a test dataset."""
        # TODO: implement
        pass
