from collections import deque
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from aislab import dp_feng

from grove.binning import Bin, parse_supervised_binning_results, BinnedFeature
from grove.constants import Criteria, SpecialChars, TreeStatistics
from grove.ds import EncodedData, TestResults
from grove.nodes import Node

from grove.trees.base_tree import BaseTree
from grove.utils import first


class NTree(BaseTree):
    def __init__(
        self,
        encoding_config: pd.DataFrame,
        max_children: int,
        min_samples_per_node: int,
        criterion: str = Criteria.GINI,
        y_dtype: Literal["num", "ord", "nom", "bin"] = "bin",
        criterion_threshold: float = 1.0,
        max_depth: int = None,
        logging_enabled: bool = False,
        statistics_enabled: bool = False,
        config_values_delimiter: str = "|",
    ):
        self.validate_init(
            max_children=max_children,
            min_samples_per_node=min_samples_per_node,
            criterion=criterion.capitalize(),
        )

        super().__init__()
        self.encoding_config = encoding_config
        self.y_dtype = y_dtype
        self.criterion = criterion.capitalize()
        self.criterion_threshold = criterion_threshold
        self.max_children = max_children
        self.max_depth = max_depth
        self.min_samples_per_node = min_samples_per_node
        self.logging_enabled = logging_enabled
        self.statistics_enabled = statistics_enabled
        self.config_values_delimiter = config_values_delimiter

        self.statistics = pd.DataFrame(columns=TreeStatistics.ALL)

    def validate_init(self, *, max_children: int, min_samples_per_node: int, criterion: int):
        if max_children < 2:
            raise ValueError("'max_children' must be greater than 1")

        if min_samples_per_node < 1:
            raise ValueError("'min_samples_per_node' must be greater than 0")

        if criterion not in Criteria.ALL:
            raise ValueError(f"'criterion' must be one of {Criteria.ALL}")

    def encode(self, x: pd.DataFrame, y: pd.DataFrame) -> EncodedData:
        cname = self.encoding_config["cname"].tolist()
        x = x[cname]
        xtp = self.encoding_config["xtp"]
        vtp = self.encoding_config["vtp"]
        order = self.encoding_config["order"]
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
            y=y,
            ytp=self.y_dtype,
            features=cname,
            vtp=vtp,
        )

    def bin(self, encoded_data: EncodedData, curr_node: Node) -> list[BinnedFeature]:
        rows_to_include = curr_node.indexes

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

    def calculate_best_split(
        self, binned_features: list[BinnedFeature]
    ) -> tuple[str, list[Bin], dict[str, npt.ArrayLike]]:
        feature_with_highest_gain: BinnedFeature = max(
            binned_features, key=lambda feature: feature.get_criterion_value(criterion=self.criterion)
        )

        if feature_with_highest_gain.get_criterion_value(criterion=self.criterion) < self.criterion_threshold:
            return ("", [], {})

        valid_bins = [bin for bin in feature_with_highest_gain.bins if bin.size]

        return feature_with_highest_gain.label, valid_bins, feature_with_highest_gain.stats

    def train(self, x: pd.DataFrame, y: pd.DataFrame):
        encoded_data = self.encode(x=x, y=y)
        y_label = y.columns[0]

        self.root = Node(indexes=encoded_data.x.index, label="Root", split_variable=None)

        def _grow(node: Node, curr_depth: int = 0):
            if self.max_depth and curr_depth == self.max_depth:
                self._leaify_node(node=node, y=encoded_data.y, y_label=y_label)
                self._log_node_statistics(node=node, depth=curr_depth)
                return

            binned_features = self.bin(encoded_data=encoded_data, curr_node=node)
            feature, bins, stats = self.calculate_best_split(binned_features=binned_features)

            if not bins:
                self._leaify_node(node=node, y=encoded_data.y, y_label=y_label)
                self._log_node_statistics(node=node, depth=curr_depth)
                return

            for bin in bins:
                curr_node_data = encoded_data.x.iloc[node.indexes]
                child_node = self._build_node(bin=bin, data=curr_node_data, feature=feature, split_stats=stats)
                node.add_child(child_node)

            self._log_node_statistics(node=node, depth=curr_depth)

            for child_node in node.children:
                _grow(node=child_node, curr_depth=curr_depth + 1)

        _grow(node=self.root)

    def _leaify_node(self, node: Node, y: pd.DataFrame, y_label: str):
        class_label = y.iloc[node.indexes][y_label].mode()[0]

        node.children = []
        node.class_label = class_label

    def _build_node_label(self, feature: str, bin: Bin) -> str:
        if bin.is_categorical:
            return f"( {feature} {SpecialChars.ELEMENT_OF}  [{', '.join(str(v) for v in bin.values)}] )"

        lb, rb = bin.bounds
        lb = lb if not np.isinf(lb) else ""
        rb = rb if not np.isinf(rb) else ""

        if lb and rb:
            return f"( {lb} <= {feature} < {rb} )"

        if lb:
            return f"( {lb} <= {feature} )"

        return f"( {feature} < {rb} )"

    def _build_node(self, bin: Bin, data: pd.DataFrame, feature: str, split_stats: dict[str, npt.ArrayLike]) -> Node:
        if bin.is_categorical:
            indexes = data[data[feature].isin(bin.bounds)].index
        else:
            indexes = data[data[feature].between(*bin.bounds, inclusive="left")].index

        return Node(
            indexes=indexes,
            label=self._build_node_label(feature=feature, bin=bin),
            split_variable=feature,
            split_variable_type=Node.CATEGORICAL if bin.is_categorical else Node.NUMERICAL,
            bounds=bin.bounds,
            split_stats=split_stats,
        )

    def _log_node_statistics(self, node: Node, depth: int):
        if not self.statistics_enabled:
            return

        split_stats = node.split_stats

        self.statistics = pd.concat(
            [
                self.statistics,
                pd.DataFrame(
                    {
                        TreeStatistics.LABEL: [node.label],
                        TreeStatistics.DEPTH: [depth],
                        TreeStatistics.SPLIT_FEATURE: [node.split_variable],
                        TreeStatistics.CHILDREN: [len(node.children)],
                        TreeStatistics.SIZE: [len(node.indexes)],
                        TreeStatistics.MY0: first(split_stats.get("my0", [])),
                        TreeStatistics.MY1: first(split_stats.get("my1", [])),
                        self.criterion: split_stats.get(self.criterion),
                    }
                ),
            ],
            ignore_index=True,
        )

    def get_statistics(self) -> pd.DataFrame:
        return self.statistics.sort_values(by=[TreeStatistics.DEPTH])

    def classify(self, data: pd.DataFrame, y_label: str):
        """Classify a new dataset."""
        labeled_data = data.copy()
        labeled_data[y_label] = None

        encoded_data = self.encode(x=data, y=pd.DataFrame()).x
        # keep the original indexes
        encoded_data.set_index(data.index, inplace=True)

        for idx, row in encoded_data.iterrows():
            curr_node = self.root
            children = deque(curr_node.children)

            while children:
                child_node = children.popleft()
                column = child_node.split_variable
                value = row[column]

                if value in child_node:
                    children.clear()
                    children.extend(child_node.children)
                    curr_node = child_node

            labeled_data.at[idx, y_label] = curr_node.class_label

        return labeled_data

    def test(self, x: pd.DataFrame, y: pd.DataFrame):
        """Test the model on a test dataset."""
        y_label = y.columns[0]
        predicted_column = f"PREDICTED_{y_label}"
        actual_column = f"ACTUAL_{y_label}"

        labeled_data = self.classify(data=x, y_label=actual_column)
        labeled_data[predicted_column] = y

        missclassifed_values = labeled_data[actual_column] != labeled_data[predicted_column]
        missclassifed_values_count = missclassifed_values.value_counts()[True]
        missclassification_error = missclassifed_values_count / len(labeled_data)

        test_results = TestResults(
            labeled_data=labeled_data,
            missclassification_error=missclassification_error,
            missclassified_indexes=labeled_data[missclassifed_values].index,
        )
        return test_results
