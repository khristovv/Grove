from collections import deque
from typing import Literal
from uuid import uuid4

import numpy as np
import numpy.typing as npt
import pandas as pd

from aislab import dp_feng

from grove.binning import Bin, parse_supervised_binning_results, BinnedFeature
from grove.constants import Criteria, SpecialChars, TreeStatistics
from grove.entities import EncodedData
from grove.validation import TestResults
from grove.nodes import Node
from grove.trees.abstract import AbstractTree
from grove.utils import first
from grove.utils.logging import Logger


class BaseTree(AbstractTree):
    def __init__(
        self,
        encoding_config: pd.DataFrame,
        y_dtype: Literal["num", "ord", "nom", "bin"],
        max_children: int,
        min_samples_per_node: int,
        criterion: str = Criteria.GINI,
        criterion_threshold: float = 1.0,
        max_depth: int = None,
        logging_enabled: bool = False,
        statistics_enabled: bool = False,
        consecutive_splits_on_same_feature_enabled: bool = True,
        config_values_delimiter: str = "|",
        identifier: str = "",
    ):
        self._validate_init(
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
        self.logger = Logger(name=self.__class__.__name__, logging_enabled=self.logging_enabled)
        self.statistics_enabled = statistics_enabled
        self.consecutive_splits_on_same_feature_enabled = consecutive_splits_on_same_feature_enabled

        self.config_values_delimiter = config_values_delimiter

        self.identifier = identifier or str(uuid4())

        self.root = None
        self.allowed_criteria = [Criteria.ALL]

        self.statistics = pd.DataFrame(columns=TreeStatistics.ALL)

    def __str__(self):
        """A method that builds a string representation of prev_split_featurethe decision tree."""

        if not self.is_training_complete:
            return "Decision tree has not been trained yet."

        lines = []

        def _next_line(node: Node, indent: str = "", is_last: bool = True):
            marker = SpecialChars.TREE_LAST_BRANCH if is_last else SpecialChars.TREE_BRANCH

            output = str(node) if node.is_root else f"{indent}{marker} {node}"
            lines.append(f"{output}\n")

            if node.is_leaf:
                return

            child_count = len(node.children)
            for index, child in enumerate(node.children):
                is_last_child = index == child_count - 1
                child_indent = indent + ("   " if is_last else f"{SpecialChars.TREE_PATH}  ")
                _next_line(node=child, indent=child_indent, is_last=is_last_child)

        _next_line(node=self.root)

        return "".join(lines)

    @property
    def is_training_complete(self) -> bool:
        """A property that returns whether the decision tree has been trained."""
        return self.root is not None

    def _validate_init(self, *, max_children: int, min_samples_per_node: int, criterion: int):
        if max_children < 2:
            raise ValueError("'max_children' must be greater than 1")

        if min_samples_per_node < 1:
            raise ValueError("'min_samples_per_node' must be greater than 0")

        if criterion not in self.allowed_criteria:
            raise ValueError(f"'criterion' must be one of {Criteria.ALL}")

    def _encode(self, x: pd.DataFrame, y: pd.DataFrame) -> EncodedData:
        self.logger.log_section("Encoding - Start")

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

        self.logger.log_section("Encoding - Complete")

        return EncodedData(
            x=encoded_x,
            xtp=xtp,
            y=y,
            ytp=self.y_dtype,
            features=cname,
            vtp=vtp,
        )

    def _bin(self, encoded_data: EncodedData, curr_node: Node) -> list[BinnedFeature]:
        self.logger.log(f"Binning node: '{curr_node}'")
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

    def _calculate_best_split(
        self,
        binned_features: list[BinnedFeature],
        prev_split_feature: str = None,
    ) -> tuple[str, list[Bin], dict[str, npt.ArrayLike]]:
        if self.consecutive_splits_on_same_feature_enabled:
            feature_with_highest_gain: BinnedFeature = max(
                binned_features, key=lambda feature: feature.get_criterion_value(criterion=self.criterion)
            )

        else:
            feature_with_highest_gain: BinnedFeature = max(
                binned_features,
                key=lambda feature: feature.get_criterion_value(criterion=self.criterion)
                if feature.label != prev_split_feature
                else -np.inf,
            )

        if feature_with_highest_gain.get_criterion_value(criterion=self.criterion) < self.criterion_threshold:
            return ("", [], {})

        valid_bins = [bin for bin in feature_with_highest_gain.bins if bin.size]

        return feature_with_highest_gain.label, valid_bins, feature_with_highest_gain.stats

    def train(self, x: pd.DataFrame, y: pd.DataFrame):
        self.logger.log_section("Training - Start", add_newline=False)

        encoded_data = self._encode(x=x, y=y)
        y_label = y.columns[0]

        self.root = Node(indexes=encoded_data.x.index, label="Root", split_variable=None)

        def _grow(node: Node, curr_depth: int = 0):
            if self.max_depth and curr_depth == self.max_depth:
                self._leafify_node(node=node, y=encoded_data.y, y_label=y_label)
                self._log_node_statistics(node=node, depth=curr_depth)
                return

            binned_features = self._bin(encoded_data=encoded_data, curr_node=node)
            feature, bins, stats = self._calculate_best_split(
                binned_features=binned_features,
                prev_split_feature=node.split_variable,
            )

            if not bins:
                self._leafify_node(node=node, y=encoded_data.y, y_label=y_label)
                self._log_node_statistics(node=node, depth=curr_depth)
                return

            for index, bin in enumerate(bins):
                curr_node_data = encoded_data.x.iloc[node.indexes]
                child_node = self._build_node(
                    bin=bin,
                    data=curr_node_data,
                    feature=feature,
                    split_stats=stats,
                    depth=curr_depth,
                    index=index,
                )
                node.add_child(child_node)

            self._log_node_statistics(node=node, depth=curr_depth)

            for child_node in node.children:
                _grow(node=child_node, curr_depth=curr_depth + 1)

        _grow(node=self.root)

        self.logger.log_section("Training - Complete")
        self.logger.log(self)

        if self.statistics_enabled:
            self.logger.log_section("Statistics")
            self.logger.log(self.get_statistics())

    def _leafify_node(self, node: Node, y: pd.DataFrame, y_label: str):
        """Each tree implmentation should provide its own logic to leafify a node"""
        raise NotImplementedError

    def _build_node_label(self, feature: str, bin: Bin) -> str:
        if bin.is_categorical:
            return f"{feature} {SpecialChars.ELEMENT_OF}  [{', '.join(str(v) for v in bin.bounds)}]"

        lb, rb = bin.bounds
        lb = lb if not np.isinf(lb) else ""
        rb = rb if not np.isinf(rb) else ""

        if lb and rb:
            return f"{lb} <= {feature} < {rb}"

        if lb:
            return f"{feature} >= {lb}"

        return f"{feature} < {rb}"

    def _build_node(
        self,
        bin: Bin,
        data: pd.DataFrame,
        feature: str,
        split_stats: dict[str, npt.ArrayLike],
        depth: int,
        index: int,
    ) -> Node:
        if bin.is_categorical:
            indexes = data[data[feature].isin(bin.bounds)].index
        else:
            indexes = data[data[feature].between(*bin.bounds, inclusive="left")].index

        return Node(
            indexes=indexes,
            coordinates=(depth, index),
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

    def _get_prediction(self, row: pd.Series):
        curr_node = self.root
        nodes = deque(curr_node.children)

        while nodes:
            child_node = nodes.popleft()
            column = child_node.split_variable
            value = row[column]

            if value in child_node:
                nodes.clear()
                nodes.extend(child_node.children)
                curr_node = child_node

        return curr_node.predicted_value

    def predict(self, x: pd.DataFrame, y_label: str, return_y_only=False) -> pd.DataFrame | pd.Series:
        """Label a new dataset."""
        if not self.is_training_complete:
            raise Exception("Model is not trained yet.")

        encoded_data = self._encode(x=x, y=pd.DataFrame()).x
        # keep the original indexes
        encoded_data.set_index(x.index, inplace=True)

        y = encoded_data.apply(self._get_prediction, axis=1)
        if return_y_only:
            return y

        labeled_data = x.copy()
        labeled_data[y_label] = y

        return labeled_data

    def _get_misclassified_values(
        self,
        labeled_data: pd.DataFrame,
        actual_column: str,
        predicted_column: str,
    ) -> pd.Series:
        """Get the misclassified values."""
        raise NotImplementedError

    def test(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        save_results: bool = False,
        output_dir: str = None,
    ):
        """Test the model on a test dataset."""
        self.logger.log_section("Testing", add_newline=False)

        y_label = y.columns[0]
        predicted_column = f"PREDICTED_{y_label}"
        actual_column = f"ACTUAL_{y_label}"

        labeled_data = self.predict(x=x, y_label=predicted_column)
        labeled_data[actual_column] = y

        misclassifed_values = self._get_misclassified_values(
            labeled_data=labeled_data,
            actual_column=actual_column,
            predicted_column=predicted_column,
        )
        misclassifed_values_count = misclassifed_values.value_counts()[True]
        misclassification_error = misclassifed_values_count / len(labeled_data)

        test_results = TestResults(
            labeled_data=labeled_data,
            tree_statistics=self.get_statistics(),
            misclassification_error=misclassification_error,
            misclassified_indexes=labeled_data[misclassifed_values].index,
        )

        if save_results:
            test_results.save(output_dir=output_dir)

        self.logger.log_section("Test Results:")
        self.logger.log(test_results)

        return test_results
