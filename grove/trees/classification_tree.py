from typing import Literal

import pandas as pd

from grove.constants import Criteria
from grove.nodes import Node
from grove.trees.base_tree import BaseTree
from grove.trees.validation import TreeTestResults
from grove.utils.metrics import accuracy, confusion_matrix, f1_score, precision, recall
from grove.utils.plotting import PlottingMixin


class ClassificationTree(BaseTree, PlottingMixin):
    def __init__(
        self,
        encoding_config: pd.DataFrame,
        y_dtype: Literal["ord", "nom", "bin"],
        max_children: int,
        min_samples_per_node: int,
        criterion: str = Criteria.GINI,
        criterion_threshold: float = 1,
        max_depth: int = None,
        logging_enabled: bool = False,
        statistics_enabled: bool = False,
        consecutive_splits_on_same_feature_enabled: bool = True,
        config_values_delimiter: str = "|",
        identifier: str = "",
    ):
        super().__init__(
            encoding_config=encoding_config,
            y_dtype=y_dtype,
            max_children=max_children,
            min_samples_per_node=min_samples_per_node,
            criterion=criterion,
            criterion_threshold=criterion_threshold,
            max_depth=max_depth,
            logging_enabled=logging_enabled,
            statistics_enabled=statistics_enabled,
            consecutive_splits_on_same_feature_enabled=consecutive_splits_on_same_feature_enabled,
            config_values_delimiter=config_values_delimiter,
            identifier=identifier,
        )

    @property
    def allowed_criteria(self) -> list[Criteria]:
        return [Criteria.GINI, Criteria.CHI2]

    def _leafify_node(self, node: Node, y: pd.Series):
        """Leafify node by calculating the majority class and its probability"""
        predicted_value = y.iloc[node.indexes].mode()[0]

        node.children = []
        node.predicted_value = predicted_value

    def _build_test_results(
        self,
        labeled_data: pd.DataFrame,
        actual_column: str,
        predicted_column: str,
    ) -> TreeTestResults:
        """Build the test results."""
        test_results = TreeTestResults(labeled_data=labeled_data)

        misclassified_records = labeled_data[labeled_data[actual_column] != labeled_data[predicted_column]]

        test_results.add_metric(
            label="Misclassified Records Count",
            value=len(misclassified_records),
        )
        test_results.add_metric(
            label="Accuracy",
            value=f"{accuracy(actual=labeled_data[actual_column], predicted=labeled_data[predicted_column]):.2}",
        )
        test_results.add_metric(
            label="Precision",
            value=f"{precision(actual=labeled_data[actual_column], predicted=labeled_data[predicted_column]):.2}",
        )
        test_results.add_metric(
            label="Recall",
            value=f"{recall(actual=labeled_data[actual_column], predicted=labeled_data[predicted_column]):.2}",
        )
        test_results.add_metric(
            label="F1 Score",
            value=f"{f1_score(actual=labeled_data[actual_column], predicted=labeled_data[predicted_column]):.2}",
        )

        return test_results

    def plot(
        self,
        labeled_data: pd.DataFrame,
        actual_column: str,
        predicted_column: str,
    ):
        """Plot the test results."""
        cm = confusion_matrix(
            actual=labeled_data[actual_column],
            predicted=labeled_data[predicted_column],
        )
        self.plot_confusion_matrix(confusion_matrix=cm)
