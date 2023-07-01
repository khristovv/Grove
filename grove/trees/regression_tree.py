import pandas as pd

from grove.constants import Criteria
from grove.nodes import Node
from grove.trees.base_tree import BaseTree


class RegressionTree(BaseTree):
    def __init__(
        self,
        encoding_config: pd.DataFrame,
        max_children: int,
        min_samples_per_node: int,
        allowed_diff: float,
        criterion_threshold: float = 1,
        max_depth: int = None,
        logging_enabled: bool = False,
        statistics_enabled: bool = False,
        config_values_delimiter: str = "|",
    ):
        self.allowed_criteria = [Criteria.F]
        self.allowed_diff = allowed_diff
        super().__init__(
            encoding_config=encoding_config,
            y_dtype="num",
            max_children=max_children,
            min_samples_per_node=min_samples_per_node,
            criterion=Criteria.F,
            criterion_threshold=criterion_threshold,
            max_depth=max_depth,
            logging_enabled=logging_enabled,
            statistics_enabled=statistics_enabled,
            config_values_delimiter=config_values_delimiter,
        )

    def _get_misclassified_values(
        self,
        labeled_data: pd.DataFrame,
        actual_column: str,
        predicted_column: str,
    ) -> pd.Series:
        """Get the misclassified values."""
        diff = labeled_data[actual_column] - labeled_data[predicted_column]
        abs_diff = diff.abs()

        return abs_diff > self.allowed_diff

    def _leafify_node(self, node: Node, y: pd.DataFrame, y_label: str):
        """Leafify node by calculating the mean of the target variable"""
        predicted_value = y.iloc[node.indexes][y_label].mean()

        node.children = []
        node.class_label = predicted_value