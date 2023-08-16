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
        allowed_diff: float = None,
        criterion_threshold: float = 1,
        max_depth: int = None,
        logging_enabled: bool = False,
        statistics_enabled: bool = False,
        consecutive_splits_on_same_feature_enabled: bool = True,
        config_values_delimiter: str = "|",
        identifier: str = "",
    ):
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
            consecutive_splits_on_same_feature_enabled=consecutive_splits_on_same_feature_enabled,
            config_values_delimiter=config_values_delimiter,
            identifier=identifier,
        )

    @property
    def allowed_criteria(self) -> list[Criteria]:
        return [Criteria.F]

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
        node.predicted_value = predicted_value

    def test(self, x: pd.DataFrame, y: pd.DataFrame, save_results: bool = False, output_dir: str = None):
        if self.allowed_diff is None:
            raise ValueError("The 'allowed_diff' parameter must be set to use the RegressionTree.test method.")

        return super().test(x, y, save_results, output_dir)
