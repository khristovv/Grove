from typing import Literal

import pandas as pd

from grove.constants import Criteria
from grove.nodes import Node
from grove.trees.base_tree import BaseTree


class ClassificationTree(BaseTree):
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
        config_values_delimiter: str = "|",
    ):
        self.allowed_criteria = [Criteria.GINI, Criteria.CHI2]
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
            config_values_delimiter=config_values_delimiter,
        )

    def _get_misclassified_values(
        self,
        labeled_data: pd.DataFrame,
        actual_column: str,
        predicted_column: str,
    ) -> pd.Series:
        """Get the misclassified values."""
        return labeled_data[actual_column] != labeled_data[predicted_column]

    def _leafify_node(self, node: Node, y: pd.DataFrame, y_label: str):
        """Leafify node by calculating the majority class and its probability"""
        class_label = y.iloc[node.indexes][y_label].mode()[0]

        node.children = []
        node.class_label = class_label
