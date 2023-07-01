from typing import Literal

import pandas as pd

from grove.constants import Criteria
from grove.nodes import Node
from grove.trees.base_tree import BaseTree


class ClassificationTree(BaseTree):
    def __init__(
        self,
        encoding_config: pd.DataFrame,
        max_children: int,
        min_samples_per_node: int,
        criterion: str = Criteria.GINI,
        y_dtype: Literal["num", "ord", "nom", "bin"] = "bin",
        criterion_threshold: float = 1,
        max_depth: int = None,
        logging_enabled: bool = False,
        statistics_enabled: bool = False,
        config_values_delimiter: str = "|",
    ):
        super().__init__(
            encoding_config,
            max_children,
            min_samples_per_node,
            criterion,
            y_dtype,
            criterion_threshold,
            max_depth,
            logging_enabled,
            statistics_enabled,
            config_values_delimiter,
        )

    def _leafify_node(self, node: Node, y: pd.DataFrame, y_label: str):
        """Leafify node by calculating the majority class and its probability"""
        class_label = y.iloc[node.indexes][y_label].mode()[0]

        node.children = []
        node.class_label = class_label
