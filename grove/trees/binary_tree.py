from typing import Iterable, Any

import pandas as pd

from grove.nodes import BinaryNode
from grove.trees.base_tree import BaseTree


class BinaryTree(BaseTree):
    def __init__(
        self,
        dataset: pd.DataFrame,
        features: Iterable[str],
        target: str,
        max_depth: int = 5,
        criteria: str = None,
    ):
        super().__init__(dataset, features, target, max_depth=max_depth)
        self.criteria = criteria or self.GINI
        self.root = BinaryNode(data=self.dataset, label="root")

    def binary_split(
        self,
        dataset: pd.DataFrame,
        split_col: str,
        split_val: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        left_ds = dataset[dataset[split_col] < split_val]
        right_ds = dataset[dataset[split_col] >= split_val]
        return left_ds, right_ds

    def train(self):
        def _build(dataset, features, node: BinaryNode = None):
            split_result = self.calculate_best_split(
                dataset, features, self.target, self.criteria
            )

            if split_result.gain == 0:
                # this is a leaf Node
                node.leafify(str(dataset[self.target].unique()[0]))
                return node

            left_ds, right_ds = self.binary_split(
                dataset,
                split_result.feature,
                split_result.value,
            )

            left_child = BinaryNode(
                data=left_ds, label=f"{split_result.feature} < {split_result.value} "
            )
            right_child = BinaryNode(
                data=right_ds, label=f"{split_result.feature} >= {split_result.value} "
            )
            node.add_child(left_child)
            node.add_child(right_child)

            _build(left_ds, self.features, left_child)
            _build(right_ds, self.features, right_child)

        _build(self.dataset, self.features, self.root)

    def fit(self):
        # TODO implement
        pass
