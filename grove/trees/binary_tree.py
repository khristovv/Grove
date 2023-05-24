from collections import deque
from typing import Iterable, Any

import pandas as pd

from grove.nodes import BinaryNode
from grove.trees.base_tree import BaseTree
from grove.ds import SplitResult
from grove.utils.gain import GINI


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
        self.criteria = criteria or GINI
        self.root = BinaryNode(data=self.dataset, label="root")

    def calculate_best_split(
        self, dataset: pd.DataFrame, features: Iterable, target: str, criteria: str
    ):
        info_gain = self._get_gain_function(criteria)
        remaining_features = deque(features)

        split_gain = 0
        split_value = 0
        split_feature = ""

        current_gini = self.gini(dataset[target])

        if current_gini == 0:
            return SplitResult(
                gain=0,
                value=None,
                feature=None,
            )

        while remaining_features:
            feature = remaining_features.popleft()

            # iterating over only unique values
            # TODO: research non gready (more efficient) methods to do this
            for value in dataset[feature].unique():
                left, right = self.binary_split(dataset, feature, value)

                new_gain = info_gain(current_gini, left[target], right[target])

                if new_gain > split_gain:
                    split_gain = new_gain
                    split_value = value
                    split_feature = feature

        return SplitResult(
            gain=split_gain,
            value=split_value,
            feature=split_feature,
        )

    def binary_split(
        self,
        dataset: pd.DataFrame,
        split_col: str,
        split_val: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        left_ds = dataset[dataset[split_col] < split_val]
        right_ds = dataset[dataset[split_col] >= split_val]
        return left_ds, right_ds

    def build(self):
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
