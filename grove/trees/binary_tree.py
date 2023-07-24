# type: ignore

from collections import deque
from typing import Iterable, Any

import pandas as pd

from grove.nodes import BinaryNode
from grove.trees.abstract import BaseTree
from grove.entities import SplitResult


# WARNING: Do not use! This was the initial attempt for a simple tree implementation.
# TODO: reimplement using a similar approach to the NTree class
class BinaryTree(BaseTree):
    def __init__(
        self,
        dataset: pd.DataFrame,  # X
        target: pd.Series,  # y
        features: Iterable[str],
        max_depth: int = 5,
    ):
        super().__init__(dataset=dataset, target=target, features=features, max_depth=max_depth)
        self.root = BinaryNode(data=self.dataset, label="root")

    def calculate_best_split(self, dataset: pd.DataFrame, target: pd.Series, features: Iterable[str]):
        remaining_features = deque(features)

        split_gain = 0
        split_value = 0
        split_feature = ""

        current_gini = self.gini(target)

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
                left, right = self.binary_split(dataset=dataset, split_on=feature, threshold=value)

                left_subset = target[target.index.isin(left.index)]
                right_subset = target[target.index.isin(right.index)]
                new_gain = self.gini_gain(current_gini, left_subset, right_subset)

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
        split_on: str,
        threshold: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        left_ds = dataset[dataset[split_on] < threshold]
        right_ds = dataset[dataset[split_on] >= threshold]
        return left_ds, right_ds

    def train(self):
        def _build(dataset: pd.DataFrame, target: pd.Series, features: Iterable[str], node: BinaryNode = None):
            split_result = self.calculate_best_split(dataset=dataset, target=target, features=features)

            if split_result.gain == 0:
                # this is a leaf Node
                node.leafify(f"{target.name} - {target.unique()[0]}")
                return node

            left_ds, right_ds = self.binary_split(
                dataset=dataset,
                split_on=split_result.feature,
                threshold=split_result.value,
            )

            left_target = target[target.index.isin(left_ds.index)]
            right_target = target[target.index.isin(right_ds.index)]

            left_child = BinaryNode(data=left_ds, label=f"{split_result.feature} < {split_result.value} ")
            right_child = BinaryNode(data=right_ds, label=f"{split_result.feature} >= {split_result.value} ")
            node.add_child(left_child)
            node.add_child(right_child)

            _build(dataset=left_ds, target=left_target, features=self.features, node=left_child)
            _build(dataset=right_ds, target=right_target, features=self.features, node=right_child)

        _build(dataset=self.dataset, target=self.target, features=self.features, node=self.root)

    def fit(self):
        # TODO implement
        pass
