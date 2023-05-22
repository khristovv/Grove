from typing import Iterable

import pandas as pd

from grove.algorithms.splitting import SplittingMixin
from grove.datastructures.nodes import BinaryNode


class AbstractTree:
    def __init__(
        self,
        dataset: pd.DataFrame,
        features: Iterable[str],
        target: str,
        max_depth: int = None,
    ):
        self.dataset = dataset
        self.features = features
        self.target = target
        self.max_depth = max_depth

    def build(self):
        raise NotImplementedError


class BaseTree(AbstractTree, SplittingMixin):
    def print(self):
        def _print(node: BinaryNode, indent: str = "", is_last: bool = True):
            marker = "└──" if is_last else "├──"

            output = str(node) if node.is_root else f"{indent}{marker} {node}"
            print(output)

            if node.is_leaf:
                return

            child_count = len(node.children)
            for index, child in enumerate(node.children):
                is_last_child = index == child_count - 1
                child_indent = indent + ("   " if is_last else "│  ")
                _print(node=child, indent=child_indent, is_last=is_last_child)

        print()
        _print(self.root)


class DecisionTree(BaseTree):
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

    def train(self):
        def _build(dataset, features, node: BinaryNode = None):
            split_result = self.calculate_best_split(dataset, features, self.target, self.criteria)

            if split_result.gain == 0:
                # this is a leaf Node
                node.leafify(str(dataset[self.target].unique()[0]))
                return node

            left_ds, right_ds = self.binary_split(
                dataset,
                split_result.feature,
                split_result.value,
            )

            left_child = BinaryNode(data=left_ds, label=f"{split_result.feature} < {split_result.value} ")
            right_child = BinaryNode(data=right_ds, label=f"{split_result.feature} >= {split_result.value} ")
            node.add_child(left_child)
            node.add_child(right_child)

            _build(left_ds, self.features, left_child)
            _build(right_ds, self.features, right_child)

        _build(self.dataset, self.features, self.root)

    def fit(self):
        # TODO implement
        pass
