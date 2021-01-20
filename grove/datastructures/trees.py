from typing import Iterable

import pandas as pd

from grove.algorithms.splitting import SplittingMixin
from grove.datastructures.nodes import BNode



class AbstractTree:
    def __init__(
        self,
        dataset: pd.DataFrame,
        features: Iterable[str],
        target: str,
        max_depth: int = None
    ):
        self.dataset = dataset
        self.features = features
        self.target = target
        self.max_depth = max_depth

    def build(self):
        raise NotImplementedError


class DecisionTree(AbstractTree, SplittingMixin):
    def __init__(
        self,
        dataset: pd.DataFrame,
        features: Iterable[str],
        target: str,
        max_depth: int = 5,
        criteria: str = None
    ):
        super().__init__(dataset, features, target, max_depth=max_depth)
        self.criteria = criteria or self.GINI

    def build(self):

        def _build(dataset, features, node=None):
            split_result = self.calculate_best_split(dataset, features, self.target, self.criteria)

            if split_result.score == 0:
                # this is a leaf Node
                return BNode(
                    data=dataset,
                    label=dataset[self.target].unique()[0],
                    ancestor=node
                )

            left_ds, right_ds = self.binary_split(dataset, split_result.feature, split_result.value)

            left_child = BNode(left_ds, f' < {split_result.feature}', node)
            right_child = BNode(right_ds, f' >= {split_result.feature}', node)
            node.left = left_child
            node.right = right_child


        root_node = BNode(data=self.dataset, label='root')

        _build(self.dataset, self.features, root_node)
