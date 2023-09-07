from typing import Literal, TypeVar

import numpy as np

from grove.nodes.abstract import AbstractNode


TNode = TypeVar("TNode", bound="Node")


class Node(AbstractNode):
    CONTINUOUS = "Continuous"
    CATEGORICAL = "Categorical"

    def __init__(
        self,
        children: list[TNode] | None = None,
        split_variable: str | None = None,
        split_variable_type: Literal["Continuous", "Categorical"] | None = None,
        split_stats: dict[str, np.ndarray] = {},
        bounds: list = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.children = children or []
        self.split_variable = split_variable
        self.split_variable_type = split_variable_type
        self.split_stats = split_stats or {}
        self.bounds = bounds or [-np.inf, np.inf]
        self.predicted_value = None

    def is_within_bounds(self, value) -> bool:
        if self.split_variable_type == self.CONTINUOUS:
            left_bound, right_bound = self.bounds
            return left_bound <= value < right_bound

        if self.split_variable_type == self.CATEGORICAL:
            return value in self.bounds

    @property
    def is_leaf(self) -> bool:
        return not len(self.children)

    @property
    def is_inner(self) -> bool:
        return not self.is_leaf

    def add_child(self, node: TNode):
        node.ancestor = self
        self.children.append(node)
