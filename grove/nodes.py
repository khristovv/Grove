from uuid import uuid4
from typing import Literal, TypeVar
import numpy as np

from pandas import DataFrame
import numpy.typing as npt

from grove.entities import Coordinates

TAbstractNode = TypeVar("TAbstractNode", bound="AbstractNode")


class AbstractNode:
    def __init__(
        self,
        indexes: DataFrame,
        coordinates: tuple[int, int] = None,
        label: str | None = "",
        ancestor: TAbstractNode | None = None,
    ):
        """
        Args:
            indexes (DataFrame): The row indexes of the dataset that are in the node.
            label (str, optional): The label of the node. Defaults to "".
            ancestor (Node, optional): The ancestor of the node. Defaults to None.
        """
        self.identifier = str(uuid4())
        self.indexes = indexes
        self.coordinates = Coordinates(*coordinates) if coordinates else None
        self.label = label
        self.ancestor = ancestor

    def __str__(self) -> str:
        if self.is_root:
            return "Root"

        return f"Node{self.coordinates} - ( {self.label} )"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        """Usefull if the node needs to be used as a dictionary key or in a set."""
        return self.identifier

    @property
    def is_root(self) -> bool:
        return self.ancestor is None


TNode = TypeVar("TNode", bound="Node")


class Node(AbstractNode):
    NUMERICAL = "Numerical"
    CATEGORICAL = "Categorical"

    def __init__(
        self,
        children: list[TNode] | None = None,
        split_variable: str | None = None,
        split_variable_type: Literal["Numerical", "Categorical"] | None = None,
        split_stats: dict[str, npt.ArrayLike] = {},
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

    def __contains__(self, value) -> bool:
        if self.split_variable_type == self.NUMERICAL:
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
