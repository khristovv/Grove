from uuid import uuid4
from typing import Literal, TypeVar

from pandas import DataFrame
import numpy.typing as npt

TAbstractNode = TypeVar("TAbstractNode", bound="AbstractNode")


class AbstractNode:
    def __init__(
        self,
        indexes: DataFrame,
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
        self.label = label
        self.indexes = indexes
        self.ancestor = ancestor

    def __str__(self) -> str:
        return f"{self.label}"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        """Usefull if the node needs to be used as a dictionary key or in a set."""
        return self.identifier


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
        self.bounds = bounds or []
        self.class_label = None

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

    @property
    def is_root(self) -> bool:
        return self.ancestor is None

    def add_child(self, node: TNode):
        node.ancestor = self
        self.children.append(node)


TBinaryNode = TypeVar("TBinaryNode", bound="BinaryNode")


# type: ignore
class BinaryNode(Node):
    """
    BinaryNode -> A node wich can have at most 2 children
    """

    def __init__(
        self,
        left: TBinaryNode | None = None,
        right: TBinaryNode | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(children=[left, right], *args, **kwargs)

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    @property
    def left(self):
        return self.children[0]

    @left.setter
    def left(self, node: TBinaryNode):
        self.children[0] = node

    @property
    def right(self):
        return self.children[1]

    @right.setter
    def right(self, node: TBinaryNode):
        self.children[1] = node

    def add_child(self, node: TBinaryNode):
        node.ancestor = self

        if not self.left:
            self.left = node
            return

        self.right = node

    def leafify(self, classifed_as: str):
        self.left = None
        self.right = None
        self.label = self.label + f"-> {classifed_as}"
