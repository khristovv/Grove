from typing import TypeVar
from pandas import DataFrame
from uuid import uuid4

TAbstractNode = TypeVar("TAbstractNode", bound="AbstractNode")


class AbstractNode:
    def __init__(self, data: DataFrame, label: str | None = "", ancestor: TAbstractNode | None = None):
        self.identifier = str(uuid4())
        self.label = label
        self.data = data
        self.ancestor = ancestor

    def __str__(self) -> str:
        return f" {self.label} "

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        """Usefull if the node needs to be used as a dictionary key or in a set."""
        return self.identifier


TNode = TypeVar("TNode", bound="Node")


class Node(AbstractNode):
    def __init__(self, children: list[TNode] | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.children: list[Node] = children or []

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
        self.ancestor = self

        if not self.left:
            self.left = node
            return

        self.right = node

    def leafify(self, classifed_as):
        self.left = None
        self.right = None
        self.label = self.label + f"-> {classifed_as}"
