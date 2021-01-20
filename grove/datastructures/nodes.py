from pandas import DataFrame
from uuid import uuid4

class AbstractNode:
    def __init__(
        self,
        data: DataFrame,
        ancestor = None
    ):
        self.data = data
        self.identifier = str(uuid4())
        self.ancestor = ancestor

    def __hash__(self) -> int:
        """Usefull if the node needs to be used as a dictionary key or in a set."""
        return self.identifier


class BNode(AbstractNode):
    """
    BNode stands for BinaryNode -> A node wich can have at most 2 children
    """
    def __init__(
        self,
        data: DataFrame,
        label: str = None,
        ancestor = None,
        left = None,
        right = None
    ):
        super().__init__(data, ancestor)
        self.label = label
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"Node({self.label})"

    @property
    def is_leaf(self) -> bool:
        return self.left is self.right is None

    @property
    def is_inner(self) -> bool:
        return bool(self.left or self.right)
