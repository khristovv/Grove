from uuid import uuid4
from typing import TypeVar

from pandas import DataFrame

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
