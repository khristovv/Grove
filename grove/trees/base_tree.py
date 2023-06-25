import pandas as pd
from grove.constants import SpecialChars

from grove.utils.gain import GainMixin
from grove.nodes import Node


class AbstractTree:
    def train(self, x: pd.DataFrame, y: pd.DataFrame):
        """A method that builds the decision tree from the training set (x, y)."""
        raise NotImplementedError

    def test(self, x: pd.DataFrame, y: pd.DataFrame):
        """A method that tests the decision tree on the test set (x, y)."""
        raise NotImplementedError

    def classify(self):
        """A method"""
        raise NotImplementedError


class BaseTree(AbstractTree, GainMixin):
    def __str__(self):
        """A method that builds a string representation of the decision tree."""

        lines = []

        def _next_line(node: Node, indent: str = "", is_last: bool = True):
            marker = SpecialChars.TREE_LAST_BRANCH if is_last else SpecialChars.TREE_BRANCH

            output = str(node) if node.is_root else f"{indent}{marker} {node}"
            lines.append(f"{output}\n")

            if node.is_leaf:
                return

            child_count = len(node.children)
            for index, child in enumerate(node.children):
                is_last_child = index == child_count - 1
                child_indent = indent + ("   " if is_last else f"{SpecialChars.TREE_PATH}  ")
                _next_line(node=child, indent=child_indent, is_last=is_last_child)

        _next_line(node=self.root)

        return "".join(lines)
