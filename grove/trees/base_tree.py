from typing import Iterable

import pandas as pd

from grove.utils.splitting import SplittingMixin
from grove.nodes import BinaryNode


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
