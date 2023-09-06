from typing import Iterable, Any


def first(iterable: Iterable, default=None) -> Any:
    return next((x for x in iterable), default)
