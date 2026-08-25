"""Shared registry machinery."""

from __future__ import annotations

from typing import Dict, Generic, Iterator, List, TypeVar

from ..core.errors import RegistrationError

T = TypeVar("T")


class BaseRegistry(Generic[T]):
    """An id-keyed store with explicit duplicate handling.

    Silent overwrite is a footgun in a system whose whole premise is auditable
    provenance, so re-registering an id raises unless ``replace=True``.
    """

    kind = "item"

    def __init__(self) -> None:
        self._items: Dict[str, T] = {}

    def register(self, key: str, item: T, replace: bool = False) -> T:
        if not key or not key.strip():
            raise RegistrationError("{} id must be non-empty".format(self.kind))
        if key in self._items and not replace:
            raise RegistrationError(
                "{} '{}' already registered; pass replace=True to override".format(
                    self.kind, key
                )
            )
        self._items[key] = item
        return item

    def get(self, key: str) -> T:
        try:
            return self._items[key]
        except KeyError:
            raise RegistrationError("unknown {} '{}'".format(self.kind, key))

    def ids(self) -> List[str]:
        return sorted(self._items)

    def all(self) -> List[T]:
        return [self._items[k] for k in self.ids()]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[str]:
        return iter(self.ids())

    def __contains__(self, key: object) -> bool:
        return key in self._items
