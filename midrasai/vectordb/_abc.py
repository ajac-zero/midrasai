from abc import ABC, abstractmethod
from typing import Any, Awaitable, TypeVar

from midrasai.typedefs import ColBERT, QueryResult

R = TypeVar("R")


class VectorDB(ABC):
    @abstractmethod
    def create_index(self, name: str) -> bool: ...

    @abstractmethod
    def create_point(
        self, id: int | str, embedding: ColBERT, data: dict[str, Any]
    ) -> Any: ...

    @abstractmethod
    def save_points(self, index: str, points: list[Any]) -> Any: ...

    @abstractmethod
    def delete_index(self, name: str) -> bool: ...

    @abstractmethod
    def search(
        self, index: str, query_vector: ColBERT, quantity: int
    ) -> list[QueryResult]: ...


class AsyncVectorDB(ABC):
    @abstractmethod
    def create_index(self, name: str) -> Awaitable[bool]: ...

    @abstractmethod
    def create_point(
        self, id: int | str, embedding: ColBERT, data: dict[str, Any]
    ) -> Awaitable[Any]: ...

    @abstractmethod
    def save_points(self, index: str, points: list[Any]) -> Awaitable[Any]: ...

    @abstractmethod
    def delete_index(self, name: str) -> Awaitable[bool]: ...

    @abstractmethod
    def search(
        self, index: str, query_vector: ColBERT, quantity: int
    ) -> Awaitable[list[QueryResult]]: ...
