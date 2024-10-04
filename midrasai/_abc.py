from abc import ABC, abstractmethod
from base64 import b64encode
from io import BytesIO
from typing import Any, Awaitable

from midrasai.types import ColBERT, Image, MidrasResponse, Mode, QueryResult


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


class BaseMidras(ABC):
    @abstractmethod
    def embed_pdf(
        self, pdf_path: str, batch_size: int = 10, include_images: bool = False
    ) -> MidrasResponse | Awaitable[MidrasResponse]: ...

    @abstractmethod
    def embed_images(
        self, images: list[Image], mode: Mode = "standard"
    ) -> MidrasResponse | Awaitable[MidrasResponse]: ...

    @abstractmethod
    def embed_queries(
        self, queries: list[str], mode: Mode = "standard"
    ) -> MidrasResponse | Awaitable[MidrasResponse]: ...

    @abstractmethod
    def create_index(self, name: str) -> bool | Awaitable[bool]: ...

    @abstractmethod
    def add_point(
        self, index: str, id: str | int, embedding: ColBERT, data: dict[str, Any]
    ) -> Any | Awaitable[Any]: ...

    @abstractmethod
    def query(
        self, index: str, query: str, quantity: int = 5
    ) -> list[QueryResult] | Awaitable[list[QueryResult]]: ...

    def base64_encode_image_list(self, pil_images: list[Image]) -> list[str]:
        base64_images = []
        for image in pil_images:
            with BytesIO() as buffer:
                image.save(buffer, format=image.format)
                base64_image = b64encode(buffer.getvalue()).decode("utf-8")
            base64_images.append(base64_image)
        return base64_images


class BaseAsyncMidras(ABC):
    @abstractmethod
    async def embed_pdf(
        self, pdf_path: str, batch_size: int = 10, include_images: bool = False
    ) -> MidrasResponse: ...

    @abstractmethod
    def embed_pil_images(
        self, pil_images: list[Image], mode: Mode = "standard"
    ) -> MidrasResponse: ...

    @abstractmethod
    def embed_texts(
        self, texts: list[str], mode: Mode = "standard"
    ) -> MidrasResponse: ...

    def base64_encode_image_list(self, pil_images: list[Image]) -> list[str]:
        base64_images = []
        for image in pil_images:
            with BytesIO() as buffer:
                image.save(buffer, format=image.format)
                base64_image = b64encode(buffer.getvalue()).decode("utf-8")
            base64_images.append(base64_image)
        return base64_images


class AsyncVectorDB(ABC):
    @abstractmethod
    async def create_index(self, name: str) -> bool: ...

    @abstractmethod
    async def create_point(
        self, id: int | str, embedding: ColBERT, data: dict[str, Any]
    ) -> Any: ...

    @abstractmethod
    async def save_points(self, index: str, points: list[Any]) -> Any: ...

    @abstractmethod
    async def delete_index(self, name: str) -> bool: ...

    @abstractmethod
    async def search(
        self, index: str, query_vector: ColBERT, quantity: int
    ) -> list[QueryResult]: ...
