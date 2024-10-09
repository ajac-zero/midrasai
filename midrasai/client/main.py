from typing import Any

import httpx

from midrasai._abc import (
    AsyncBaseMidras,
    BaseMidras,
    VectorDB,
)
from midrasai._constants import CLOUD_URL
from midrasai.types import ColBERT, MidrasResponse, Mode
from midrasai.vectordb import Qdrant


class Midras(BaseMidras):
    def __init__(
        self,
        api_key: str,
        *,
        vector_database: VectorDB | None = None,
        base_url: str | None = None,
    ):
        self.api_key = api_key
        self.client = httpx.Client(base_url=CLOUD_URL if base_url is None else base_url)
        self.index = vector_database if vector_database else Qdrant(location=":memory:")

    def embed_pdf(
        self, pdf: str | bytes, batch_size: int = 10, include_images: bool = False
    ) -> MidrasResponse:
        if isinstance(pdf, str):
            with open(pdf, "rb") as f:
                file_data = f.read()
        elif isinstance(pdf, bytes):
            file_data = pdf

        files = {"file": ("test.pdf", file_data, "application/pdf")}
        response = self.client.post(
            "/embed/pdf",
            files=files,
            headers={"Authorization": f"Bearer {self.api_key}"},
            params={"batch_size": batch_size, "include_images": include_images},
        )

        if response.status_code == 200:
            return MidrasResponse.model_validate(response.json())
        else:
            raise ValueError("Internal server error")

    def embed_images(self, images: list, mode: Mode = Mode.Standard) -> MidrasResponse:
        encoded_images = self.base64_encode_image_list(images)

        response = self.client.post(
            "/embed/images",
            json={"images": encoded_images},
            headers={"Authorization": f"Bearer {self.api_key}"},
            params={"mode": mode},
        )

        if response.status_code == 200:
            return MidrasResponse.model_validate(response.json())
        else:
            raise ValueError("Internal server error")

    def create_index(self, name: str) -> bool:
        return self.index.create_index(name)

    def add_point(
        self, index: str, id: str | int, embedding: ColBERT, data: dict[str, Any]
    ):
        point = self.index.create_point(id=id, embedding=embedding, data=data)
        return self.index.save_points(index, [point])

    def embed_queries(
        self, queries: list[str], mode: Mode = Mode.Standard
    ) -> MidrasResponse:
        response = self.client.post(
            "/embed/queries",
            json={"queries": queries},
            headers={"Authorization": f"Bearer {self.api_key}"},
            params={"mode": mode},
        )

        if response.status_code == 200:
            return MidrasResponse.model_validate(response.json())
        else:
            raise ValueError("Internal server error")

    def query(self, index: str, query: str, quantity: int = 5):
        query_vector = self.embed_queries([query]).embeddings[0]
        return self.index.search(index, query_vector, quantity)


class AsyncMidras(AsyncBaseMidras):
    def __init__(
        self,
        api_key: str,
        *,
        vector_database: VectorDB | None = None,
        base_url: str | None = None,
    ):
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            base_url=CLOUD_URL if base_url is None else base_url
        )
        self.index = vector_database if vector_database else Qdrant(location=":memory:")

    async def embed_pdf(
        self, pdf: str | bytes, batch_size: int = 10, include_images: bool = False
    ) -> MidrasResponse:
        if isinstance(pdf, str):
            with open(pdf, "rb") as f:
                file_data = f.read()
        elif isinstance(pdf, bytes):
            file_data = pdf

        files = {"file": ("test.pdf", file_data, "application/pdf")}
        response = await self.client.post(
            "/embed/pdf",
            files=files,
            headers={"Authorization": f"Bearer {self.api_key}"},
            params={"batch_size": batch_size, "include_images": include_images},
        )

        if response.status_code == 200:
            return MidrasResponse.model_validate(response.json())
        else:
            raise ValueError("Internal server error")

    async def embed_images(
        self, images: list, mode: Mode = Mode.Standard
    ) -> MidrasResponse:
        encoded_images = self.base64_encode_image_list(images)

        response = await self.client.post(
            "/embed/images",
            json={"images": encoded_images},
            headers={"Authorization": f"Bearer {self.api_key}"},
            params={"mode": mode},
        )

        if response.status_code == 200:
            return MidrasResponse.model_validate(response.json())
        else:
            raise ValueError("Internal server error")

    async def create_index(self, name: str) -> bool:
        return self.index.create_index(name)

    async def add_point(
        self, index: str, id: str | int, embedding: ColBERT, data: dict[str, Any]
    ):
        point = self.index.create_point(id=id, embedding=embedding, data=data)
        return self.index.save_points(index, [point])

    async def embed_queries(
        self, queries: list[str], mode: Mode = Mode.Standard
    ) -> MidrasResponse:
        response = await self.client.post(
            "/embed/queries",
            json={"queries": queries},
            headers={"Authorization": f"Bearer {self.api_key}"},
            params={"mode": mode},
        )

        if response.status_code == 200:
            return MidrasResponse.model_validate(response.json())
        else:
            raise ValueError("Internal server error")

    async def query(self, index: str, query: str, quantity: int = 5):
        query_vector = (await self.embed_queries([query])).embeddings[0]
        return self.index.search(index, query_vector, quantity)
