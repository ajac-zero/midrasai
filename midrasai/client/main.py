from typing import Any

import httpx
from pdf2image import convert_from_path

from midrasai._abc import (
    AsyncBaseMidras,
    BaseMidras,
    VectorDB,
)
from midrasai._constants import CLOUD_URL
from midrasai.types import ColBERT, Image, MidrasRequest, MidrasResponse, Mode
from midrasai.vectordb import Qdrant


class Midras(BaseMidras):
    def __init__(self, api_key: str, vector_database: VectorDB | None = None):
        self.api_key = api_key
        self.client = httpx.Client(base_url=CLOUD_URL)
        self.index = vector_database if vector_database else Qdrant(location=":memory:")

    def embed_pdf(
        self, pdf_path: str, batch_size: int = 10, include_images: bool = False
    ) -> MidrasResponse:
        images = convert_from_path(pdf_path)
        embeddings = []
        total_spent = 0

        for i in range(0, len(images), batch_size):
            image_batch = images[i : i + batch_size]
            response = self.embed_images(image_batch)
            embeddings.extend(response.embeddings)
            total_spent += response.credits_spent

        return MidrasResponse(
            credits_spent=total_spent,
            embeddings=embeddings,
            images=images if include_images else None,
        )

    def embed_images(
        self, images: list[Image], mode: Mode = "standard"
    ) -> MidrasResponse:
        base64_images = self.base64_encode_image_list(images)

        request = MidrasRequest(
            key=self.api_key,
            mode=mode,
            base64images=base64_images,
        )

        response = self.client.post(
            "/embed/images", json=request.model_dump(), timeout=180
        )

        return response.json()

    def create_index(self, name: str) -> bool:
        return self.index.create_index(name)

    def add_point(
        self, index: str, id: str | int, embedding: ColBERT, data: dict[str, Any]
    ):
        point = self.index.create_point(id=id, embedding=embedding, data=data)
        return self.index.save_points(index, [point])

    def embed_queries(
        self, queries: list[str], mode: Mode = "standard"
    ) -> MidrasResponse:
        request = MidrasRequest(
            key=self.api_key,
            mode=mode,
            queries=queries,
        )
        response = self.client.post(
            "/embed/queries", json=request.model_dump(), timeout=180
        )
        return response.json()

    def query(self, index: str, query: str, quantity: int = 5):
        query_vector = self.embed_queries([query]).embeddings[0]
        return self.index.search(index, query_vector, quantity)


class AsyncMidras(AsyncBaseMidras):
    def __init__(self, api_key: str, vector_database: VectorDB | None = None):
        self.api_key = api_key
        self.client = httpx.AsyncClient(base_url=CLOUD_URL)
        self.index = vector_database if vector_database else Qdrant(location=":memory:")

    async def embed_pdf(
        self, pdf_path: str, batch_size: int = 10, include_images: bool = False
    ) -> MidrasResponse:
        images = convert_from_path(pdf_path)
        embeddings = []
        total_spent = 0

        for i in range(0, len(images), batch_size):
            image_batch = images[i : i + batch_size]
            response = await self.embed_images(image_batch)
            embeddings.extend(response.embeddings)
            total_spent += response.credits_spent

        return MidrasResponse(
            credits_spent=total_spent,
            embeddings=embeddings,
            images=images if include_images else None,
        )

    async def embed_images(
        self, images: list[Image], mode: Mode = "standard"
    ) -> MidrasResponse:
        base64_images = self.base64_encode_image_list(images)

        request = MidrasRequest(
            key=self.api_key,
            mode=mode,
            base64images=base64_images,
        )

        response = await self.client.post(
            "/embed/images", json=request.model_dump(), timeout=180
        )

        print(response.status_code)

        return response.json()

    async def create_index(self, name: str) -> bool:
        return self.index.create_index(name)

    async def add_point(
        self, index: str, id: str | int, embedding: ColBERT, data: dict[str, Any]
    ):
        point = self.index.create_point(id=id, embedding=embedding, data=data)
        return self.index.save_points(index, [point])

    async def embed_queries(
        self, queries: list[str], mode: Mode = "standard"
    ) -> MidrasResponse:
        request = MidrasRequest(
            key=self.api_key,
            mode=mode,
            queries=queries,
        )
        response = await self.client.post(
            "/embed/queries", json=request.model_dump(), timeout=180
        )
        return response.json()

    async def query(self, index: str, query: str, quantity: int = 5):
        query_vector = (await self.embed_queries([query])).embeddings[0]
        return self.index.search(index, query_vector, quantity)
