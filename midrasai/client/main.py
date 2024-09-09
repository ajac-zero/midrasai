from typing import Any

import httpx
from pdf2image import convert_from_path
from PIL.Image import Image

from midrasai._constants import CLOUD_URL
from midrasai.client._abc import BaseMidras
from midrasai.client._vector_module import AsyncQdrantModule, QdrantModule
from midrasai.client.utils import base64_encode_image_list
from midrasai.typedefs import Base64Image, ColBERT, MidrasRequest, MidrasResponse, Mode


class BaseAPIMidras(BaseMidras):
    def __init__(self, midras_key: str):
        self.api_key = midras_key

    def validate_response(self, response: httpx.Response) -> MidrasResponse:
        if response.status_code >= 500:
            raise ValueError(response.text)
        return MidrasResponse(**response.json())


class Midras(BaseAPIMidras):
    def __init__(self, midras_key: str, qdrant: str = ":memory:", *args, **kwargs):
        super().__init__(midras_key)
        self.client = httpx.Client(base_url=CLOUD_URL)
        self.index = QdrantModule(location=":memory:", *args, **kwargs)

    def embed_pdf(
        self, pdf_path: str, batch_size: int = 10, include_images: bool = False
    ) -> MidrasResponse:
        images = convert_from_path(pdf_path)
        embeddings = []
        total_spent = 0

        for i in range(0, len(images), batch_size):
            image_batch = images[i : i + batch_size]
            response = self.embed_pil_images(image_batch)
            embeddings.extend(response.embeddings)
            total_spent += response.credits_spent

        return MidrasResponse(
            credits_spent=total_spent,
            embeddings=embeddings,
            images=images if include_images else None,
        )

    def embed_pil_images(
        self, pil_images: list[Image], mode: Mode = "standard"
    ) -> MidrasResponse:
        base64_images = base64_encode_image_list(pil_images)
        return self.embed_base64_images(base64_images, mode)

    def create_index(self, name: str) -> bool:
        return self.index.create_index(name)

    def add_point(
        self, index: str, id: str | int, embedding: ColBERT, data: dict[str, Any]
    ):
        point = self.index.create_point(id=id, embedding=embedding, data=data)
        return self.index.save_points(index, [point])

    def embed_base64_images(
        self, base64_images: list[Base64Image], mode: Mode = "standard"
    ) -> MidrasResponse:
        request = MidrasRequest(
            api_key=self.api_key,
            mode=mode,
            inputs=base64_images,
            image_input=True,
        )
        response = self.client.post("", json=request.dict(), timeout=180)
        return self.validate_response(response)

    def embed_text(self, texts: list[str], mode: Mode = "standard") -> MidrasResponse:
        request = MidrasRequest(
            api_key=self.api_key,
            mode=mode,
            inputs=texts,
            image_input=True,
        )
        response = self.client.post("", json=request.dict(), timeout=180)
        return self.validate_response(response)

    def query_text(self, index: str, query: str, quantity: int = 5):
        query_vector = self.embed_text([query]).embeddings[0]
        return self.index.search(index, query_vector, quantity)


class AsyncMidras(BaseAPIMidras):
    def __init__(self, midras_key: str, qdrant: str = ":memory:", *args, **kwargs):
        super().__init__(midras_key)
        self.client = httpx.AsyncClient(base_url=CLOUD_URL)
        self.index = AsyncQdrantModule(location=qdrant, *args, **kwargs)

    async def embed_pdf(
        self, pdf_path: str, batch_size: int = 10, include_images: bool = False
    ) -> MidrasResponse:
        images = convert_from_path(pdf_path)
        embeddings = []
        total_spent = 0

        for i in range(0, len(images), batch_size):
            image_batch = images[i : i + batch_size]
            response = await self.embed_pil_images(image_batch)
            embeddings.extend(response.embeddings)
            total_spent += response.credits_spent

        return MidrasResponse(
            credits_spent=total_spent,
            embeddings=embeddings,
            images=images if include_images else None,
        )

    async def embed_pil_images(
        self, pil_images: list[Image], mode: Mode = "standard"
    ) -> MidrasResponse:
        base64_images = base64_encode_image_list(pil_images)
        return await self.embed_base64_images(base64_images, mode)

    async def create_index(self, name: str) -> bool:
        return await self.index.create_index(name)

    def add_point(
        self, index: str, id: str | int, embedding: ColBERT, data: dict[str, Any]
    ):
        point = self.index.create_point(id=id, embedding=embedding, data=data)
        return self.index.save_points(index, [point])

    async def embed_base64_images(
        self, base64_images: list[Base64Image], mode: Mode = "standard"
    ) -> MidrasResponse:
        request = MidrasRequest(
            api_key=self.api_key,
            mode=mode,
            inputs=base64_images,
            image_input=True,
        )
        response = await self.client.post("", json=request.dict(), timeout=180)
        return self.validate_response(response)

    async def embed_text(
        self, texts: list[str], mode: Mode = "standard"
    ) -> MidrasResponse:
        request = MidrasRequest(
            api_key=self.api_key,
            mode=mode,
            inputs=texts,
            image_input=True,
        )
        response = await self.client.post("", json=request.dict(), timeout=180)
        return self.validate_response(response)

    async def query_text(self, index: str, query: str, quantity: int = 5):
        query_vector = (await self.embed_text([query])).embeddings[0]
        return await self.index.search(index, query_vector, quantity)
