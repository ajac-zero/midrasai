from abc import ABC, abstractmethod

import httpx
from PIL.Image import Image

from midrasai._constants import CLOUD_URL
from midrasai.client.utils import base64_encode_image_list
from midrasai.typedefs import Base64Image, MidrasResponse, Mode


class BaseMidras(ABC):
    def validate_response(self, response: httpx.Response) -> MidrasResponse:
        if response.status_code >= 500:
            raise ValueError(response.text)
        return MidrasResponse(**response.json())

    def embed_pil_images(
        self, pil_images: list[Image], mode: Mode = "standard"
    ) -> MidrasResponse:
        base64_images = base64_encode_image_list(pil_images)
        return self.embed_base64_images(base64_images, mode)

    @abstractmethod
    def embed_base64_images(): ...


class Midras(BaseMidras):
    def __init__(self, api_key: str):
        self.client = httpx.Client(base_url=CLOUD_URL)
        self.api_key = api_key

    def embed_base64_images(
        self, base64_images: list[Base64Image], mode: Mode = "standard"
    ) -> MidrasResponse:
        json = {
            "api_key": self.api_key,
            "mode": mode,
            "inputs": base64_images,
            "image_input": True,
        }
        response = self.client.post("", json=json, timeout=180)
        return self.validate_response(response)

    def embed_text(self, texts: list[str], mode: Mode = "standard") -> MidrasResponse:
        json = {
            "api_key": self.api_key,
            "mode": mode,
            "inputs": texts,
            "image_input": False,
        }
        response = self.client.post("", json=json, timeout=180)
        return self.validate_response(response)


class AsyncMidras(BaseMidras):
    def __init__(self, api_key: str):
        self.client = httpx.AsyncClient(base_url=CLOUD_URL)
        self.api_key = api_key

    async def embed_base64_images(
        self, base64_images: list[Base64Image], mode: Mode = "standard"
    ) -> MidrasResponse:
        json = {
            "api_key": self.api_key,
            "mode": mode,
            "inputs": base64_images,
            "image_input": True,
        }
        response = await self.client.post("", json=json, timeout=180)
        return self.validate_response(response)

    async def embed_text(
        self, texts: list[str], mode: Mode = "standard"
    ) -> MidrasResponse:
        json = {
            "api_key": self.api_key,
            "mode": mode,
            "inputs": texts,
            "image_input": False,
        }
        response = await self.client.post("", json=json, timeout=180)
        return self.validate_response(response)
