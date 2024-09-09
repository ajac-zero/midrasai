from abc import ABC, abstractmethod
from typing import Awaitable

from PIL.Image import Image

from midrasai.typedefs import MidrasResponse, Mode


class BaseMidras(ABC):
    @abstractmethod
    def embed_pdf(
        self, pdf_path: str, batch_size: int = 10, include_images: bool = False
    ) -> MidrasResponse | Awaitable[MidrasResponse]: ...

    @abstractmethod
    def embed_pil_images(
        self, pil_images: list[Image], mode: Mode = "standard"
    ) -> MidrasResponse | Awaitable[MidrasResponse]: ...

    @abstractmethod
    def embed_base64_images(
        self, base64_images: list[str], mode: Mode = "standard"
    ) -> MidrasResponse | Awaitable[MidrasResponse]: ...

    @abstractmethod
    def embed_text(
        self, texts: list[str], mode: Mode = "standard"
    ) -> MidrasResponse | Awaitable[MidrasResponse]: ...
