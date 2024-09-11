from pdf2image import convert_from_path
from PIL.Image import Image

from midrasai.client._abc import BaseMidras
from midrasai.local.model import ColPali
from midrasai.local.processing import ColPaliProcessor
from midrasai.typedefs import Base64Image, MidrasResponse, Mode


class LocalMidras(BaseMidras):
    def __init__(self, qdrant: str = ":memory:", *args, **kwargs):
        super().__init__()
        self.model = ColPali.initialize()
        self.processor = ColPaliProcessor()

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
        embeddings = self.processor.batch_embed_images(self.model, pil_images)
        return MidrasResponse(credits_spent=0, embeddings=embeddings)

    def embed_base64_images(
        self, base64_images: list[Base64Image], mode: Mode = "standard"
    ) -> MidrasResponse: ...

    def embed_text(self, texts: list[str], mode: Mode = "standard") -> MidrasResponse:
        embeddings = self.processor.batch_embed_queries(self.model, texts)
        return MidrasResponse(credits_spent=0, embeddings=embeddings)
