from PIL.Image import Image

from midrasai.client import Midras
from midrasai.local.model import ColPali
from midrasai.local.processing import ColPaliProcessor
from midrasai.typedefs import Base64Image, MidrasResponse


class LocalMidras(Midras):
    def __init__(self, qdrant: str = ":memory:", *args, **kwargs):
        super().__init__(None, qdrant, *args, **kwargs)
        self.model = ColPali.initialize()
        self.processor = ColPaliProcessor()

    def embed_pil_images(self, pil_images: list[Image]) -> MidrasResponse:
        embeddings = self.processor.batch_embed_images(self.model, pil_images)
        return MidrasResponse(credits_spent=0, embeddings=embeddings)

    def embed_text(self, texts: list[str]) -> MidrasResponse:
        embeddings = self.processor.batch_embed_queries(self.model, texts)
        return MidrasResponse(credits_spent=0, embeddings=embeddings)
