from typing import cast

import pdf2image
import torch
from colpali_engine import ColPali, ColPaliProcessor

from midrasai._abc import BaseMidras, VectorDB
from midrasai.types import MidrasResponse
from midrasai.vectordb import Qdrant


class LocalMidras(BaseMidras):
    def __init__(
        self, device_map: str = "cuda:0", vector_database: VectorDB | None = None
    ):
        model_name = "vidore/colpali-v1.2"
        self.model = cast(
            ColPali,
            ColPali.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
            ),
        )
        self.processor = cast(
            ColPaliProcessor,
            ColPaliProcessor.from_pretrained(model_name),
        )
        self.index = vector_database if vector_database else Qdrant(location=":memory:")

    def embed_pdf(self, pdf, batch_size=10, include_images=False) -> MidrasResponse:
        if isinstance(pdf, str):
            images = pdf2image.convert_from_path(pdf)
        elif isinstance(pdf, bytes):
            images = pdf2image.convert_from_bytes(pdf)
        else:
            raise ValueError("Invalid type")

        embeddings = []

        for i in range(0, len(images), batch_size):
            image_batch = images[i : i + batch_size]
            response = self.embed_images(image_batch)
            embeddings.extend(response.embeddings)

        return MidrasResponse(
            embeddings=embeddings,
            images=images if include_images else None,
        )

    def embed_images(self, images, mode="local"):
        _ = mode
        batch_images = self.processor.process_images(images).to(self.model.device)
        with torch.no_grad():
            image_embeddings = self.model(**batch_images)
        return MidrasResponse(embeddings=image_embeddings.tolist())

    def embed_queries(self, queries, mode="local"):
        _ = mode
        batch_queries = self.processor.process_queries(queries).to(self.model.device)
        with torch.no_grad():
            query_embeddings = self.model(**batch_queries)
        return MidrasResponse(embeddings=query_embeddings.tolist())

    def create_index(self, name):
        return self.index.create_index(name)

    def add_point(self, index, id, embedding, data):
        point = self.index.create_point(id=id, embedding=embedding, data=data)
        return self.index.save_points(index, [point])

    def query(self, index, query, quantity=5):
        query_vector = self.embed_queries([query]).embeddings[0]
        return self.index.search(index, query_vector, quantity)

    def delete_index(self, name):
        return self.index.delete_index(name)
