from typing import Literal

import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor, PaliGemmaProcessor


class ColPaliProcessor:
    def __init__(
        self, model_name: Literal["vidore/colpali"] = "vidore/colpali"
    ) -> None:
        self.processor: PaliGemmaProcessor = AutoProcessor.from_pretrained(model_name)

    def _process_images(self, images: list[Image.Image], max_length: int = 50):
        batch_doc = self.processor(
            text=["Describe the image."] * len(images),
            images=[image.convert("RGB") for image in images],
            return_tensors="pt",
            padding="longest",
            max_length=max_length + self.processor.image_seq_length,
        )
        return batch_doc

    def create_image_dataloader(self, images: list[Image.Image]):
        dataloader = DataLoader(
            images,  # type: ignore
            batch_size=4,
            shuffle=False,
            collate_fn=lambda x: self._process_images(x),
        )
        return dataloader

    def embed_dataloader(self, model, dataloader: DataLoader):
        embedding_list: list[torch.Tensor] = []
        for batch_doc in dataloader:
            with torch.no_grad():
                batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
                embeddings_doc = model(**batch_doc)
            embedding_list.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
        return embedding_list

    def batch_embed_images(self, model, images: list[Image.Image]):
        dataloader = self.create_image_dataloader(images)

        embedding_tensor = self.embed_dataloader(model, dataloader)

        embedding_list = [tensor.tolist() for tensor in embedding_tensor]

        return embedding_list

    def _process_queries(self, queries: list[str], max_length: int = 50):
        texts_query = []
        for query in queries:
            query = f"Question: {query}<unused0><unused0><unused0><unused0><unused0>"
            texts_query.append(query)

        batch_query = self.processor(
            images=[Image.new("RGB", (448, 448), (255, 255, 255)).convert("RGB")]
            * len(texts_query),
            # NOTE: the image is not used in batch_query but it is required for calling the processor
            text=texts_query,
            return_tensors="pt",
            padding="longest",
            max_length=max_length + self.processor.image_seq_length,
        )
        del batch_query["pixel_values"]

        batch_query["input_ids"] = batch_query["input_ids"][
            ..., self.processor.image_seq_length :
        ]
        batch_query["attention_mask"] = batch_query["attention_mask"][
            ..., self.processor.image_seq_length :
        ]
        return batch_query

    def create_query_dataloader(self, queries: list[str]):
        dataloader = DataLoader(
            queries,  # type: ignore
            batch_size=4,
            shuffle=False,
            collate_fn=lambda x: self._process_queries(x),
        )
        return dataloader

    def embed_queries(self, model, dataloader: DataLoader):
        query_embeddings: list[torch.Tensor] = []
        for batch_query in dataloader:
            with torch.no_grad():
                batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
                embedding_query = model(**batch_query)
            query_embeddings.extend(list(torch.unbind(embedding_query.to("cpu"))))
        return query_embeddings

    def batch_embed_queries(self, model, queries: list[str]):
        dataloader = self.create_query_dataloader(queries)

        query_tensors = self.embed_queries(model, dataloader)

        query_embeddings = [tensor.tolist() for tensor in query_tensors]

        return query_embeddings
