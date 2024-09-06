from abc import ABC, abstractmethod
from typing import Any

from qdrant_client import AsyncQdrantClient, QdrantClient, models

from midrasai.typedefs import ColBERT


class BaseQdrantModule(ABC):
    @abstractmethod
    def __init__(self) -> None:
        self.client: QdrantClient | AsyncQdrantClient

    def create_collection(self, name: str) -> bool:
        return self.client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
        )

    def create_entry(
        self, id: int | str, vector: ColBERT, payload: dict[str, Any]
    ) -> models.PointStruct:
        return models.PointStruct(id=id, payload=payload, vector=vector)

    def save_entries(
        self, collection_name: str, entries: list[models.PointStruct]
    ) -> models.UpdateResult:
        return self.client.upsert(collection_name, entries)

    def delete_collection(self, name: str) -> bool:
        return self.client.delete_collection(collection_name=name)


class QdrantModule(BaseQdrantModule):
    def __init__(self, location: str, *args, **kwargs) -> None:
        self.client = QdrantClient(location=location, *args, **kwargs)

    def query(
        self, collection_name: str, query_vector: ColBERT
    ) -> models.QueryResponse:
        return self.client.query_points(collection_name, query=query_vector)


class AsyncQdrantModule(BaseQdrantModule):
    def __init__(self, location: str, *args, **kwargs) -> None:
        self.client = AsyncQdrantClient(location=location, *args, **kwargs)

    async def query(
        self, collection_name: str, query_vector: ColBERT
    ) -> models.QueryResponse:
        return await self.client.query_points(collection_name, query=query_vector)
