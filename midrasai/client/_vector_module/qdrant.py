from typing import Any

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, QdrantClient, models

from midrasai.typedefs import ColBERT


class QueryResult(BaseModel):
    id: int
    score: float
    data: dict[str, Any]


class BaseQdrantModule:
    def create_entry(
        self, id: int | str, vector: ColBERT, payload: dict[str, Any]
    ) -> models.PointStruct:
        return models.PointStruct(id=id, payload=payload, vector=vector)


class QdrantModule(BaseQdrantModule):
    def __init__(self, location: str, *args, **kwargs) -> None:
        self.client = QdrantClient(location=location, *args, **kwargs)

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

    def save_entries(
        self, collection_name: str, entries: list[models.PointStruct]
    ) -> models.UpdateResult:
        return self.client.upsert(collection_name, entries)

    def delete_collection(self, name: str) -> bool:
        return self.client.delete_collection(collection_name=name)

    def query(
        self, collection_name: str, query_vector: ColBERT, k: int
    ) -> list[QueryResult]:
        result = self.client.query_points(collection_name, query=query_vector, limit=k)
        return [
            QueryResult(id=point.id, score=point.score, data=point.payload)
            for point in result.points
        ]


class AsyncQdrantModule(BaseQdrantModule):
    def __init__(self, location: str, *args, **kwargs) -> None:
        self.client = AsyncQdrantClient(location=location, *args, **kwargs)

    async def create_collection(self, name: str) -> bool:
        return await self.client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
        )

    async def save_entries(
        self, collection_name: str, entries: list[models.PointStruct]
    ) -> models.UpdateResult:
        return await self.client.upsert(collection_name, entries)

    async def delete_collection(self, name: str) -> bool:
        return await self.client.delete_collection(collection_name=name)

    async def query(
        self, collection_name: str, query_vector: ColBERT
    ) -> list[QueryResult]:
        result = await self.client.query_points(collection_name, query=query_vector)
        return [
            QueryResult(id=point.id, score=point.score, data=point.payload)
            for point in result.points
        ]
