from typing import Any

from qdrant_client import AsyncQdrantClient, QdrantClient, models

from midrasai.client._vector_module.abc import BaseVectorModule, QueryResult
from midrasai.typedefs import ColBERT


class BaseQdrantModule(BaseVectorModule):
    def create_point(
        self, id: int | str, embedding: ColBERT, data: dict[str, Any]
    ) -> models.PointStruct:
        return models.PointStruct(id=id, payload=data, vector=embedding)


class QdrantModule(BaseQdrantModule):
    def __init__(self, location: str, *args, **kwargs) -> None:
        self.client = QdrantClient(location=location, *args, **kwargs)

    def create_index(self, name: str) -> bool:
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

    def save_points(
        self, index: str, points: list[models.PointStruct]
    ) -> models.UpdateResult:
        return self.client.upsert(index, points)

    def delete_index(self, name: str) -> bool:
        return self.client.delete_collection(collection_name=name)

    def search(
        self, index: str, query_vector: ColBERT, quantity: int
    ) -> list[QueryResult]:
        result = self.client.query_points(index, query=query_vector, limit=quantity)
        return [
            QueryResult(id=point.id, score=point.score, data=point.payload)
            for point in result.points
        ]


class AsyncQdrantModule(BaseQdrantModule):
    def __init__(self, location: str, *args, **kwargs) -> None:
        self.client = AsyncQdrantClient(location=location, *args, **kwargs)

    async def create_index(self, name: str) -> bool:
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

    async def save_points(
        self, index: str, points: list[models.PointStruct]
    ) -> models.UpdateResult:
        return await self.client.upsert(index, points)

    async def delete_index(self, name: str) -> bool:
        return await self.client.delete_collection(collection_name=name)

    async def search(
        self, index: str, query_vector: ColBERT, quantity: int
    ) -> list[QueryResult]:
        result = await self.client.query_points(
            index, query=query_vector, quantity=quantity
        )
        return [
            QueryResult(id=point.id, score=point.score, data=point.payload)
            for point in result.points
        ]
