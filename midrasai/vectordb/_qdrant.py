from typing import Any, Awaitable, Callable, Dict, Optional, Union

from qdrant_client import AsyncQdrantClient, QdrantClient, models

from midrasai._abc import AsyncVectorDB, VectorDB
from midrasai.types import ColBERT, QueryResult


class Qdrant(VectorDB):
    def __init__(
        self,
        location: Optional[str] = None,
        url: Optional[str] = None,
        port: Optional[int] = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: Optional[bool] = None,
        api_key: Optional[str] = None,
        prefix: Optional[str] = None,
        timeout: Optional[int] = None,
        host: Optional[str] = None,
        path: Optional[str] = None,
        force_disable_check_same_thread: bool = False,
        grpc_options: Optional[Dict[str, Any]] = None,
        auth_token_provider: Optional[
            Union[Callable[[], str], Callable[[], Awaitable[str]]]
        ] = None,
        **kwargs: Any,
    ):
        self.client = QdrantClient(
            location,
            url,
            port,
            grpc_port,
            prefer_grpc,
            https,
            api_key,
            prefix,
            timeout,
            host,
            path,
            force_disable_check_same_thread,
            grpc_options,
            auth_token_provider,
            **kwargs,
        )

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

    def create_point(
        self, id: int | str, embedding: ColBERT, data: dict[str, Any]
    ) -> models.PointStruct:
        return models.PointStruct(id=id, payload=data, vector=embedding)

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


class AsyncQdrant(AsyncVectorDB):
    def __init__(
        self,
        location: Optional[str] = None,
        url: Optional[str] = None,
        port: Optional[int] = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: Optional[bool] = None,
        api_key: Optional[str] = None,
        prefix: Optional[str] = None,
        timeout: Optional[int] = None,
        host: Optional[str] = None,
        path: Optional[str] = None,
        force_disable_check_same_thread: bool = False,
        grpc_options: Optional[Dict[str, Any]] = None,
        auth_token_provider: Optional[
            Union[Callable[[], str], Callable[[], Awaitable[str]]]
        ] = None,
        **kwargs: Any,
    ):
        self.client = AsyncQdrantClient(
            location,
            url,
            port,
            grpc_port,
            prefer_grpc,
            https,
            api_key,
            prefix,
            timeout,
            host,
            path,
            force_disable_check_same_thread,
            grpc_options,
            auth_token_provider,
            **kwargs,
        )

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

    async def create_point(
        self, id: int | str, embedding: ColBERT, data: dict[str, Any]
    ) -> models.PointStruct:
        return models.PointStruct(id=id, payload=data, vector=embedding)

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
