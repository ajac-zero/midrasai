from typing import Any, Dict, Literal, TypeAlias

from PIL.Image import Image
from pydantic import BaseModel

Embedding: TypeAlias = list[float]
ColBERT: TypeAlias = list[Embedding]
Base64Image: TypeAlias = str
Mode: TypeAlias = Literal["standard", "turbo"]


class MidrasRequest(BaseModel):
    api_key: str
    mode: Mode = "standard"
    inputs: list[str]
    image_input: bool = False


class MidrasResponse(BaseModel):
    credits_spent: int
    embeddings: list[ColBERT]
    images: list[Image] | None = None

    class Config:
        arbitrary_types_allowed = True


class QueryResult(BaseModel):
    id: int | str
    score: float
    data: Dict[str, Any] | None
