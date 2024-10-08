from enum import Enum
from typing import Any, Dict, TypeAlias

from pydantic import BaseModel, ConfigDict

Embedding: TypeAlias = list[float]
ColBERT: TypeAlias = list[Embedding]
Base64Image: TypeAlias = str


class Mode(str, Enum):
    Standard = "standard"
    Turbo = "turbo"
    Local = "local"


class MidrasRequest(BaseModel):
    key: str
    mode: Mode = Mode.Standard
    base64images: list[str] | None = None
    queries: list[str] | None = None


class MidrasResponse(BaseModel):
    embeddings: list[ColBERT]
    images: list | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class QueryResult(BaseModel):
    id: int | str
    score: float
    data: Dict[str, Any] | None
