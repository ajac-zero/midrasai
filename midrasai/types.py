from typing import Any, Dict, Literal, TypeAlias

from PIL import Image as PILImage
from PIL import ImageFile
from pydantic import BaseModel

Embedding: TypeAlias = list[float]
ColBERT: TypeAlias = list[Embedding]
Base64Image: TypeAlias = str
Mode: TypeAlias = Literal["standard", "turbo", "local"]
Image: TypeAlias = PILImage.Image | ImageFile.ImageFile


class MidrasRequest(BaseModel):
    key: str
    mode: Mode = "standard"
    base64images: list[str] | None = None
    queries: list[str] | None = None


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
