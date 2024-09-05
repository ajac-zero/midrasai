from typing import Literal, TypeAlias

embedding: TypeAlias = list[float]
colbert: TypeAlias = list[embedding]
base64_image: TypeAlias = str
mode: TypeAlias = Literal["standard", "turbo"]
