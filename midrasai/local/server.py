from base64 import b64decode
from contextlib import asynccontextmanager
from io import BytesIO
from typing import cast

from fastapi import FastAPI, File, UploadFile
from PIL import Image
from pydantic import BaseModel

from midrasai.local.main import LocalMidras
from midrasai.types import MidrasResponse

midras = cast(LocalMidras, None)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global midras
    midras = LocalMidras()
    yield


app = FastAPI(lifespan=lifespan)


class ImageInput(BaseModel):
    images: list[str]

    @property
    def pil_images(self):
        return [
            cast(Image.Image, Image.open(BytesIO(b64decode(image))))
            for image in self.images
        ]


class TextInput(BaseModel):
    queries: list[str]


@app.post("/embed/queries")
def embed_queries(input: TextInput) -> MidrasResponse:
    query_embeddings = midras.embed_queries(input.queries)
    return query_embeddings


@app.post("/embed/images")
def embed_images(input: ImageInput) -> MidrasResponse:
    image_embeddings = midras.embed_images(input.pil_images)
    return image_embeddings


@app.post("/embed/pdf")
def embed_pdf(file: UploadFile = File(...)) -> MidrasResponse:
    image_embeddings = midras.embed_pdf(file.file.read())
    return image_embeddings
