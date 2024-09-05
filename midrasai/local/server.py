import base64, io
from contextlib import asynccontextmanager

from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

from midrasai.local.model import ColPali
from midrasai.local.processing import ColPaliProcessor
from midrasai.typedefs import base64_image, colbert

model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model["colpali"] = ColPali.initialize()
    model["processor"] = ColPaliProcessor(model_name="vidore/colpali")
    yield


app = FastAPI(lifespan=lifespan)


class BaseInput(BaseModel):
    api_key: str
    mode: str


class TextInput(BaseInput):
    texts: list[str]


class ImageInput(BaseInput):
    images: list[base64_image]


class Output(BaseModel):
    credits_spent: int
    embeddings: list[colbert]


def base64_to_pil(base64_image: base64_image) -> Image.Image:
    image_bytes = base64.b64decode(base64_image)
    image_stream = io.BytesIO(image_bytes)
    return Image.open(image_stream)


@app.post("/embed/images")
async def embed_images(input: ImageInput):
    pil_images = [base64_to_pil(image) for image in input.images]
    embeddings = model["processor"].batch_embed_images(model["colpali"], pil_images)
    return Output(credits_spent=0, embeddings=embeddings)


@app.post("/embed/text")
async def embed_text(input: TextInput):
    embeddings = model["processor"].batch_embed_queries(model["colpali"], input.texts)
    return Output(credits_spent=0, embeddings=embeddings)
