from base64 import b64decode
from contextlib import asynccontextmanager
from io import BytesIO

from colpali_engine import ColPali, ColPaliProcessor
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
from torch import bfloat16, no_grad

model = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    model["model"] = ColPali.from_pretrained(
        "vidore/colpali-v1.2",
        torch_dtype=bfloat16,
        device_map="cuda:0",
    )
    model["processor"] = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")
    yield


app = FastAPI(lifespan=lifespan)


class ImageInput(BaseModel):
    images: list[str]

    @property
    def pil_images(self):
        return [Image.open(BytesIO(b64decode(image))) for image in self.images]


class TextInput(BaseModel):
    queries: list[str]


@app.post("/embed/queries")
def embed_text(input: TextInput):
    batch_queries = (
        model["processor"].process_queries(input.queries).to(model["model"].device)
    )
    with no_grad():
        query_embeddings = model["model"](**batch_queries)
    return {"embeddings": query_embeddings.tolist()}


@app.post("/embed/images")
def embed_images(input: ImageInput):
    batch_queries = (
        model["processor"].process_queries(input.pil_images).to(model["model"].device)
    )
    with no_grad():
        query_embeddings = model["model"](**batch_queries)
    return {"embeddings": query_embeddings.tolist()}
