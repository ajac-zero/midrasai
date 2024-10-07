import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from midrasai.local.server import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


def test_embed_queries(client: TestClient):
    queries = ["Hello!", "I exist!"]
    r = client.post("/embed/queries", json={"queries": queries})

    assert r.status_code == 200
    
    data = r.json()
    assert data["images"] is None

    embeddings = data["embeddings"]
    assert len(embeddings) == 2

    for colbert in embeddings:
        assert isinstance(colbert, list)
        assert isinstance(colbert[0], list)


def test_embed_images(client: TestClient):
    images = [
        Image.open(open("./tests/assets/Colpali-example1.png", "rb")),
        Image.open(open("./tests/assets/Colpali-example2.png", "rb")),
        Image.open(open("./tests/assets/Colpali-example3.png", "rb")),
    ]

    encoded_images = []
    for image in images:
        with io.BytesIO() as buffer:
            image.save(buffer, format="JPEG")
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        encoded_images.append(img_str)

    r = client.post("/embed/images", json={"images": encoded_images})

    assert r.status_code == 200

    data = r.json()
    assert data["images"] is None

    embeddings = data["embeddings"]
    assert len(embeddings) == 3

    for colbert in embeddings:
        assert isinstance(colbert, list)
        assert isinstance(colbert[0], list)
