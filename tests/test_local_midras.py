import pytest
from PIL import Image

from midrasai.local import Midras
from midrasai.types import MidrasResponse, QueryResult


@pytest.fixture(scope="module")
def m():
    return Midras()


@pytest.fixture(scope="module")
def state():
    return {}


def test_create_index(m: Midras):
    r = m.create_index("test_index")
    assert r is True


def test_embed_query(m: Midras, state):
    r = m.embed_queries(["hello", "it's me"])
    assert isinstance(r, MidrasResponse)
    assert len(r.embeddings) == 2
    assert isinstance(r.embeddings[0], list)
    assert isinstance(r.embeddings[0][0], list)

    state["query"] = r.embeddings


def test_embed_pdf(m: Midras, state):
    r = m.embed_pdf("./tests/assets/Attention_is_all_you_need.pdf", include_images=True)
    assert isinstance(r, MidrasResponse)
    assert len(r.images) == 15  # type: ignore
    assert len(r.embeddings) == 15
    assert isinstance(r.embeddings[0], list)
    assert isinstance(r.embeddings[0][0], list)

    state["pdf"] = r.embeddings


def test_embed_images(m: Midras, state):
    r = m.embed_images(
        [
            Image.open(open("./tests/assets/Colpali-example1.png", "rb")),  # type: ignore
            Image.open(open("./tests/assets/Colpali-example2.png", "rb")),  # type: ignore
            Image.open(open("./tests/assets/Colpali-example3.png", "rb")),  # type: ignore
        ]
    )
    assert isinstance(r, MidrasResponse)
    assert len(r.embeddings) == 3
    assert isinstance(r.embeddings[0], list)
    assert isinstance(r.embeddings[0][0], list)

    state["images"] = r.embeddings


def test_add_point(m: Midras, state):
    for i, e in enumerate(state["pdf"]):
        r = m.add_point(index="test_index", id=i, embedding=e, data={"test": "midras"})
        assert r is not None


def test_query(m: Midras):
    q = m.query("test_index", "whats a transformer")

    assert len(q) == 5
    for r in q:
        assert isinstance(r, QueryResult)
        assert r.data.get("test") == "midras"  # type: ignore
