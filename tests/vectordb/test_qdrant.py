import pytest

from midrasai.vectordb import Qdrant


@pytest.fixture
def qdrant():
    return Qdrant(":memory:")


def test_create_collection(qdrant: Qdrant):
    result = qdrant.create_index("test_index")
    assert result is True
