import pytest
import httpx
import io
from PIL import Image
import base64
from unittest.mock import Mock, patch
from midrasai.client import Midras

# Mock API key for testing
TEST_API_KEY = "test_api_key"

@pytest.fixture
def midras_client():
    return Midras(api_key=TEST_API_KEY)

def test_midras_initialization():
    client = Midras(api_key=TEST_API_KEY)
    assert client.api_key == TEST_API_KEY
    assert isinstance(client.client, httpx.Client)
    assert client.client.base_url == "https://midras-api-worker.ajcardoza2000.workers.dev/"

def test_midras_initialization_custom_url():
    custom_url = "https://custom-api-url.com/"
    client = Midras(api_key=TEST_API_KEY, base_url=custom_url)
    assert client.client.base_url == custom_url

@patch('httpx.Client.post')
def test_embed_base64_images(mock_post, midras_client):
    mock_response = Mock()
    mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
    mock_post.return_value = mock_response

    base64_images = ["base64_encoded_image1", "base64_encoded_image2"]
    result = midras_client.embed_base64_images(base64_images)

    assert result == [[0.1, 0.2, 0.3]]
    mock_post.assert_called_once_with(
        "/embed/images",
        json={"api_key": TEST_API_KEY, "mode": "standard", "images": base64_images}
    )

@patch('httpx.Client.post')
def test_embed_text(mock_post, midras_client):
    mock_response = Mock()
    mock_response.json.return_value = {"embeddings": [[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]}
    mock_post.return_value = mock_response

    texts = ["sample text 1", "sample text 2"]
    result = midras_client.embed_text(texts)

    assert result == [[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    mock_post.assert_called_once_with(
        "/embed/text",
        json={"api_key": TEST_API_KEY, "mode": "standard", "texts": texts}
    )

@patch('httpx.Client.post')
def test_embed_base64_images_custom_mode(mock_post, midras_client):
    mock_response = Mock()
    mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
    mock_post.return_value = mock_response

    base64_images = ["base64_encoded_image1"]
    result = midras_client.embed_base64_images(base64_images, mode="custom")

    assert result == [[0.1, 0.2, 0.3]]
    mock_post.assert_called_once_with(
        "/embed/images",
        json={"api_key": TEST_API_KEY, "mode": "custom", "images": base64_images}
    )

@patch('httpx.Client.post')
def test_embed_text_custom_mode(mock_post, midras_client):
    mock_response = Mock()
    mock_response.json.return_value = {"embeddings": [[0.4, 0.5, 0.6]]}
    mock_post.return_value = mock_response

    texts = ["sample text"]
    result = midras_client.embed_text(texts, mode="custom")

    assert result == [[0.4, 0.5, 0.6]]
    mock_post.assert_called_once_with(
        "/embed/text",
        json={"api_key": TEST_API_KEY, "mode": "custom", "texts": texts}
    )

def get_base64_image():
    # Create a simple image
    img = Image.new('RGB', (100, 100), color='red')
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def test_embed_base64_images_integration(midras_client: Midras):
    base64_image = get_base64_image()
    result = midras_client.embed_base64_images([base64_image])

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], list)
    assert all(isinstance(x, float) for x in result[0])

def test_embed_text_integration(midras_client):
    texts = ["This is a sample text for embedding."]
    result = midras_client.embed_text(texts)

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], list)
    assert all(isinstance(x, float) for x in result[0])

def test_embed_multiple_texts_integration(midras_client):
    texts = [
        "This is the first sample text.",
        "This is the second sample text.",
        "And this is the third sample text."
    ]
    result = midras_client.embed_text(texts)

    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(embed, list) for embed in result)
    assert all(isinstance(x, float) for embed in result for x in embed)

def test_embed_base64_images_custom_mode_integration(midras_client):
    base64_image = get_base64_image()
    result = midras_client.embed_base64_images([base64_image], mode="custom")

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], list)
    assert all(isinstance(x, float) for x in result[0])

def test_embed_text_custom_mode_integration(midras_client):
    texts = ["This is a sample text for custom mode embedding."]
    result = midras_client.embed_text(texts, mode="custom")

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], list)
    assert all(isinstance(x, float) for x in result[0])

def test_api_key_validation(midras_client):
    invalid_client = Midras(api_key="invalid_key")
    with pytest.raises(Exception):  # Replace with specific exception if known
        invalid_client.embed_text(["This should fail."])
