import base64
import io

from PIL.Image import Image


def base64_encode_image(image: Image, format: str = "PNG") -> str:
    with io.BytesIO() as buffer:
        image.save(buffer, format)
        encoded_image = base64.b64encode(buffer.getvalue()).decode()
    return encoded_image


def base64_encode_image_list(image_list: list[Image], format: str = "PNG") -> list[str]:
    return [base64_encode_image(image, format) for image in image_list]
