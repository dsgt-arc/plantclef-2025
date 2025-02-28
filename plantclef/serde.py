"""Module for encoding and decoding data structures to and from raw bytes"""

from PIL import Image
import numpy as np
import zlib
import io


def deserialize_image(bytes: bytes) -> Image.Image:
    """Decode the image from raw bytes using PIL."""
    buffer = io.BytesIO(bytes)
    return Image.open(buffer)


def serialize_image(image: Image.Image) -> bytes:
    """Encode the image as raw bytes using PIL."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def deserialize_mask(bytes: bytes) -> np.ndarray:
    """Decode the numpy mask array from raw bytes using np.load()."""
    buffer = io.BytesIO(zlib.decompress(bytes))
    return np.load(buffer)


def serialize_mask(mask: np.ndarray) -> bytes:
    """Encode the numpy mask array as raw bytes using np.save()."""
    buffer = io.BytesIO()
    np.save(buffer, mask)
    return zlib.compress(buffer.getvalue())
