"""Module for encoding and decoding data structures to and from raw bytes"""

from PIL import Image
import numpy as np
import zlib
import io


def deserialize_image(buffer: bytes | bytearray) -> Image.Image:
    """Decode the image from raw bytes using PIL."""
    buffer = io.BytesIO(bytes(buffer))
    return Image.open(buffer)


def serialize_image(image: Image.Image) -> bytes:
    """Encode the image as raw bytes using PIL."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def deserialize_mask(buffer: bytes | bytearray, use_compression=True) -> np.ndarray:
    """Decode the numpy mask array from raw bytes using np.load()."""
    if use_compression:
        buffer = zlib.decompress(bytes(buffer))
    return np.load(io.BytesIO(buffer))


def serialize_mask(mask: np.ndarray, use_compression=True) -> bytes:
    """Encode the numpy mask array as raw bytes using np.save()."""
    fp = io.BytesIO()
    np.save(fp, mask)
    buffer = fp.getvalue()
    if use_compression:
        buffer = zlib.compress(buffer)
    return buffer
