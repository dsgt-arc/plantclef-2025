import numpy as np
from skimage.morphology import binary_closing, binary_opening, diamond


def closing(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Dilations followed by erosions."""
    footprint = diamond(iterations, decomposition="sequence")
    img = binary_closing(mask, footprint)
    img = binary_opening(mask, footprint)
    return img


def opening(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Erosions followed by dilations."""
    footprint = diamond(iterations, decomposition="sequence")
    img = binary_opening(mask, footprint)
    img = binary_closing(mask, footprint)
    return img
