"""Statistics for masks."""

import numpy as np
from skimage.measure import label


def mask_mean(mask: np.ndarray) -> float:
    """Calculate the mean of a mask."""
    return float(mask.mean())


def mask_num_components(mask: np.ndarray) -> int:
    """Calculate the number of components in a mask."""
    return int(label(mask).max())
