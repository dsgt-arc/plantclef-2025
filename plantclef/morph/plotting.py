import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_mask(mask: np.ndarray, figsize=(3, 3)) -> plt.Figure:
    """Plot a mask."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(mask, cmap="gray")
    ax.axis("off")
    return fig


def plot_mask_stats(pdf: pd.DataFrame) -> plt.Figure:
    """Plot the mask statistics."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    ax = axes[0]
    ax.plot(pdf["iteration"], pdf["opening_mean"], label="opening")
    ax.plot(pdf["iteration"], pdf["closing_mean"], label="closing")
    ax.plot(pdf["iteration"], pdf["opening_closing_mean"], label="opening + closing")
    ax.legend()
    ax.set_xlabel("iteration")
    ax.set_ylabel("mean pixel")
    ax.set_title("Mask percentage")

    ax = axes[1]
    ax.plot(pdf["iteration"], pdf["opening_num_components"], label="opening")
    ax.plot(pdf["iteration"], pdf["closing_num_components"], label="closing")
    ax.plot(
        pdf["iteration"],
        pdf["opening_closing_num_components"],
        label="opening + closing",
    )
    ax.legend()
    ax.set_xlabel("iteration")
    ax.set_ylabel("number of components")
    ax.set_title("Connected components")
    plt.tight_layout()
    return fig
