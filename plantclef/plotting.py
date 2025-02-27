import io
import math
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_images_from_binary(
    df,
    data_col: str,
    label_col: str,
    grid_size=(3, 3),
    crop_square: bool = False,
    figsize: tuple = (12, 12),
    dpi: int = 80,
):
    """
    Display images in a grid with binomial names as labels.

    :param df: DataFrame with the embeddings data.
    :param data_col: Name of the data column.
    :param label_col: Name of the species being displayed as image labels.
    :param grid_size: Tuple (rows, cols) representing the grid size.
    :param crop_square: Boolean, whether to crop images to a square format by taking the center.
    :param figsize: Regulates the size of the figure.
    :param dpi: Dots Per Inch, determines the resolution of the output image.
    """
    # Unpack the number of rows and columns for the grid
    rows, cols = grid_size

    # Collect binary image data from DataFrame
    subset_df = df.limit(rows * cols).collect()
    image_data_list = [row[data_col] for row in subset_df]
    image_names = [row[label_col] for row in subset_df]

    # Create a matplotlib subplot with the specified grid size
    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)

    # Flatten the axes array for easy iteration if it's 2D
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for ax, binary_data, name in zip(axes, image_data_list, image_names):
        # Convert binary data to an image and display it
        image = Image.open(io.BytesIO(binary_data))

        # Crop image to square if required
        if crop_square:
            min_dim = min(image.size)  # Get the smallest dimension
            width, height = image.size
            left = (width - min_dim) / 2
            top = (height - min_dim) / 2
            right = (width + min_dim) / 2
            bottom = (height + min_dim) / 2
            image = image.crop((left, top, right, bottom))

        ax.imshow(image)
        name = name.replace("_", " ")
        wrapped_name = "\n".join(textwrap.wrap(name, width=25))
        ax.set_title(wrapped_name, fontsize=16, pad=1)
        ax.set_xticks([])
        ax.set_yticks([])
        spines = ["top", "right", "bottom", "left"]
        for s in spines:
            ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_images_from_embeddings(
    df,
    data_col: str,
    label_col: str,
    grid_size: tuple = (3, 3),
    figsize: tuple = (12, 12),
    dpi: int = 80,
):
    """
    Display images in a grid with species names as labels.

    :param df: DataFrame with the embeddings data.
    :param data_col: Name of the data column.
    :param label_col: Name of the species being displayed as image labels.
    :param grid_size: Tuple (rows, cols) representing the grid size.
    :param figsize: Regulates the size of the figure.
    :param dpi: Dots Per Inch, determines the resolution of the output image.
    """
    # Unpack the number of rows and columns for the grid
    rows, cols = grid_size

    # Collect binary image data from DataFrame
    subset_df = df.limit(rows * cols).collect()
    embedding_data_list = [row[data_col] for row in subset_df]
    image_names = [row[label_col] for row in subset_df]

    # Create a matplotlib subplot with specified grid size
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12), dpi=dpi)

    # Flatten the axes array for easy iteration
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for ax, embedding, name in zip(axes, embedding_data_list, image_names):
        # Find the next perfect square size greater than or equal to the embedding length
        next_square = math.ceil(math.sqrt(len(embedding))) ** 2
        padding_size = next_square - len(embedding)

        # Pad the embedding if necessary
        if padding_size > 0:
            embedding = np.pad(
                embedding, (0, padding_size), "constant", constant_values=0
            )

        # Reshape the embedding to a square
        side_length = int(math.sqrt(len(embedding)))
        image_array = np.reshape(embedding, (side_length, side_length))

        # Normalize the embedding to [0, 255] for displaying as an image
        normalized_image = (
            (image_array - np.min(image_array))
            / (np.max(image_array) - np.min(image_array))
            * 255
        )
        image = Image.fromarray(normalized_image).convert("L")

        ax.imshow(image, cmap="gray")
        ax.set_xlabel(name)  # Set the species name as xlabel
        ax.xaxis.label.set_size(14)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def plot_masks_from_binary(
    joined_df,
    image_data_col: str,
    mask_data_col: str,
    label_col: str,
    grid_size=(3, 3),
    crop_square: bool = False,
    figsize: tuple = (12, 12),
    dpi: int = 80,
):
    """
    Display masks in a grid with image names as labels.

    :param joined_df: DataFrame with the original and masked data.
    :param data_col: Name of the original image data column.
    :param mask_data_col: Name of the masked data column.
    :param label_col: Name of the species being displayed as image labels.
    :param grid_size: Tuple (rows, cols) representing the grid size.
    :param crop_square: Boolean, whether to crop images to a square format by taking the center.
    :param figsize: Regulates the size of the figure.
    :param dpi: Dots Per Inch, determines the resolution of the output image.
    """
    # Unpack the number of rows and columns for the grid
    rows, cols = grid_size

    # Collect binary image data from mask DataFrame
    subset_df = joined_df.limit(rows * cols).collect()
    image_data = [row[image_data_col] for row in subset_df]
    mask_image_data = [row[mask_data_col] for row in subset_df]
    image_names = [row[label_col] for row in subset_df]

    # Create a matplotlib subplot with the specified grid size
    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)

    # Flatten the axes array for easy iteration if it's 2D
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for ax, binary_data, mask_binary_data, name in zip(
        axes, image_data, mask_image_data, image_names
    ):
        # Convert binary data to an image
        image = Image.open(io.BytesIO(binary_data))
        image_array = np.array(image)
        mask_array = np.load(io.BytesIO(mask_binary_data))
        mask_array = np.expand_dims(mask_array, axis=-1)
        mask_array = np.repeat(mask_array, 3, axis=-1)
        mask_img = image_array * mask_array

        # Plot the mask
        ax.imshow(mask_img.astype(np.uint8))
        name = name.replace("_", " ")
        wrapped_name = "\n".join(textwrap.wrap(name, width=25))
        ax.set_title(wrapped_name, fontsize=16, pad=1)
        ax.set_xticks([])
        ax.set_yticks([])
        spines = ["top", "right", "bottom", "left"]
        for s in spines:
            ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_individual_masks_comparison(
    joined_df,
    label_col: str,
    grid_size=(3, 5),
    crop_square: bool = False,
    figsize: tuple = (15, 10),
    fontsize: int = 16,
    wrap_width: int = 15,
    dpi: int = 80,
):
    """
    Display masks in a grid with image names as labels.

    :param joined_df: DataFrame with the original and masked data.
    :param label_col: Name of the species being displayed as image labels.
    :param grid_size: Tuple (rows, cols) representing the grid size.
    :param crop_square: Boolean, whether to crop images to a square format by taking the center.
    :param figsize: Regulates the size of the figure.
    :param dpi: Dots Per Inch, determines the resolution of the output image.
    """
    # Unpack the number of rows and columns for the grid
    rows, cols = grid_size

    # Collect binary image data from DataFrame
    subset_df = joined_df.limit(rows).collect()

    # Create subplots for image and masks
    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)

    # Ensure axes is always 2D for consistent iteration
    axes = axes.reshape(rows, cols)

    for row_idx, row in enumerate(subset_df):
        # Load original image
        image = Image.open(io.BytesIO(row["data"])).convert("RGB")
        image_array = np.array(image)

        # Load masks
        mask_names = ["leaf_mask", "flower_mask", "plant_mask", "combined_mask"]
        masks = [np.load(io.BytesIO(row[mask])) for mask in mask_names]

        # Expand masks to match image dimensions
        masks = [np.expand_dims(mask, axis=-1) for mask in masks]  # (H, W, 1)
        masks = [np.repeat(mask, 3, axis=-1) for mask in masks]  # (H, W, 3)

        # Plot original image
        axes[row_idx, 0].imshow(image_array)
        wrapped_name = "\n".join(textwrap.wrap(row[label_col], width=wrap_width))
        axes[row_idx, 0].set_title(wrapped_name, fontsize=fontsize, pad=1)

        # Plot each mask
        for col_idx, (mask, mask_name) in enumerate(zip(masks, mask_names), start=1):
            masked_image = image_array * mask
            axes[row_idx, col_idx].imshow(masked_image.astype(np.uint8))
            name = mask_name.replace("_", " ").title()
            wrap_mask_name = "\n".join(textwrap.wrap(name, width=wrap_width))
            axes[row_idx, col_idx].set_title(wrap_mask_name, fontsize=fontsize, pad=1)

        # Remove ticks and spines
        for ax in axes[row_idx, :]:
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ["top", "right", "bottom", "left"]:
                ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_species_histogram(df, species_count: int = 100, bar_width: float = 0.8):
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

    species_df = (
        df.filter(f"n >= {species_count}").orderBy("n", ascending=False).toPandas()
    )

    # Get the top and bottom 5 species
    top5_df = species_df.head(5)

    # Plot all species
    ax.bar(
        species_df["species"], species_df["n"], color="lightslategray", width=bar_width
    )

    # Highlight the top 5 species in different colors
    if species_count >= 600:
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
        for i, row in top5_df.iterrows():
            ax.bar(
                row["species"],
                row["n"],
                color=colors[i],
                label=row["species"],
                width=bar_width,
            )
        ax.legend(title="Top 5 Species")

    ax.set_xlabel("Species")
    ax.set_ylabel("Count")
    ax.set_title(
        f"PlantCLEF 2024 Histogram of Plant Species with Count >= {species_count}",
        weight="bold",
        fontsize=16,
    )
    ax.set_xticks([])
    ax.set_xmargin(0)
    ax.xaxis.label.set_size(14)  # Set the font size for the xlabel
    ax.yaxis.label.set_size(14)  # Set the font size for the xlabel
    ax.grid(color="blue", linestyle="--", linewidth=1, alpha=0.2)
    spines = ["top", "right", "bottom", "left"]
    for s in spines:
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.show()
