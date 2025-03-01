import math
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from plottable import ColumnDefinition, Table
from pyspark.sql import functions as F
from .serde import deserialize_image, deserialize_mask
from matplotlib.font_manager import FontProperties

bold_font = FontProperties(weight="bold")


def crop_image_square(image: Image.Image) -> np.ndarray:
    min_dim = min(image.size)  # Get the smallest dimension
    width, height = image.size
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2
    image = image.crop((left, top, right, bottom))
    image_array = np.array(image)
    return image_array


def crop_mask_square(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape[:2]  # Get the dimensions of the mask
    min_dim = min(height, width)  # Get the smallest dimension
    top = (height - min_dim) // 2
    bottom = top + min_dim
    left = (width - min_dim) // 2
    right = left + min_dim
    return mask[top:bottom, left:right]


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
        image = deserialize_image(binary_data)

        # Crop image to square if required
        if crop_square:
            image = crop_image_square(image)

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
        image = deserialize_image(binary_data)
        image_array = np.array(image)
        mask_array = deserialize_mask(mask_binary_data)
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
    mask_names: list = ["leaf_mask", "flower_mask", "rock_mask"],
    positive_classes: list = ["leaf_mask", "flower_mask"],
    label_col: str = "image_name",
    num_rows: int = 3,
    crop_square: bool = False,
    figsize: tuple = (15, 10),
    fontsize: int = 16,
    wrap_width: int = 15,
    dpi: int = 80,
):
    """
    Display masks in a grid with image names as labels.

    :param joined_df: DataFrame with the original and masked data.
    :param mask_names: List of mask names to plot individually.
    :param label_col: Name of the species being displayed as image labels.
    :param grid_size: Tuple (rows, cols) representing the grid size.
    :param crop_square: Boolean, whether to crop images to a square format by taking the center.
    :param figsize: Regulates the size of the figure.
    :param dpi: Dots Per Inch, determines the resolution of the output image.
    """
    # unpack the number of rows and columns for the grid
    cols = len(mask_names) + 2

    # collect binary image data from DataFrame
    subset_df = joined_df.limit(num_rows).collect()

    # create subplots for image and masks
    fig, axes = plt.subplots(num_rows, cols, figsize=figsize, dpi=dpi)

    # ensure axes is always 2D for consistent iteration
    axes = axes.reshape(num_rows, cols)

    for row_idx, row in enumerate(subset_df):
        # load original image
        image = deserialize_image(row["data"]).convert("RGB")
        image_array = np.array(image)

        # Load masks
        masks = {
            mask_name: deserialize_mask(row[mask_name]) for mask_name in mask_names
        }

        # crop image to square if required
        if crop_square:
            image_array = crop_image_square(image)
            for name, mask in masks.items():
                masks[name] = crop_mask_square(mask)

        # Expand masks to match image dimensions
        masks_rgb = {
            name: np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)
            for name, mask in masks.items()
        }  # (H, W, 3)

        # plot original image
        axes[row_idx, 0].imshow(image_array)
        wrapped_name = "\n".join(textwrap.wrap(row[label_col], width=wrap_width))
        axes[row_idx, 0].set_title(wrapped_name, fontsize=fontsize, pad=1)

        # initialize combined mask
        combined_mask = np.zeros_like(next(iter(masks.values())))

        # plot each mask
        for col_idx, mask_name in enumerate(mask_names, start=1):
            mask = masks_rgb[mask_name]

            # Overlay mask onto the original image
            masked_image = image_array * mask
            axes[row_idx, col_idx].imshow(masked_image.astype(np.uint8))

            name = mask_name.replace("_", " ").title()
            wrap_mask_name = "\n".join(textwrap.wrap(name, width=wrap_width))
            axes[row_idx, col_idx].set_title(wrap_mask_name, fontsize=fontsize, pad=1)

            # Combine masks if they are in positive_classes
            if mask_name in positive_classes:
                combined_mask |= masks[mask_name]  # Use bitwise OR to merge

        # plot combined positive masks
        combined_mask_rgb = np.clip(combined_mask, 0, 1)  # mask is binary
        combined_mask_rgb = np.repeat(
            np.expand_dims(combined_mask, axis=-1), 3, axis=-1
        )  # (H, W, 3)
        combined_overlay = (image_array * combined_mask_rgb).astype(np.uint8)

        axes[row_idx, -1].imshow(combined_overlay)
        wrap_mask_name = "\n".join(
            textwrap.wrap("Combined Positive Masks", width=wrap_width)
        )
        axes[row_idx, -1].set_title(wrap_mask_name, fontsize=fontsize, pad=1)

        # remove ticks and spines
        for ax in axes[row_idx, :]:
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ["top", "right", "bottom", "left"]:
                ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_mask_percentage(
    joined_df,
    image_name: str = "CBN-Pyr-03-20230706.jpg",
    positive_classes: list = ["leaf_mask", "flower_mask"],
    figsize: tuple = (15, 10),
    fontsize: int = 16,
    dpi: int = 80,
):
    # Create figure with GridSpec layout
    # Top: image + table, Bottom: 2x5 masks
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])
    # Top section: Image and Table
    top_gs = gs[0].subgridspec(1, 2, width_ratios=[1, 1])

    # Load the original image
    selected_df = joined_df.where(F.col("image_name") == image_name)
    row = selected_df.first()
    img = deserialize_image(row.data)
    mask_cols = [c for c in selected_df.columns if "mask" in c]
    # convert image to numpy array for combined masking
    image_array = np.array(img)

    # Left: Display original image
    ax_img = fig.add_subplot(top_gs[0, 0])
    ax_img.imshow(img)
    ax_img.axis("off")
    wrapped_name = "\n".join(textwrap.wrap(image_name, width=25))
    ax_img.set_title(wrapped_name, fontsize=fontsize, pad=1)
    # ax_img.set_title(image_name)

    # Right: Display tabular information
    ax_table = fig.add_subplot(top_gs[0, 1])
    mask_mean = {}
    for col in mask_cols:
        mask = deserialize_mask(row[col])
        mask_mean[col] = mask.mean()
    pdf = pd.DataFrame(list(mask_mean.items()), columns=["mask", "percentage"])

    # Create table data
    pdf["mask"] = pdf["mask"].str.replace("_", " ").str.title()
    pdf["percentage"] = pdf["percentage"].round(3)
    pdf = pdf.set_index("mask")
    pdf.index.name = "Mask"
    pdf.columns = [col.title() for col in pdf.columns]

    # Define the Matplotlib axis
    ax_table.axis("off")
    col_defs = [
        ColumnDefinition(
            name="Mask",
            textprops={"fontsize": 16, "ha": "left"},
            width=1,
            title="Mask",
        ),
        ColumnDefinition(
            name="Percentage",
            textprops={"fontsize": 16, "ha": "left"},
            width=1,
            title="Mask Percentage",
        ),
    ]
    # Use plottable to create a modern table
    Table(
        pdf,
        column_definitions=col_defs,
        row_dividers=True,
        footer_divider=True,
        textprops={"fontsize": 14, "ha": "left"},
        row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
        col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
        column_border_kw={"linewidth": 1, "linestyle": "-"},
    )

    # Bottom section: 2x5 mask images
    bottom_gs = gs[1].subgridspec(2, 5)

    # Load masks
    masks = {mask_name: deserialize_mask(row[mask_name]) for mask_name in mask_cols}
    combined_mask = np.zeros_like(next(iter(masks.values())))
    masks["combined_mask"] = combined_mask
    # plot each mask
    for i, ((mask_name, mask), ax_pos) in enumerate(zip(masks.items(), bottom_gs)):
        ax = fig.add_subplot(ax_pos)
        col_name = mask_name.replace("_", " ").title()
        # combine masks if they are in positive_classes
        if mask_name in positive_classes:
            masks["combined_mask"] |= masks[mask_name]  # Use bitwise OR to merge

        # plot combined positive masks
        if i == len(masks) - 1:
            combined_mask_rgb = np.clip(combined_mask, 0, 1)  # mask is binary
            combined_mask_rgb = np.repeat(
                np.expand_dims(combined_mask, axis=-1), 3, axis=-1
            )  # (H, W, 3)
            mask = (image_array * combined_mask_rgb).astype(np.uint8)
            col_name = "Combined Positive Masks"

        ax.imshow(mask, cmap="gray")
        wrapped_name = "\n".join(textwrap.wrap(col_name, width=14))
        ax.set_title(wrapped_name, fontsize=fontsize, pad=1)
        ax.axis("off")

    # Adjust layout and display
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
