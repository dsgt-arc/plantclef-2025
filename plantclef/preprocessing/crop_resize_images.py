import argparse

import cv2
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.types import BinaryType

from plantclef.spark import get_spark
from plantclef.config import get_home_dir


# Define a UDF to crop and resize images
def crop_resize_images(data, target_width=256, target_height=256):
    """
    Crop the center of the image and resize it to the specific size.

    Args:
        data (bytes): The binary data of the image.
        size (tuple): the target size of the image after cropping and resizing.
    Returns:
        bytes: The binary data of the cropped and resized image.
    """
    # Convert binary data to NumPy array, then to image
    image = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Determine the size for cropping to a square
    height, width = image.shape[:2]
    crop_size = min(height, width)

    # Calculate crop coordinates to get the center square
    start_x = width // 2 - crop_size // 2
    start_y = height // 2 - crop_size // 2

    # Crop the center square
    image_cropped = image[start_y : start_y + crop_size, start_x : start_x + crop_size]

    # Resize the image
    target_size = target_width, target_height
    image_resized = cv2.resize(image_cropped, target_size, interpolation=cv2.INTER_AREA)

    # Convert the image back to binary data
    _, img_encoded = cv2.imencode(".jpg", image_resized)
    img_byte_arr = img_encoded.tobytes()

    return img_byte_arr


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process images and metadata for a dataset stored on PACE."
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=6,
        help="Number of cores used in Spark driver",
    )
    parser.add_argument(
        "--memory",
        type=str,
        default="16g",
        help="Amount of memory to use in Spark driver",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="subset_top5_train",
        help="Dataset name downloaded from tar file",
    )
    parser.add_argument(
        "--square-size",
        type=int,
        default=128,
        help="Size of the resized image. Default is 128x128",
    )
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=500,
        help="Number of partitions for the output DataFrame",
    )

    return parser.parse_args()


def main():
    """Main function that processes data and writes the output dataframe to PACE."""
    args = parse_args()

    # Initialize Spark with settings for using the big-disk-dev VM
    spark = get_spark(
        cores=args.cores,
        memory=args.memory,
        **{"spark.sql.shuffle.partitions": args.num_partitions},
    )

    # set input and output paths
    home_dir = get_home_dir()
    data_path = f"{home_dir}/p-dsgt_clef2025-0/shared/plantclef/data/parquet"
    input_path = f"{data_path}/{args.dataset_name}"
    output_path = f"{data_path}/crop_resize_{args.square_size}_{args.dataset_name}"

    # Read the Parquet file into a DataFrame
    df = spark.read.parquet(input_path)

    # Register the UDF with BinaryType return type
    crop_resize_udf = F.udf(crop_resize_images, BinaryType())

    # Apply the UDF to crop and resize the images
    crop_df = df.withColumn(
        "cropped_image_data",
        crop_resize_udf(
            F.col("data"), F.lit(args.square_size), F.lit(args.square_size)
        ),
    )

    # Drop the original 'data' column and rename 'cropped_image_data' to 'data'
    final_df = (
        crop_df.drop("data")
        .withColumnRenamed("cropped_image_data", "data")
        .repartition(args.num_partitions, "species_id")
    )

    # Write the DataFrame to PACE in Parquet format
    final_df.write.mode("overwrite").parquet(output_path)
    print(f"Subset dataframe written to: {output_path}")

    # check number of rows in the final dataframe
    count = final_df.count()
    print(f"Number of rows in subset dataframe: {count}")


if __name__ == "__main__":
    main()
