import os
import argparse
from pathlib import Path

from pyspark.sql import functions as F

from plantclef.spark import get_spark

"""
Before running this script, make sure you have downloaded and extracted the dataset into the data folder.
Use the bash file `download_extract_dataset.sh` in the scripts folder.
"""


def get_home_dir():
    """Get the home directory for the current user on PACE."""
    return Path(os.path.expanduser("~"))


def create_spark_dataframe(
    spark, image_base_path: Path, metadata_path: str, metadata_filename: str
):
    """Converts images into binary data and joins with a Metadata DataFrame"""
    # Load all files from the base directory as binary data
    image_df = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", "*.jpg")
        .option("recursiveFileLookup", "true")
        .load(image_base_path.as_posix())
    )

    # Construct the string to be replaced - adjust this based on your actual base path
    base_path_to_remove = "file:" + str(image_base_path.parents[0])

    # Remove "file:{image_base_path.parents[0]" from path column
    image_df = image_df.withColumn(
        "path", F.regexp_replace("path", base_path_to_remove, "")
    )

    # Split the path into an array of elements
    split_path = F.split(image_df["path"], "/")

    # Select and rename columns to fit the target schema, including renaming 'content' to 'data'
    image_df = image_df.select(
        "path",
        F.element_at(split_path, -1).alias("image_name"),
        F.col("content").alias("data"),
    )

    # Read the iNaturalist metadata CSV file
    metadata_df = spark.read.csv(
        f"{metadata_path}/{metadata_filename}.csv",
        header=True,
        inferSchema=True,
        sep=";",  # specify semicolon as delimiter
    )

    # Drop duplicate entries based on 'image_path' before the join
    metadata_df = metadata_df.dropDuplicates(["image_name"])

    # Perform an inner join on the 'image_path' column
    combined_df = image_df.join(metadata_df, "image_name", "inner").repartition(
        500, "species_id"
    )

    return combined_df


def parse_args():
    """Parse command-line arguments."""
    home_dir = get_home_dir()
    dataset_base_path = f"{home_dir}/p-dsgt_clef2025-0/shared/plantclef/data"

    parser = argparse.ArgumentParser(
        description="Process images and metadata for a dataset stored on GCS."
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=os.cpu_count(),
        help="Number of cores used in Spark driver",
    )
    parser.add_argument(
        "--memory",
        type=str,
        default="16g",
        help="Amount of memory to use in Spark driver",
    )
    parser.add_argument(
        "--image-root-path",
        type=str,
        default=f"{dataset_base_path}/train/PlantCLEF2024",
        help="Base directory path for image data",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=f"{dataset_base_path}/metadata",
        help="Root directory path for metadata",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=f"{dataset_base_path}/parquet_files/train",
        help="Path for output Parquet files",
    )
    parser.add_argument(
        "--metadata-filename",
        type=str,
        default="PlantCLEF2024singleplanttrainingdata",
        help="Train Metadata CSV filename (without extension)",
    )

    return parser.parse_args()


def main():
    """Main function that processes data and writes the output dataframe to GCS"""
    args = parse_args()

    # Initialize Spark with settings for using the big-disk-dev VM
    spark = get_spark(
        cores=args.cores, memory=args.memory, **{"spark.sql.shuffle.partitions": 500}
    )

    # Convert root-path to a Path object here
    image_base_path = Path(args.image_root_path)

    # Create image dataframe
    final_df = create_spark_dataframe(
        spark=spark,
        image_base_path=image_base_path,
        metadata_path=args.metadata_path,
        metadata_filename=args.metadata_filename,
    )

    # Write the DataFrame to GCS in Parquet format
    final_df.write.mode("overwrite").parquet(args.output_path)


if __name__ == "__main__":
    main()
