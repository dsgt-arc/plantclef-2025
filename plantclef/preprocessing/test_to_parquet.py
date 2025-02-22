"""
Before running this script, make sure you have downloaded and extracted the test dataset into the data folder.
Use the sbatch file `slurm-extract-data.sbatch` in the `plantclef-2025/scripts/sbatch` folder.
"""

import os
from typing_extensions import Annotated
from pathlib import Path

from pyspark.sql import functions as F

from plantclef.spark import get_spark


def get_home_dir():
    """Get the home directory for the current user on PACE."""
    return Path(os.path.expanduser("~"))


def create_spark_test_dataframe(spark, image_base_path: Path):
    # Load all files from the base directory as binary data
    # Convert Path object to string when passing to PySpark
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
    image_final_df = image_df.select(
        F.element_at(split_path, -1).alias("image_name"),
        "path",
        F.col("content").alias("data"),
    ).repartition(500)

    return image_final_df


def main(
    input_path: Annotated[str, "Path to the input data"],
    output_path: Annotated[str, "Path to the output data"],
    cores: Annotated[int, "Number of cores used in Spark driver"] = 6,
    memory: Annotated[str, "Amount of memory to use in Spark driver"] = "16g",
):
    """
    Main function that processes data and writes the
    output dataframe to plantclef directory on PACE.
    """

    # Initialize Spark
    spark = get_spark(
        cores=cores, memory=memory, **{"spark.sql.shuffle.partitions": 20}
    )

    # create path object
    input_path = Path(input_path)

    # Create test image dataframe
    final_df = create_spark_test_dataframe(spark=spark, image_base_path=input_path)

    # Write the DataFrame to PACE in Parquet format
    final_df.write.mode("overwrite").parquet(output_path)
    print(f"Subset dataframe written to: {output_path}")

    # print schema and count
    final_df.printSchema()
    count = final_df.count()
    print(f"Number of rows in subset dataframe: {count}")


if __name__ == "__main__":
    main()
