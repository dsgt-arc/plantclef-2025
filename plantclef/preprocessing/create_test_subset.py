from typing_extensions import Annotated

from plantclef.spark import get_spark


def get_subset_dataframe(
    spark,
    test_data_path: str,
    num_rows: int = 20,
):
    """
    Reads a parquet dataset from train_path, computes the top N species by image count,
    and returns a DataFrame filtered to only include images belonging to these species.

    Parameters:
        spark (SparkSession): The active Spark session.
        test_data_path (str): The path to the train parquet files.
        num_rows (int): The number of rows to select (default: 20).

    Returns:
        DataFrame: A subset of the original DataFrame.
    """
    # read the parquet files into a spark DataFrame
    test_df = spark.read.parquet(test_data_path).cache()

    # get subset of top N species
    subset_df = test_df.limit(num_rows).cache()
    return subset_df


def main(
    input_path: Annotated[str, "Path to the input data"],
    output_path: Annotated[str, "Path to the output data"],
    cores: Annotated[int, "Number of cores used in Spark driver"] = 6,
    memory: Annotated[str, "Amount of memory to use in Spark driver"] = "16g",
    num_rows: Annotated[int, "Number of rows include (default: 20)"] = 20,
):
    """
    Main function that processes data and writes the
    output dataframe to plantclef directory on PACE.
    """

    # initialize Spark with settings for the driver
    spark = get_spark(
        cores=cores, memory=memory, **{"spark.sql.shuffle.partitions": 200}
    )

    # get subset dataframe with top N species
    subset_df = get_subset_dataframe(
        spark=spark,
        test_data_path=input_path,
        num_rows=num_rows,
    )

    # write the DataFrame to PACE in Parquet format
    subset_df.write.mode("overwrite").parquet(output_path)
    print(f"Subset dataframe written to: {output_path}")

    # print schema and count
    subset_df.printSchema()
    count = subset_df.count()
    print(f"Number of rows in subset dataframe: {count}")
