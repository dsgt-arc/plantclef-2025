import luigi

from pyspark.sql import DataFrame
from plantclef.spark import spark_resource


class ProcessMaskOverlay(luigi.Task):
    """Task to process the overlay masks."""

    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    sample_col = luigi.Parameter()
    num_partitions = luigi.IntParameter()
    sample_id = luigi.IntParameter()
    num_sample_ids = luigi.IntParameter()
    cpu_count = luigi.IntParameter()
    sql_statement = luigi.Parameter(default="SELECT * FROM __THIS__")

    def requires(self):
        pass

    def transform(self, df: DataFrame) -> DataFrame:
        return df

    def run(self):
        kwargs = {"cores": self.cpu_count}
        with spark_resource(**kwargs) as spark:
            df = spark.read.parquet(self.input_path)
            df.printSchema()
            df.show()
