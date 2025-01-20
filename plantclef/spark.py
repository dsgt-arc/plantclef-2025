import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from pyspark.sql import SparkSession

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def get_spark(
    cores=os.cpu_count(),
    memory=os.environ.get("PYSPARK_DRIVER_MEMORY", "16g"),
    executor_memory=os.environ.get("PYSPARK_EXECUTOR_MEMORY", "1g"),
    local_dir=os.environ.get("SPARK_LOCAL_DIR", "/tmp"),
    app_name="clef",
    **kwargs,
):
    """Get a spark session for a single driver."""
    local_dir = f"{local_dir}/{int(time.time())}"
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    builder = (
        SparkSession.builder.config("spark.driver.memory", memory)
        .config("spark.executor.memory", executor_memory)
        .config("spark.driver.cores", cores)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.maxResultSize", "8g")
        .config("spark.local.dir", local_dir)
    )
    for k, v in kwargs.items():
        builder = builder.config(k, v)
    return builder.appName(app_name).master(f"local[{cores}]").getOrCreate()


@contextmanager
def spark_resource(*args, **kwargs):
    """A context manager for a spark session."""
    spark = None
    try:
        spark = get_spark(*args, **kwargs)
        yield spark
    finally:
        if spark is not None:
            spark.stop()
