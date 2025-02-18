#!/usr/bin/env python
# /// script
# dependencies = ["xmltodict", "typer", "pyspark", "matplotlib", "pandas"]
# ///

"""Script for logging nvidia-smi calls to disk."""

import json
import subprocess
import time
import typer
import xmltodict
import os

app = typer.Typer(no_args_is_help=True)


def nvidia_smi():
    """Calls nvidia-smi and returns XML output as a string."""
    cmd = "nvidia-smi -q -x".split()
    res = subprocess.run(cmd, capture_output=True, check=True)
    return res.stdout.decode("utf-8")  # Decode bytes to str


def xml2json(xml):
    """Converts nvidia-smi XML output to JSON."""
    return json.dumps(xmltodict.parse(xml))


@app.command()
def monitor(output: str, interval: int = 30, verbose: bool = False):
    """Monitors nvidia-smi and logs to disk."""
    # Determine and print the absolute path where the log file will be written
    abs_output = os.path.abspath(output)
    print("Logging file will be written to:", abs_output, flush=True)

    while True:
        xml_output = nvidia_smi()
        json_output = xml2json(xml_output)
        with open(output, "a") as f:
            f.write(json_output + "\n")
            f.flush()  # Ensure immediate write to disk
        if verbose:
            print(
                f"Logged nvidia-smi output; sleeping for {interval} seconds", flush=True
            )
        time.sleep(interval)


@app.command()
def parse(input: str):
    """Parses nvidia-logs.ndjson."""
    from pyspark.sql import SparkSession, functions as F
    from pyspark.sql.types import ArrayType, StructType, StructField, StringType
    from matplotlib import pyplot as plt
    from datetime import datetime

    spark = SparkSession.builder.appName("nvidia-logs").getOrCreate()
    spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

    # Define the schema for process_info
    process_schema = ArrayType(
        StructType([
            StructField("compute_instance_id", StringType(), True),
            StructField("gpu_instance_id", StringType(), True),
            StructField("pid", StringType(), True),
            StructField("process_name", StringType(), True),
            StructField("type", StringType(), True),
            StructField("used_memory", StringType(), True),
        ])
    )

    # Read the JSON log file
    df = spark.read.json(input)

    # Create a new column "process_info" that always holds an array:
    # - If nvidia_smi_log.gpu.processes is a string ("None"), use an empty array.
    # - Else if nvidia_smi_log.gpu.processes.process_info is already an array, leave it as is.
    # - Else wrap the struct into an array.
    df = df.withColumn(
        "process_info",
        F.when(
            F.col("nvidia_smi_log.gpu.processes").cast("string") == "None",
            F.lit([]).cast(process_schema)
        ).otherwise(
            F.when(
                # Check if process_info is already an array (has an element at index 0)
                F.col("nvidia_smi_log.gpu.processes.process_info").getItem(0).isNotNull(),
                F.col("nvidia_smi_log.gpu.processes.process_info")
            ).otherwise(
                F.array(F.col("nvidia_smi_log.gpu.processes.process_info"))
            )
        )
    )

    # Now select the relevant fields, using our new process_info column
    sub = df.select(
        F.unix_timestamp("nvidia_smi_log.timestamp", "EEE MMM dd HH:mm:ss yyyy").alias("timestamp"),
        "nvidia_smi_log.gpu.product_name",
        "nvidia_smi_log.gpu.utilization",
        "process_info"
    ).orderBy(F.desc("timestamp"))

    gpu_name = sub.first().product_name

    # First: plot overall GPU utilization
    util = sub.select(
        "timestamp",
        F.split("utilization.gpu_util", " ")[0].cast("int").alias("gpu_util"),
        F.split("utilization.memory_util", " ")[0].cast("int").alias("memory_util"),
    ).orderBy("timestamp")
    utilpd = util.toPandas()
    ds = datetime.fromtimestamp(utilpd["timestamp"].min()).isoformat()

    plt.figure()
    plt.title(f"GPU utilization on {gpu_name} at {ds}")
    plt.xlabel("Elapsed time (minutes)")
    plt.ylabel("Utilization")
    ax = plt.gca()
    ts = (utilpd["timestamp"] - utilpd["timestamp"].min()) / 60
    ax.plot(ts, utilpd.gpu_util, label="gpu_util")
    ax.plot(ts, utilpd.memory_util, label="memory_util")
    plt.legend()

    output_png = input.replace(".ndjson", "-utilization.png")
    plt.savefig(output_png)
    print(f"Saved utilization plot to {output_png}", flush=True)

    output_csv = input.replace(".ndjson", "-utilization.csv")
    utilpd.to_csv(output_csv, index=False)
    print(f"Saved utilization data to {output_csv}", flush=True)

    # Next: process the process_info field and compute basic statistics
    output_process = input.replace(".ndjson", "-processes.csv")
    (
        sub.select("timestamp", F.explode("process_info").alias("process"))
        .withColumn("used_memory_mb", F.split(F.col("process.used_memory"), " ")[0].cast("int"))
        .groupBy("process.pid")
        .agg(
            F.min("timestamp").alias("start"),
            F.max("timestamp").alias("end"),
            F.count("timestamp").alias("interval_count"),
            F.min("used_memory_mb").alias("min_used_memory_mb"),
            F.max("used_memory_mb").alias("max_used_memory_mb"),
            F.avg("used_memory_mb").alias("avg_used_memory_mb"),
            F.stddev("used_memory_mb").alias("stddev_used_memory_mb"),
        )
        .withColumn("duration_sec", F.col("end") - F.col("start"))
        .orderBy("pid")
        .toPandas()
        .to_csv(output_process, index=False)
    )
    print(f"Saved process data to {output_process}", flush=True)



if __name__ == "__main__":
    app()
