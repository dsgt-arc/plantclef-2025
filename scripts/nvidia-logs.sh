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
            print(f"Logged nvidia-smi output; sleeping for {interval} seconds", flush=True)
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

    try:
        df = spark.read.json(input)
        sub = df.select(
            F.unix_timestamp("nvidia_smi_log.timestamp", "EEE MMM dd HH:mm:ss yyyy").alias("timestamp"),
            "nvidia_smi_log.gpu.product_name",
            "nvidia_smi_log.gpu.utilization",
            "nvidia_smi_log.gpu.processes.process_info",
        ).orderBy(F.desc("timestamp"))
    except:
        print("Error reading JSON file. Please ensure the file is in the correct format.", flush=True)
        return

    gpu_name = sub.first().product_name

    # Plot overall utilization
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

    # Save plot and CSV data to disk
    output_png = input.replace(".ndjson", "-utilization.png")
    plt.savefig(output_png)
    print(f"Saved utilization plot to {output_png}", flush=True)

    output_csv = input.replace(".ndjson", "-utilization.csv")
    utilpd.to_csv(output_csv, index=False)
    print(f"Saved utilization data to {output_csv}", flush=True)

    # Define the schema for process info
    process_schema = StructType([
        StructField("compute_instance_id", StringType(), True),
        StructField("gpu_instance_id", StringType(), True),
        StructField("pid", StringType(), True),
        StructField("process_name", StringType(), True),
        StructField("type", StringType(), True),
        StructField("used_memory", StringType(), True)
    ])

    # Choose the correct column for explosion:
    if isinstance(sub.schema["process_info"].dataType, ArrayType):
        process_col = F.explode("process_info")
    else:
        # Wrap the struct in an array if it's not already an array
        process_col = F.explode(F.array("process_info"))

    output_process = input.replace(".ndjson", "-processes.csv")
    (
        sub.select(
            "timestamp",
            process_col.alias("process")
        )
        # Force conversion: if 'process' is a string, convert it to a struct using the defined schema.
        .withColumn("process", F.from_json(F.col("process"), process_schema))
        .withColumn("used_memory_mb", F.split(F.col("process.used_memory"), " ")[0].cast("int"))
        .groupBy("process.pid")
        .agg(
            F.min("timestamp").alias("start"),
            F.max("timestamp").alias("end"),
            F.count("timestamp").alias("interval_count"),
            F.min("used_memory_mb").alias("min_used_memory_mb"),
            F.max("used_memory_mb").alias("max_used_memory_mb"),
            F.avg("used_memory_mb").alias("avg_used_memory_mb"),
            F.stddev("used_memory_mb").alias("stddev_used_memory_mb")
        )
        .withColumn("duration_sec", F.col("end") - F.col("start"))
        .orderBy("pid")
    ).toPandas().to_csv(output_process, index=False)
    print(f"Saved process data to {output_process}", flush=True)

if __name__ == "__main__":
    app()
