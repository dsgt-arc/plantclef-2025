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


app = typer.Typer(no_args_is_help=True)


def nvidia_smi():
    """Calls nvidia-smi and returns xml output."""
    cmd = "nvidia-smi -q -x".split()
    res = subprocess.run(cmd, capture_output=True, check=True)
    return res.stdout


def xml2json(xml):
    """Converts nvidia-smi xml output to json."""
    return json.dumps(xmltodict.parse(xml))


@app.command()
def monitor(output: str, interval: int = 30, verbose: bool = False):
    """Monitors nvidia-smi and logs to disk."""
    while True:
        res = nvidia_smi()
        res = xml2json(res)
        with open(output, "a") as f:
            f.write(res)
            f.write("\n")
        if verbose:
            print(f"logged nvidia-smi sleeping for {interval} seconds")
        time.sleep(interval)


@app.command()
def parse(input: str):
    """Parses nvidia-logs.ndjson."""
    from pyspark.sql import SparkSession, functions as F
    from matplotlib import pyplot as plt
    from datetime import datetime

    spark = SparkSession.builder.appName("nvidia-logs").getOrCreate()
    spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

    df = spark.read.json(input)
    sub = df.select(
        F.unix_timestamp("nvidia_smi_log.timestamp", "EEE MMM dd HH:mm:ss yyyy").alias(
            "timestamp"
        ),
        "nvidia_smi_log.gpu.product_name",
        "nvidia_smi_log.gpu.utilization",
        "nvidia_smi_log.gpu.processes.process_info",
    ).orderBy(F.desc("timestamp"))

    gpu_name = sub.first().product_name

    # first plot overall utilization
    util = sub.select(
        "timestamp",
        F.split("utilization.gpu_util", " ")[0].cast("int").alias("gpu_util"),
        F.split("utilization.memory_util", " ")[0].cast("int").alias("memory_util"),
    ).orderBy("timestamp")
    utilpd = util.toPandas()
    ds = datetime.fromtimestamp(utilpd["timestamp"].min()).isoformat()

    plt.figure()
    plt.title(f"GPU utilization on {gpu_name} at {ds}")
    plt.xlabel("elapsed time (minutes)")
    plt.ylabel("utilization")
    ax = plt.gca()
    ts = (utilpd["timestamp"] - utilpd["timestamp"].min()) / 60
    ax.plot(ts, utilpd.gpu_util, label="gpu_util")
    ax.plot(ts, utilpd.memory_util, label="memory_util")
    plt.legend()

    # write to disk
    output = input.replace(".ndjson", "-utilization.png")
    plt.savefig(output)
    print(f"saved utilization plot to {output}")
    output = input.replace(".ndjson", "-utilization.csv")
    utilpd.to_csv(output, index=False)
    print(f"saved utilization data to {output}")

    # now give some basic statistics about the processes
    output = input.replace(".ndjson", "-processes.csv")
    (
        sub.select(
            "timestamp",
            F.explode("process_info").alias("process"),
        )
        .withColumn(
            "used_memory_mb", F.split("process.used_memory", " ")[0].cast("int")
        )
        .groupBy(
            "process.pid",
        )
        .agg(
            # how many intervals was the process running
            F.min("timestamp").alias("start"),
            F.max("timestamp").alias("end"),
            F.count("timestamp").alias("interval_count"),
            # min/max/avg/stddev used memory
            F.min("used_memory_mb").alias("min_used_memory_mb"),
            F.max("used_memory_mb").alias("max_used_memory_mb"),
            F.avg("used_memory_mb").alias("avg_used_memory_mb"),
            F.stddev("used_memory_mb").alias("stddev_used_memory_mb"),
        )
        .withColumn("duration_sec", F.col("end") - F.col("start"))
        .orderBy("pid")
    ).toPandas().to_csv(output, index=False)
    print(f"saved process data to {output}")


if __name__ == "__main__":
    app()
