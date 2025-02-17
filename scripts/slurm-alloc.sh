#!/usr/bin/env python
from argparse import ArgumentParser
import subprocess

parser = ArgumentParser()
parser.add_argument(
    "--account",
    type=str,
    default="paceship-dsgt_clef2025",
    help="Slurm account to use.",
)
parser.add_argument(
    "--gpu",
    action="store_true",
    help="Request a GPU.",
)
args = parser.parse_args()

# salloc -A paceship-dsgt_clef2025 -N1 -n1 -c2 --mem-per-cpu=4G -t1:00:00
cmd = [
    "salloc",
    f"--account={args.account}",
    "--nodes=1",
    "--ntasks=1",
    "--cpus-per-task=4",
    "--time=1:00:00",
    "--qos=inferno",
]
if args.gpu:
    cmd += ["--gres=gpu:v100:1"]
cmd += ["--mem-per-gpu=32G" if args.gpu else "--mem-per-cpu=4G"]
cmd = " ".join(cmd)
print(cmd)
subprocess.run(cmd, shell=True)
