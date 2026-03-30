import sys


import subprocess
import os
from datetime import datetime

# Base command
BASE_CMD = [
    "python", "scripts/train_general.py",
    "-d", "/home/velazquez/viralminer/dataset/",
]

paths = [
        # "branch/frequency",
        # "branch/frequency-paper",
        # "branch/lp",
        # "branch/pattern",
        # "branch/pattern-paper",
        "merger/frequency+pattern+lp",
        "merger/lp+frequency",
        "merger/lp+pattern",
        "merger/viraminer",
        "merger/viraminer-paper",
    ]

runs = []
for path in paths:
    runs.append({
        "path" : path,
        "save_dir": f"model_anell_zoo/{path}/",
        "config": f"ready_to_train_files_anell/final/onehot/{path}/init+norm.json",
    })

# Log directory
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

for i, run in enumerate(runs):
    logfile_out = os.path.join(log_dir, f"run_{str(i).zfill(2)}_{run['path'].replace('/', '_')}_output.log")
    logfile_err = os.path.join(log_dir, f"run_{str(i).zfill(2)}_{run['path'].replace('/', '_')}_errors.log")

    cmd = BASE_CMD + [
        "-s", run["save_dir"],
        "-p", run["config"]
    ]

    print(f"Starting run {i} → log: {logfile_out}")

    with open(logfile_out, "w") as f_out, open(logfile_err, "w") as f_err:
        process = subprocess.Popen(
            cmd,
            stdout=f_out,
            stderr=f_err
        )
        process.wait()

    print(f"Finished run {i}")