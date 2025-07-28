from sys import stdin
from pathlib import Path
import sys
import os
import minizinc
import json
import datetime

if os.environ.get("MZN_DEBUG", "OFF") == "ON":
  import logging
  logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

if __name__ == "__main__":
  output_dir = sys.argv[1]
  model_data = Path(sys.argv[2])
  solver = sys.argv[3]
  solver_version = sys.argv[4]
  timeout_ms = sys.argv[5]
  cores = sys.argv[6]
  threads = sys.argv[7]
  arch = sys.argv[8]
  fp = sys.argv[9]
  wac1_threshold = sys.argv[10]
  extras = []
  for i in range(11, len(sys.argv)):
    arg = sys.argv[i].strip().replace(' ', '-')
    if arg != "" and arg != "-s": # we use "-s" when there are "no special options to be used".
      extras.append(arg)
      # Remove leading "-" from extras (these are used for specifying options)
      if extras[-1].startswith("-"):
        extras[-1] = extras[-1][1:]
  fp2 = fp
  if fp == "wac1":
    fp2 += f"_{wac1_threshold}"
  uid = solver.replace('.', '-') + "_" + solver_version + "_" + arch + "_" + fp2 + "_" + model_data.stem
  if cores != "1" or threads != "1":
    uid += f"_{cores}cores_{threads}threads"
  uid += f"_timeout{timeout_ms}ms"
  if len(extras) > 0:
    uid += "_"
    uid += "_".join(extras)

  if(output_dir[-1] == "/"):
    output_dir = output_dir[:-1]
  if(Path(output_dir).exists() == False):
    os.mkdir(output_dir)
  log_filename = Path(output_dir + "/" + uid + ".json")

  stat_base = {
    "configuration": uid,
    "model_data": str(model_data),
    "mzn_solver": solver,
    "version": solver_version,
    "timeout_ms": timeout_ms,
    "datetime": datetime.datetime.now().isoformat(),
    "status": str(minizinc.result.Status.UNKNOWN),
    "cores": cores,
    "threads": threads,
    "arch": arch,
    "fixpoint": fp,
    "wac1_threshold": wac1_threshold
  }

  # If the file exists, we do not delete what is already inside but append new content.
  # We start all benchmarks with a special line {"lattice-land/bench": "start"}.
  print("Writing to file: ", log_filename)
  with open(log_filename, "a") as file:
    header = {"type": "lattice-land", "lattice-land": "start"}
    json.dump(header, file)
    file.write("\n")
    msg = {"type": "statistics", "statistics": stat_base}
    json.dump(msg, file)
    file.write("\n")
    content = ""
    for line in stdin:
      content += line
    msg = {"backtracking": content}
    json.dump(msg, file)
    file.write("\n")
