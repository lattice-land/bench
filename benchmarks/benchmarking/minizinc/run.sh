#!/bin/bash

# Exits when an error occurs.
set -e

# I. Define the campaign to run.

MZN_SOLVER="org.choco.choco"
VERSION="v4.10.13"
MZN_TIMEOUT=1200000
NUM_JOBS=8
MACHINE="aion"
INSTANCE_FILE="mzn2023.csv"

MZN_COMMAND="minizinc --solver $MZN_SOLVER -s --json-stream -t $MZN_TIMEOUT --output-mode json --output-time --output-objective"
OUTPUT_DIR=$(pwd)"/../campaign/$MACHINE/$MZN_SOLVER-$VERSION"
mkdir -p $OUTPUT_DIR

## II. Gather the list of Slurm nodes to run the experiments on many nodes if available.

if [ -n "${SLURM_JOB_NODELIST}" ]; then
  # get host name
  NODES_HOSTNAME="nodes_hostname.txt"
  scontrol show hostname $SLURM_JOB_NODELIST > $NODES_HOSTNAME
  # Collect public key and accept them
  while read -r node; do
      ssh-keyscan "$node" >> ~/.ssh/known_hosts
  done < "$NODES_HOSTNAME"
  MULTINODES_OPTION="--sshloginfile $NODES_HOSTNAME"

  if [ -z "$1" ]; then
    echo "Usage: $0 path_to_slurm_script (we store the slurm script in the campaign for replicability)"
    exit 1
  fi
  cp "$1" $OUTPUT_DIR/ # this is the Slurm script
fi

# III. Run the experiments in parallel (one per sockets).

DUMP_PY_PATH=$(pwd)/minizinc/dump.py

# For replicability.
cp "$0" $OUTPUT_DIR/
cp $DUMP_PY_PATH $OUTPUT_DIR/
cp $INSTANCE_FILE $OUTPUT_DIR/

parallel --no-run-if-empty $MULTINODES_OPTION --rpl '{} uq()' --jobs $NUM_JOBS -k --colsep ',' --skip-first-line $MZN_COMMAND {2} {3} '|' python3 $DUMP_PY_PATH $OUTPUT_DIR {1} {2} {3} $MZN_SOLVER :::: $INSTANCE_FILE
