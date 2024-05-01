#!/bin/bash

# Exits when an error occurs.
set -e

BENCH_PATH=$(dirname $(realpath "$0"))
BENCH_ROOT_PATH="$BENCH_PATH/../../"

# I. Define the campaign to run.

MZN_SOLVER="org.choco.choco"
VERSION="v4.10.13"
MZN_TIMEOUT=1200000
NUM_JOBS=8
MACHINE="aion"
INSTANCES_PATH="$BENCH_ROOT_PATH/benchmarking/mzn2023.csv"

MZN_COMMAND="minizinc --solver $MZN_SOLVER -s --json-stream -t $MZN_TIMEOUT --output-mode json --output-time --output-objective"
OUTPUT_DIR="$BENCH_ROOT_PATH/campaign/$MACHINE/$MZN_SOLVER-$VERSION"
mkdir -p $OUTPUT_DIR

## II. Gather the list of Slurm nodes to run the experiments on many nodes if available.

if [ -n "${SLURM_JOB_NODELIST}" ]; then
  # get host name
  NODES_HOSTNAME="$BENCH_PATH/nodes_hostname.txt"
  scontrol show hostname $SLURM_JOB_NODELIST > $NODES_HOSTNAME
  # Collect public key and accept them
  while read -r node; do
      ssh-keyscan "$node" >> ~/.ssh/known_hosts
  done < "$NODES_HOSTNAME"
  MULTINODES_OPTION="--sshloginfile $NODES_HOSTNAME"
fi

# III. Run the experiments in parallel (one per sockets).

DUMP_PY_PATH="$BENCH_PATH/dump.py"

# For replicability.
cp -r $BENCH_PATH $OUTPUT_DIR/
cp $INSTANCES_PATH $OUTPUT_DIR/$(basename "$BENCH_PATH")/

parallel --no-run-if-empty $MULTINODES_OPTION --rpl '{} uq()' --jobs $NUM_JOBS -k --colsep ',' --skip-first-line $MZN_COMMAND $BENCH_PATH/{2} $BENCH_PATH/{3} '|' python3 $DUMP_PY_PATH $OUTPUT_DIR {1} {2} {3} $MZN_SOLVER :::: $INSTANCES_PATH
