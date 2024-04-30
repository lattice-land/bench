#!/bin/bash

# Exits when an error occurs.
set -e

# I. Define the campaign to run and hardware information.

MZN_SOLVER="org.choco.choco"
VERSION="v4.10.13"
MZN_TIMEOUT=1200000
NUM_JOBS=8

HARDWARE="\"AMD Epyc ROME 7H12@2.6GHz; RAM 256GO\""
SHORT_HARDWARE="aion"
MZN_COMMAND="minizinc --solver $MZN_SOLVER -s --json-stream -t $MZN_TIMEOUT --output-mode json --output-time --output-objective"
INSTANCE_FILE="mzn2021-23.csv"
OUTPUT_DIR=$(pwd)"/../data/campaign/$MZN_SOLVER-$VERSION-$SHORT_HARDWARE"
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
  cp $(realpath "$(dirname "$0")")/slurm.sh $OUTPUT_DIR/
fi

# III. Run the experiments in parallel (one per sockets).

DUMP_PY_PATH=$(pwd)/dump.py

cp $0 $OUTPUT_DIR/ # for replicability.
cp $DUMP_PY_PATH $OUTPUT_DIR/
cp $INSTANCE_FILE $OUTPUT_DIR/

parallel --no-run-if-empty $MULTINODES_OPTION --rpl '{} uq()' --jobs $NUM_JOBS -k --colsep ',' --skip-first-line $MZN_COMMAND {2} {3} '|' python3 $DUMP_PY_PATH $OUTPUT_DIR {1} {2} {3} $MZN_SOLVER :::: $INSTANCE_FILE
