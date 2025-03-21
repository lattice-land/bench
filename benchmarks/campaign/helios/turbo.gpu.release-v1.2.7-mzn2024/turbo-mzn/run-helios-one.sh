#!/bin/bash -l
#SBATCH --time=00:10:00
#SBATCH -p plgrid-gpu-gh200
#SBATCH -A plgturbo-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 72
#SBATCH --mem=0
#SBATCH --qos=normal
#SBATCH --export=ALL
#SBATCH --output=slurm-turbo-gpu-mzn2024.out

# Exits when an error occurs.
set -e
set -x # useful for debugging.

# Shortcuts of paths to benchmarking directories.
MZN_WORKFLOW_PATH=$(dirname $(realpath "run-helios.sh"))
BENCHMARKING_DIR_PATH="$MZN_WORKFLOW_PATH/.."
BENCHMARKS_DIR_PATH="$MZN_WORKFLOW_PATH/../.."

# Configure the environment.
if [ -z "$1" ]; then
  echo "Usage: $0 <machine.sh>"
  echo "  Name of the machine running the experiments with the configuration of the environment."
  exit 1
fi
source ${MZN_WORKFLOW_PATH}/$1
source ${BENCHMARKS_DIR_PATH}/../pybench/bin/activate

# If it has an argument, we retry the jobs that failed on a previous run.
# If the experiments were not complete, you can simply rerun the script, parallel will ignore the jobs that are already done.
if [ -n "$2" ]; then
  parallel --retry-failed --joblog $2
  exit 0
fi

# I. Define the campaign to run.

MZN_SOLVER="turbo.gpu.release"
VERSION="v1.2.7" # Note that this is only for the naming of the output directory, we do not verify the actual version of the solver.
# This is to avoid MiniZinc to kill Turbo before it can print the statistics.
MZN_TIMEOUT=3600000
REAL_TIMEOUT=300000
ARCH="hybrid"
CORES=72 # The number of core used on the node.
THREADS=264 # The number of core used on the node.
FP="wac1"
WAC1_THRESHOLD=4096
MACHINE=$(basename "$1" ".sh")
INSTANCES_PATH="$BENCHMARKS_DIR_PATH/benchmarking/mzn2024_patch.csv"

# II. Prepare the command lines and output directory.
MZN_COMMAND="minizinc --solver $MZN_SOLVER -s --json-stream -t $MZN_TIMEOUT --output-mode json --output-time --output-objective -p $THREADS -arch $ARCH -fp $FP -wac1_threshold $WAC1_THRESHOLD -hardware $MACHINE -version $VERSION -timeout $REAL_TIMEOUT -globalmem "
OUTPUT_DIR="$BENCHMARKS_DIR_PATH/campaign/$MACHINE/$MZN_SOLVER-$VERSION-mzn2024"
mkdir -p $OUTPUT_DIR

# If we are on the HPC, we encapsulate the command in a srun command to reserve the resources needed.
if [ -n "${SLURM_JOB_NODELIST}" ]; then
  SRUN_COMMAND="srun --exclusive --cpus-per-task=$CORES --gpus-per-task=1 --nodes=1 --ntasks=1 --cpu-bind=verbose"
  NUM_PARALLEL_EXPERIMENTS=$((SLURM_JOB_NUM_NODES * 1)) # How many experiments are we running in parallel? One per GPU per default.
else
  NUM_PARALLEL_EXPERIMENTS=1
fi

DUMP_PY_PATH="$MZN_WORKFLOW_PATH/dump.py"

# For replicability.
cp -r $MZN_WORKFLOW_PATH $OUTPUT_DIR/
cp $INSTANCES_PATH $OUTPUT_DIR/$(basename "$MZN_WORKFLOW_PATH")/

# Store the description of the hardware on which this campaign is run.
# lshw -json > $OUTPUT_DIR/$(basename "$MZN_WORKFLOW_PATH")/hardware-"$MACHINE".json 2> /dev/null

# III. Run the experiments in parallel.
# The `parallel` command spawns one `srun` command per experiment, which executes the minizinc solver with the right resources.

minizinc --solver turbo.gpu.release -s --json-stream -t 360000 --output-mode json --output-time --output-objective -p 132 -arch hybrid -fp wac1 -wac1_threshold 4096 -hardware helios -version v1.2.7 -timeout 300000 -globalmem -sub 15 --fzn air9.fzn  /net/scratch/hscra/plgrid/plgptalbot/lattice-land/bench/benchmarks/benchmarking/turbo-mzn/../../data/mzn-challenge/2024/aircraft-disassembly/aircraft.mzn /net/scratch/hscra/plgrid/plgptalbot/lattice-land/bench/benchmarks/benchmarking/turbo-mzn/../../data/mzn-challenge/2024/aircraft-disassembly/B737NG-600-09-Anon.json.dzn
