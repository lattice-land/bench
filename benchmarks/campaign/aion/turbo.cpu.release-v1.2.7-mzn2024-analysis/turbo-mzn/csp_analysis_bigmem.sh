#!/bin/bash -l
#SBATCH --time=05:00:00
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --qos=normal
#SBATCH --export=ALL
#SBATCH --output=slurm-turbo-analysis_cpu_bigmem.out

# Exits when an error occurs.
set -e
set -x # useful for debugging.

# Shortcuts of paths to benchmarking directories.
MZN_WORKFLOW_PATH=$(dirname $(realpath "run.sh"))
BENCHMARKING_DIR_PATH="$MZN_WORKFLOW_PATH/.."
BENCHMARKS_DIR_PATH="$MZN_WORKFLOW_PATH/../.."

# Configure the environment.
if [ -z "$1" ]; then
  echo "Usage: $0 <machine.sh>"
  echo "  Name of the machine running the experiments with the configuration of the environment."
  exit 1
fi
source $1
source ${BENCHMARKS_DIR_PATH}/../pybench/bin/activate

# If it has an argument, we retry the jobs that failed on a previous run.
# If the experiments were not complete, you can simply rerun the script, parallel will ignore the jobs that are already done.
if [ -n "$2" ]; then
  parallel --retry-failed --joblog $2
  exit 0
fi

# I. Define the campaign to run.

MZN_SOLVER="turbo.cpu.release"
VERSION="v1.2.7" # Note that this is only for the naming of the output directory, we do not verify the actual version of the solver.
ARCH="cpu"
CORES=1 # The number of core used on the node.
THREADS=1 # The number of core used on the node.
MACHINE=$(basename "$1" ".sh")
INSTANCES_PATH="$BENCHMARKS_DIR_PATH/benchmarking/mzn2024_bigmem.csv"

# II. Prepare the command lines and output directory.
MZN_COMMAND="minizinc --solver $MZN_SOLVER -s --json-stream --output-mode json --output-time --output-objective -p $THREADS -arch $ARCH -network_analysis -cutnodes 1 -hardware $MACHINE -version $VERSION"
OUTPUT_DIR="$BENCHMARKS_DIR_PATH/campaign/$MACHINE/$MZN_SOLVER-$VERSION-mzn2024-analysis"
mkdir -p $OUTPUT_DIR

# If we are on the HPC, we encapsulate the command in a srun command to reserve the resources needed.
if [ -n "${SLURM_JOB_NODELIST}" ]; then
  SRUN_COMMAND="srun --exclusive --cpus-per-task=$CORES --nodes=1 --ntasks=1 --cpu-bind=verbose"
  NUM_PARALLEL_EXPERIMENTS=$((SLURM_JOB_NUM_NODES * 1)) # How many experiments are we running in parallel? One per GPU per default.
else
  NUM_PARALLEL_EXPERIMENTS=1
fi

DUMP_PY_PATH="$MZN_WORKFLOW_PATH/dump.py"

# For replicability.
cp -r $MZN_WORKFLOW_PATH $OUTPUT_DIR/
cp $INSTANCES_PATH $OUTPUT_DIR/$(basename "$MZN_WORKFLOW_PATH")/

# Store the description of the hardware on which this campaign is run.
lshw -json > $OUTPUT_DIR/$(basename "$MZN_WORKFLOW_PATH")/hardware-"$MACHINE".json 2> /dev/null

# III. Run the experiments in parallel.
# The `parallel` command spawns one `srun` command per experiment, which executes the minizinc solver with the right resources.

COMMANDS_LOG="$OUTPUT_DIR/$(basename "$MZN_WORKFLOW_PATH")/jobs.log"
parallel --verbose --no-run-if-empty --rpl '{} uq()' -k --colsep ',' --skip-first-line -j $NUM_PARALLEL_EXPERIMENTS --joblog $COMMANDS_LOG $SRUN_COMMAND $MZN_COMMAND $BENCHMARKING_DIR_PATH/{2} $BENCHMARKING_DIR_PATH/{3} '2>&1' '|' python3 $DUMP_PY_PATH $OUTPUT_DIR {1} {2} {3} $MZN_SOLVER $VERSION "0" $CORES $THREADS $ARCH "ac1" "0" :::: $INSTANCES_PATH
