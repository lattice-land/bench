#!/bin/bash -l
#SBATCH --time=00:01:00
#SBATCH --nodes=4
#SBATCH --partition=batch
#SBATCH --ntasks-per-node=1 # when benchmarking sequential solver, we still book the whole node to avoid possible interference.
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --export=ALL
#SBATCH --output=slurm-ace.out

# Exits when an error occurs.
set -e
set -x # useful for debugging.

# Shortcuts of paths to benchmarking directories.
ACE_WORKFLOW_PATH=$(dirname $(realpath "run.sh"))
BENCHMARKING_DIR_PATH="$ACE_WORKFLOW_PATH/.."
BENCHMARKS_DIR_PATH="$ACE_WORKFLOW_PATH/../.."

# Configure the environment.
if [ -z "$1" ]; then
  echo "Usage: $0 <machine.sh>"
  echo "  Name of the machine running the experiments with the configuration of the environment."
  exit 1
fi
source $1

# If it has an argument, we retry the jobs that failed on a previous run.
# If the experiments were not complete, you can simply rerun the script, parallel will ignore the jobs that are already done.
if [ -n "$2" ]; then
  parallel --retry-failed --joblog $2
  exit 0
fi

# I. Define the campaign to run.

TIMEOUT=1200000
CORES=1 # The number of core used on the node.
THREADS=1 # The number of threads used by the solver.
MACHINE=$(basename "$1" ".sh")
INSTANCES_PATH="$BENCHMARKS_DIR_PATH/benchmarking/xcsp22_minicop.csv"

MEM_GB_PER_XP=32 # similar to Minizinc competition.
# II. Prepare the command lines and output directory.
VERSION= "2.3"
SOLVER= "ACE-$VERSION"
ACE_COMMAND="java -Xmx${MEM_GB_PER_XP}g -jar $HOME/deps/ACE/build/libs/$SOLVER.jar"
ACE_OPTIONS="-t=$TIMEOUT -npc=True -ev" # be careful, in ACE the options must be situed after the instance file.
OUTPUT_DIR="$BENCHMARKS_DIR_PATH/campaign/$MACHINE/$SOLVER"
mkdir -p $OUTPUT_DIR

# If we are on the HPC, we encapsulate the command in a srun command to reserve the resources needed.
if [ -n "${SLURM_JOB_NODELIST}" ]; then
  SRUN_COMMAND="srun --exclusive --cpus-per-task=$CORES --nodes=1 --ntasks=1 --cpu-bind=verbose"
  NUM_PARALLEL_EXPERIMENTS=$SLURM_JOB_NUM_NODES # How many experiments are we running in parallel? One per node per default.
else
  NUM_PARALLEL_EXPERIMENTS=1
fi

# For replicability.
cp -r $ACE_WORKFLOW_PATH $OUTPUT_DIR/
cp $INSTANCES_PATH $OUTPUT_DIR/$(basename "$ACE_WORKFLOW_PATH")/

# Store the description of the hardware on which this campaign is run.
lshw -json > $OUTPUT_DIR/$(basename "$ACE_WORKFLOW_PATH")/hardware-"$MACHINE".json 2> /dev/null

# III. Run the experiments in parallel.
# The `parallel` command spawns one `srun` command per experiment, which executes the minizinc solver with the right resources.

COMMANDS_LOG="$OUTPUT_DIR/$(basename "$ACE_WORKFLOW_PATH")/jobs.log"
parallel --verbose --no-run-if-empty --rpl '{} uq()' -k --colsep ',' --skip-first-line -j $NUM_PARALLEL_EXPERIMENTS --resume --joblog $COMMANDS_LOG $SRUN_COMMAND $ACE_COMMAND $BENCHMARKING_DIR_PATH/{3} $ACE_OPTIONS '2>&1' '|' python3 dump.py $OUTPUT_DIR/{1}"_"{2}.log $SOLVER $VERSION $CORES $THREADS $TIMEOUT $MEM_GB_PER_XP $BENCHMARKING_DIR_PATH/{3} $ACE_OPTIONS :::: $INSTANCES_PATH
