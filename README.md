# Benchmarking Constraint Solvers

In this document, we discuss our workflow to benchmark constraint solvers.

## Prerequisites

```
git clone --recursive git@github.com:lattice-land/bench.git
cd bench
salloc -p interactive --qos debug -C batch    # skip if not on HPC
module load lang/Python/3.8.6-GCCcore-10.2.0  # skip if not on HPC
python -m venv benchmarks/pybench
source benchmarks/pybench/bin/activate
pip install mzn-bench
```

## Structure

We adopt the _convention over configuration_ philosophy, meaning we expect files to be structured in a certain way.
By cloning this project, you will have the following directory structure:

* benchmarks/
  - analyzing/
    - analysis.ipynb
    - analysis.py
    - mznbench_check.sh
    - mznbench_collect.sh
  - benchmarking/
    - configure.sh
    - dump.py
    - run-mzn.sh
    - slurm.sh
  - campaign/
  - [mzn-challenge/](https://github.com/MiniZinc/mzn-challenge/)
  - [minizinc-benchmarks/](https://github.com/MiniZinc/minizinc-benchmarks/)

## Benchmarking

We call a _campaign_ a benchmarking session uniquely identified by a solver's identifier (`com.google.or-tools`, `org.choco.choco`), the version of the solver and a short description of the hardware (can be the name of the HPC system, e.g., aion at the University of Luxembourg).
The _set of instances_ is a simple CSV file with three columns.
You can generate automatically this CSV using:

```
mzn-bench collect-instances ../data/mzn-challenge/2023 > mzn2023.csv
```

The main parameters of your campaign are specified in the script `benchmarking/run-mzn.sh` that you must adapt depending on your needs.
In particular, we have the following variables:

* `INSTANCE_FILE` is the name of the CSV generated above.
* `MZN_SOLVER` is the MiniZinc identifier of the solver as given in the MiniZinc configuration files located at `~/.minizinc/solvers/`.
* `VERSION` is the version of the solver used.
* `MZN_TIMEOUT` is the timeout in milliseconds to solve each instance.
* `NUM_JOBS` is the number of jobs to be run in parallel on each node (usually you want one job per (virtual) CPUs, should be the same as the value of the Slurm option `--ntasks-per-node`, if using Slurm).
* `OUTPUT_DIR` is the campaign directory, you can actually choose another campaign layout if you want.
* `HARDWARE` is the name of the hardware on which the experiments are run.
* `SHORT_HARDWARE` is the short name of the hardware, for instance the HPC platform.

The `parallel` command runs the set of experiments in parallel.
It works locally on your computer, but when you are on a HPC, the experiments are automatically run _across nodes_ (for 10 nodes and 8 sockets per nodes, it runs 80 experiments in parallel at each instant).
This command is useful when the Slurm jobs queue is limited in size per user, and you have thousands of experiments to run.
The full command (in `run-mzn.sh`) is:
```
parallel --no-run-if-empty $MULTINODES_OPTION --rpl '{} uq()' --jobs $NUM_JOBS -k --colsep ',' --skip-first-line $MZN_COMMAND {2} {3} '|' python3 $DUMP_PY_PATH $OUTPUT_DIR {1} {2} {3} $MZN_SOLVER :::: $INSTANCE_FILE
```
Here are the options of the command used for our purpose:

* `--no-run-if-empty`: if you have an empty line in the `INSTANCE_FILE`, it doesn't consider this empty line as an experiment to run.
* `$MULTINODES_OPTION`: connect to each node in SSH to run the experiments (when executed on HPC).
* `--rpl '{} uq()'`: in `INSTANCE_FILE` the paths of the files are already quoted, we ask `parallel` to avoid quoting them again.
* `--jobs $NUM_JOBS`: specify how many jobs per node do we run in parallel.
* `-k --colsep ','`: specify the column separator of the CSV file (here a comma).
* `--skip-first-line`: skip the first line of the CSV file (used for each column's name).

It is normally not necessary to change these options.
The rest of the command is constituted of the solver to run and the list of the experiments:

* `:::: $INSTANCE_FILE`: we have one experiment per line, and each column's value is stored in the placeholders `{1}`, `{2}` and `{3}` that can be used in the command preceding `::::`.
* `$MZN_COMMAND {2} {3} '|' python3 $DUMP_PY_PATH $OUTPUT_DIR {1} {2} {3} $MZN_SOLVER`: each experiment consists in running the Minizinc solver (specified in the variable `$MZN_COMMAND`). The solver prints on the standard output JSON text (thanks to the `minizinc` options `--json-stream --output-mode json --output-time --output-objective`). These JSON texts are parsed by the script `dump.py` and then formatted and stored in the campaign directory (`$OUTPUT_DIR`). For each experiment, there are two files created:
  - `turbo-gpu-release_accap_accap_a4_f30_t15_sol.yml`: the solutions found by the solver.
  - `turbo-gpu-release_accap_accap_a4_f30_t15_stats.yml`: the statistics of the solver at the end of the timeout.

The `dump.py` script takes the campaign directory followed by the three columns of the CSV file (problem's name, model path and data path) and the name of the solver.
It is possible to pass more arguments in case these 4 are not sufficient to generate a unique identifier for the experiment.

### HPC Slurm Configuration

The file `run-mzn.sh` can be used locally (the `slurm` multinodes option will simply be ignored).
If you want to use it on the HPC, there are two additional scripts to complete:

* `slurm.sh` which contains all the reservation detail (account name, number of nodes, time, ...).
* `configure.sh` which loads the right modules and set the path correctly.

Normally, `configure.sh` is merged with `slurm.sh`, but when running on multiple nodes through SSH (as `parallel`), we must run this script on each node.
Therefore, the standard approach is to run this script in the `.bashrc` file of your HPC account and change the directory to `benchmarking`.

```
source /project/scratch/p200244/lattice-land/bench/benchmarks/benchmarking/configure.sh
cd /project/scratch/p200244/lattice-land/bench/benchmarks/benchmarking
```

When running on multiple nodes, the SSH connections triggered by `parallel` will automatically set up the configuration, and we will be in the right directory to execute the command.

### More Experiments with Different Solver's Options

Sometimes, we want to benchmark the same model and data, but with different options of the solver.
For instance, I benchmarked Turbo with no special option, the option `-noatomics` and the option `-globalmem`:

```
parallel [...] $MZN_COMMAND {4} {2} {3} '|' python3 $DUMP_PY_PATH $OUTPUT_DIR {1} {2} {3} $MZN_SOLVER {4} :::: $INSTANCE_FILE ::: "-s " "-noatomics " "-globalmem "
```

After `:::: $INSTANCE_FILE` we can specify additional experiments using `::: <list of parameters>`.
If there are 100 lines in your `INSTANCE_FILE` and three parameters, there will be 300 experiments to run.
We notice three things:
* `{4}` is the new option listed after `:::` and it is appended directly to the `$MZN_COMMAND`.
* Since each of these options leads to different experiments, I pass `{4}` to the `dump.py` script too, so it generates a unique name.
* The option `-s ` is actually a "fake option" because we cannot use the empty string in the parameters list of `parallel`. It is interpreted by `minizinc` as a redundant option. In `dump.py`, there is a special case which erases this option from the UID.

The option `--dry-run` of `parallel` becomes very interesting: it lists all the commands that will be launched by `parallel`, without executing them.
You can verify the well-formedness of the commands before starting the experiments.

## Analyzing

Once the experiments are finished, you can verify the results using:

```
./analyzing/mznbench_check.sh campaign/turbo.gpu.profiling-v1.1.3-A100
```

Then, you can generate a summary of the experiments, to be further analyzed using:

```
./analyzing/mznbench_collect.sh campaign/turbo.gpu.release-v1.1.7-A5000
```

It creates two files:
* `campaign/turbo.gpu.release-v1.1.7-A5000.csv`: The statistics and best objective found for each experiment.
* `campaign/turbo.gpu.release-v1.1.7-A5000-objectives.csv`: All the objectives found for each experiment.

The analysis of the benchmarks is then carried out in the Jupyter notebook `analysis.ipynb` which uses `analysis.py` to gather/summarize/display the data.
Once again, these files can be extended or modified for your particular purposes.
