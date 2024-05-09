# Benchmarking Constraint Solvers

In this document, we discuss our workflow to benchmark constraint solvers.

## Prerequisites

```
git clone --recursive git@github.com:lattice-land/bench.git
cd bench
salloc -p interactive --qos debug -C batch    # skip if not on HPC
module load lang/Python/3.8.6-GCCcore-10.2.0  # skip if not on HPC
python -m venv pybench
source pybench/bin/activate
pip install git+https://github.com/MiniZinc/mzn-bench.git#egg=mzn-bench[scripts]
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
    - minizinc/
      - aion.sh
      - precision5820.sh
      - dump.py
      - run.sh
    - ace/
      - aion.sh
      - precision5820.sh
  - campaign/
  - data/
    - [mzn-challenge/](https://github.com/MiniZinc/mzn-challenge/)
    - [minizinc-benchmarks/](https://github.com/MiniZinc/minizinc-benchmarks/)
    - XCSP/  (to be generated)
    - xcsp.sh

You can download XCSP instances by executing the script `./xcsp.sh XCSP23` and `./xcsp.sh XCSP22`.

## Benchmarking

We call a _campaign_ a benchmarking session uniquely identified by a solver's identifier (`com.google.or-tools`, `org.choco.choco`), the version of the solver and a short description of the hardware (can be the name of the HPC system, e.g., aion at the University of Luxembourg).

### Instances

* For Minizinc instances, you can run:
```
mzn-bench collect-instances ../data/mzn-challenge/2023 > mzn2023.csv
```
* For XCSP instances, you can run:
```
./collect.sh ../data/XCSP/XCSP22/MiniCOP > xcsp22_minicop.csv
```

### Starting a campaign

Depending on the solver you want to benchmark you must select a _workflow_, we have two:

* minizinc/: for installed Minizinc-based solvers.
* ace/: for the ACE solver.

The first thing to do is to copy locally a directory depending on the campaign you are preparing:
```
cp minizinc mzn2023
```
Inside `mzn2023` you can modify the parameters in the script `run.sh` (changing the solvers, timeout, cores, etc.).
It is also in `run.sh` that you set up the number of nodes you want to allocate to run the experiments.
By default, we run 1 experiment per node to avoid interferences.
Once the parameters set, you can start the experiment using:
```
./run aion.sh
```
where the script name is the name of the machine and contain any needed initialization (e.g., loading modules or a Python virtual environment).
You can create yours for your own machine.

The results of the experiments are in the `campaign/aion/org.choco.choco-v4.10.14` directory.
In that directory, you also have the folder of your workflow copied and inside the `jobs.log` which contains all commands that have been executed.
If you call `./run.sh aion.sh` again, only the commands that have not been executed will be executed, this can be useful if you set up a SLURM walltime that was too short.
If you call `./run.sh aion.sh ../../campaign/aion/org.choco.choco-v4.10.14/mzn2023/jobs.log`, the script will re-execute all commands that might have failed previously.

### Note on the `parallel` command

The `parallel` command runs the set of experiments in parallel.
It works locally on your computer, but when you are on a HPC, the experiments are automatically run _across nodes_ (for 10 nodes, it runs 10 experiments in parallel at each instant).
This command is useful when the Slurm jobs queue is limited in size per user, and you have thousands of experiments to run.
The full command (in `minizinc/run.sh`) is:
```
parallel --no-run-if-empty --rpl '{} uq()' -k --colsep ',' --skip-first-line -j $NUM_PARALLEL_EXPERIMENTS --resume --joblog $COMMANDS_LOG $SRUN_COMMAND $MZN_COMMAND $BENCHMARKING_DIR_PATH/{2} $BENCHMARKING_DIR_PATH/{3} '2>&1' '|' python3 $DUMP_PY_PATH $OUTPUT_DIR {1} {2} {3} $MZN_SOLVER $CORES $THREADS :::: $INSTANCES_PATH
```
Here are the options of the command used for our purpose:

* `--no-run-if-empty`: if you have an empty line in the `INSTANCE_FILE`, it doesn't consider this empty line as an experiment to run.
* `--rpl '{} uq()'`: in `INSTANCE_FILE` the paths of the files are already quoted, we ask `parallel` to avoid quoting them again.
* `-j $NUM_PARALLEL_EXPERIMENTS`: specify how many jobs we run in parallel (across all nodes).
* `-k --colsep ','`: specify the column separator of the CSV file (here a comma).
* `--skip-first-line`: skip the first line of the CSV file (used for each column's name).

It is normally not necessary to change these options.
The rest of the command is constituted of the solver to run and the list of the experiments:

* `:::: $INSTANCE_FILE`: we have one experiment per line, and each column's value is stored in the placeholders `{1}`, `{2}` and `{3}` that can be used in the command preceding `::::`.
* `$MZN_COMMAND $BENCHMARKING_DIR_PATH/{2} $BENCHMARKING_DIR_PATH/{3} '2>&1' '|' python3 $DUMP_PY_PATH $OUTPUT_DIR {1} {2} {3} $MZN_SOLVER $CORES $THREADS`: each experiment consists in running the Minizinc solver (specified in the variable `$MZN_COMMAND`). The solver prints on the standard output JSON text (thanks to the `minizinc` options `--json-stream --output-mode json --output-time --output-objective`). These JSON texts are parsed by the script `dump.py` and then formatted and stored in the campaign directory (`$OUTPUT_DIR`).
The `dump.py` script takes the campaign directory followed by the three columns of the CSV file (problem's name, model path and data path), the name of the solver, and the number of cores and thread used.
It is possible to pass more arguments in case these are not sufficient to generate a unique identifier for the experiment.

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

## Postprocessing

Once the experiments are finished, you can create verify the answers given by the solver and create two CSV summary files (in `benchmarking/minizinc`):

```
./postprocess.sh ../../campaign/aion/org.choco.choco-v4.10.14/
```

It creates two files:
* `campaign/aion/org.choco.choco-v4.10.14.csv`: The statistics and best objective found for each experiment.
* `campaign/aion/org.choco.choco-v4.10.14-objectives.csv`: All the objectives found for each experiment.

The analysis of the benchmarks is then carried out in the Jupyter notebook `analysis.ipynb` which uses `analysis.py` to gather/summarize/display the data.
These files can be extended or modified for your particular purposes.
