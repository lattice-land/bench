# Benchmarking Constraint Solvers

In this document, we discuss our workflow to benchmark constraint solvers.

## Prerequisites

```
git clone --recursive git@github.com:lattice-land/bench.git
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
    - run.sh
    - slurm.sh
  - campaign/
  - [mzn-challenge/](https://github.com/MiniZinc/mzn-challenge/)
  - [minizinc-benchmarks/](https://github.com/MiniZinc/minizinc-benchmarks/)

## Getting Starting

We call a _campaign_ a benchmarking session uniquely identified by a solver's MiniZinc identifier (`com.google.or-tools`, `org.choco.choco`), the version of the solver and a short description of the hardware (for Turbo, I use the name of the GPU, e.g. `A100`).
This is defined in the script `benchmarking/run.sh`; you can adapt the identifier of your campaign depending on your needs.

The _set of instances_ is a simple CSV file with three columns, for instance [mzn2021-23.csv](https://github.com/ptal/turbo/blob/0f50d1d82b4a60eca51b965197192b6e5dc4d61d/benchmarks/benchmarking/mzn2021-23.csv) which contains all instances solvable by Turbo from the MiniZinc competition 2021, 2022 and 2023.
You can generate automatically this CSV using:

```
mzn-bench collect-instances ../mzn-challenge/2023 > mzn2023.csv
```

This CSV is directly written in the variable `INSTANCE_FILE` of `run.sh`, this is where you can also modify the `MZN_TIMEOUT`.
