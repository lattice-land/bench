# Choco Solver

## Installing

```
cd deps
git clone https://github.com/chocoteam/choco-solver
cd choco-solver
git checkout v4.10.15 # or any version more recent
wget https://github.com/chocoteam/choco-solver/releases/download/v4.10.15/choco-solver-4.10.15-light.jar
ln -s choco-solver-4.10.15-light.jar choco.jar
cp parsers/src/main/minizinc/choco.msc ~/.minizinc/solvers/
```

Then edit
```
vim parsers/src/main/minizinc/fzn-choco
```
with
```
CHOCO_JAR=~/deps/choco-solver/choco.jar
```

Then edit
```
vim ~/.minizinc/solvers/choco.msc
```
with (where you replace `PATH_HOME` with the output of `echo $HOME`):
```
  "mznlib": "PATH_HOME/deps/choco-solver/parsers/src/main/minizinc/mzn_lib/",
  "executable": "PATH_HOME/deps/choco-solver/parsers/src/main/minizinc/fzn-choco",
```

We are going to install the latest version of Java since it can change a lot the efficiency.
For that, we must use the Python version of Easybuild:

```
cd ~
module load lang/Python
python -m venv eb-venv
source eb-venv/bin/activate
python3 -m pip install easybuild # install latest version of easybuild.
EASYBUILD_PREFIX=$HOME/.local/easybuild/${ULHPC_CLUSTER}/turbo/${RESIF_ARCH}
eb -r Java-21.0.2.eb
```

## Running

You can then test with:
```
module use $HOME/.local/easybuild/${ULHPC_CLUSTER}/turbo/${RESIF_ARCH}/modules/all
module load compiler/GCCcore/10.2.0 # for minizinc
module load lang/Java/21.0.2 # for Choco
cd ~/lattice-land/bench/benchmarks/data/mzn-challenge/2023/mrcpsp/ # in this repository
minizinc --solver choco -s -t 60000 mrcpsp.mzn j30_25_5.dzn
```

If you have an error of the form "Error: type error: Type array[int,int] of var set of int is not allowed in as a FlatZinc builtin argument, arrays must be one dimensional", then you need to edit

```
vim parsers/src/main/minizinc/mzn_lib/redefinitions-2.5.2.mzn
```
and append to this file the following definitions:
```
predicate array_var_float_element2d_nonshifted(var int: idx1, var int: idx2, array[int,int] of var float: x, var float: c) =
  let {
    int: dim = card(index_set_2of2(x));
    int: min_flat = min(index_set_1of2(x))*dim+min(index_set_2of2(x))-1;
  } in array_var_float_element_nonshifted((idx1*dim+idx2-min_flat)::domain, array1d(x), c);

predicate array_var_set_element2d_nonshifted(var int: idx1, var int: idx2, array[int,int] of var set of int: x, var set of int: c) =
  let {
    int: dim = card(index_set_2of2(x));
    int: min_flat = min(index_set_1of2(x))*dim+min(index_set_2of2(x))-1;
  } in array_var_set_element_nonshifted((idx1*dim+idx2-min_flat)::domain, array1d(x), c);
```


