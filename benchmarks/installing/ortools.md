# Or-Tools Solver

## Installing

```
cd deps
git clone git@github.com:google/or-tools.git
cd or-tools
git checkout v9.9 # or latest release
```

We compile Or-Tools as follows:
```
salloc -p interactive --qos debug -C batch # or just "si" on uni.lu machines
module load devel/CMake/3.20.1-GCCcore-10.2.0 # load CMake, or use module spider CMake to find a version of CMake available.
cmake -S . -B build -DBUILD_DEPS=ON
cmake --build build --config Release --target all -j 2 -v
```

We next copy the configuration file:
```
cp ortools/flatzinc/cpsat.msc.in ~/.minizinc/solvers/ortools.sat.msc
```
Then edit the copied file with (where you replace `PATH_HOME` with the output of `echo $HOME`):
```
  "version": "9.9",
  "mznlib": "PATH_HOME/deps/or-tools/ortools/flatzinc/mznlib",
  "executable": "PATH_HOME/deps/or-tools/build/bin/fzn-cp-sat",
```

## Running

You can check whether it works by running:
```
module load compiler/GCCcore/10.2.0 # for minizinc
cd ~/lattice-land/bench/benchmarks/data/mzn-challenge/2023/mrcpsp/
minizinc --solver com.google.ortools.sat -t 60000 mrcpsp.mzn j30_25_5.dzn
```
