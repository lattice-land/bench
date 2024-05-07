# Installing Minizinc

```
mkdir deps
cd deps
git clone git@github.com:MiniZinc/libminizinc.git
cd libminizinc
git checkout 2.8.3 # or a more recent version
```

We then use CMake and make to compile the executable Minizinc.
You first need to connect to a node and load CMake:

```
salloc -p interactive --qos debug -C batch # or just "si" on uni.lu machines
module load devel/CMake/3.20.1-GCCcore-10.2.0 # load CMake, or use module spider CMake to find a version of CMake available.
```

To compile:

```
cmake -Bbuild
cmake --build build
```

Normally, you will have the executable `build/minizinc`.
You can add it to your path in `.bashrc`:

```
export PATH=$PATH:$HOME/deps/libminizinc/build
```

Further, before using `minizinc`, you will have to load the module GCC, don't forget it in your SLURM scripts (without it you will have an error like "minizinc: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.26' not found (required by minizinc)"):

```
module load compiler/GCCcore/10.2.0
```

You can then install solvers that are compatible with Minizinc and use them through the `minizinc` command line:

* [Turbo]()
* [Choco]()
* [Ortools]()
* [Picat]()
