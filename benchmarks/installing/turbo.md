# Turbo Solver

## Installing

If you intend to extend Turbo, please see the [developers documentation](https://github.com/lattice-land/.github) instead.

```
cd deps
git clone git@github.com:ptal/turbo.git
cd turbo
git checkout v1.1.7 # or any version more recent
```

To compile Turbo, you need four modules: CUDA-12.4.0, Doxygen-1.9.8, libxml2-2.11.5, CMake-3.27.6.
On the HPC of uni.lu, the version of CMake and Doxygen provided by default are not recent enough, we need to compile them ourselves.
```
salloc -p interactive --qos debug -C batch --mem=0 # note --mem=0 to have more memory for the compilation and avoid ptxas to crash.
module load lang/Python
python -m venv eb-venv
source eb-venv/bin/activate
python3 -m pip install easybuild # install latest version of easybuild.
EASYBUILD_PREFIX=$HOME/.local/easybuild/${ULHPC_CLUSTER}/turbo/${RESIF_ARCH} # set up an easybuild repository for Turbo's build dependencies.
eb --accept-eula-for=CUDA -r CUDA-12.4.0.eb
eb -r Doxygen-1.9.8-GCCcore-13.2.0.eb
eb -r libxml2-2.11.5-GCCcore-13.2.0.eb
eb -r CMake-3.27.6-GCCcore-13.2.0.eb
eb -r Python-3.11.5-GCCcore-13.2.0.eb
```

Once it is installed, we can load these modules:

```
module use $HOME/.local/easybuild/${ULHPC_CLUSTER}/turbo/${RESIF_ARCH}/modules/all
module load system/CUDA/12.4.0
module load devel/CMake/3.27.6-GCCcore-13.2.0
module load devel/Doxygen/1.9.8-GCCcore-13.2.0
module load lib/libxml2/2.11.5-GCCcore-13.2.0
module load lang/Python/3.11.5-GCCcore-13.2.0
```

We compile Turbo as follows (note that `-DCMAKE_CUDA_ARCHITECTURES=75` depends on the GPU of the HPC, if you are compiling on a node with a GPU, you can remove this parameter):
```
cmake -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release -DGPU=ON -DREDUCE_PTX_SIZE=ON -DCMAKE_VERBOSE_MAKEFILE=ON -Bbuild/gpu-release
cmake --build build/gpu-release
```

The CPU version should be compilable with:
```
cmake --workflow --preset cpu-release --fresh # CPU version
```

We next copy the configuration file:
```
cp /benchmarks/minizinc/turbo.gpu.release.msc ~/.minizinc/solvers/turbo.gpu.release.msc
cp /benchmarks/minizinc/turbo.cpu.release.msc ~/.minizinc/solvers/turbo.cpu.release.msc
```
Then edit `~/.minizinc/solvers/turbo.gpu.release.msc` the copied files with (where you replace `PATH_HOME` with the output of `echo $HOME`):
```
  "mznlib": "PATH_HOME/deps/turbo/benchmarks/minizinc/mzn-lib",
  "executable": "PATH_HOME/deps/turbo/build/gpu-release/turbo",
```

and `~/.minizinc/solvers/turbo.cpu.release.msc` with:
```
  "mznlib": "PATH_HOME/deps/turbo/benchmarks/minizinc/mzn-lib",
  "executable": "PATH_HOME/deps/turbo/build/cpu-release/turbo",
```
