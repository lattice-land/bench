#!/bin/bash

module load lang/Python/3.8.6-GCCcore-10.2.0
module load lang/Java/16.0.1
export PATH=$PATH:$HOME/deps/libminizinc/build
source $HOME/lattice-land/bench/pybench/bin/activate
