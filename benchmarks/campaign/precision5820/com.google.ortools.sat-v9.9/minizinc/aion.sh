#!/bin/bash

# See installation instructions for Choco.
module use $HOME/.local/easybuild/${ULHPC_CLUSTER}/turbo/${RESIF_ARCH}/modules/all

module load compiler/GCCcore/10.2.0
module load lang/Python/3.8.6-GCCcore-10.2.0
module load lang/Java/21.0.2 # for Choco