#!/bin/bash

module use $HOME/.local/easybuild/${ULHPC_CLUSTER}/turbo/${RESIF_ARCH}/modules/all
module load system/CUDA/12.4.0
module load devel/CMake/3.27.6-GCCcore-13.2.0
module load devel/Doxygen/1.9.8-GCCcore-13.2.0
module load lib/libxml2/2.11.5-GCCcore-13.2.0
module load lang/Python/3.11.5-GCCcore-13.2.0
