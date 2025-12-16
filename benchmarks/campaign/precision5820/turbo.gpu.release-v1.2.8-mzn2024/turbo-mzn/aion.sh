#!/bin/bash -l

module use "/work/projects/software_set/easybuild/${ULHPC_CLUSTER}/2023b/${RESIF_ARCH}/modules/all" # Use the latest software chain of AION.
module load devel/CMake
module load devel/Doxygen
module load lib/libxml2
module load lang/Python
