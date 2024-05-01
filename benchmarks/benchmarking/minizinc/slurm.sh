#!/bin/bash -l
#SBATCH --time=03:00:00
#SBATCH --nodes=10
#SBATCH --partition=batch
#SBATCH --ntasks-per-node=8 # On Aion, we have 8 virtual processors per node.
#SBATCH --ntasks-per-socket=1 # On Aion, a socket is a virtual processor.
#SBATCH --cpus-per-task=8 # On Aion, each task has 8 cores available, although they should only use one.
#SBATCH --export=ALL
#SBATCH --output=slurm.out

echo $SLURM_JOB_NODELIST

./run.sh
