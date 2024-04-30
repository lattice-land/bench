#!/bin/bash -l
#SBATCH --time=03:00:00
#SBATCH --nodes=10
#SBATCH --partition=batch
#SBATCH --account=p200244
#SBATCH --qos=default
#SBATCH --ntasks-per-node=8 # On Aion, we have 8 virtual processors per node.
#SBATCH --ntasks-per-socket=1 # On Aion, a socket is a virtual processor.
#SBATCH --cpus-per-task=8 # On Aion, each task has 8 cores available, although they should only use one.
#SBATCH --cpu-bind=sockets # Request each task to be run on an independent socket (to try to avoid interferences).
#SBATCH --export=ALL
#SBATCH --output=slurm.out

echo $SLURM_JOB_NODELIST

./run.sh
