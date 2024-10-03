#!/bin/bash

#SBATCH --job-name=another_correct_splitting
#SBATCH --output=terminal/another_correct_splitting%j.out
#SBATCH --error=error/another_correct_splitting%j.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --mem=1500G
#SBATCH --time=25:00:00

## General
export OMP_NUM_THREADS=8


## Jobs
export NODE_RANK=$SLURM_NODEID
export N_NODES=$SLURM_NNODES
export GPUS_PER_NODE=${SLURM_GPUS_PER_NODE#*:}
export NUM_PROCS=$((N_NODES * GPUS_PER_NODE))
export WORLD_SIZE=$NUM_PROCS

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((((RANDOM<<15)|RANDOM)%63001+2001))

pyenv shell biomodal

# Execute the Python script
python -m biomodalities.data.data_split