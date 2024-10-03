#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --output=./terminal/{{JOB_NAME}}_%j.out
#SBATCH --error=./error/{{JOB_NAME}}_%j.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=50:00:00
#SBATCH --gpus-per-node=1
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

# Path to the YAML configuration file
CONFIG_PATH="./cfg/config/replogle_eval.yaml"

# Execute the Python script with the specified configuration and any necessary overrides
python main_eval.py --config $CONFIG_PATH --seed {{SEED}} --run_name {{JOB_NAME}} --eval_method {{JOB_TYPE}} --obsm_key {{OBSM_KEY}} 
