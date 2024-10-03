#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --output=./terminal/{{JOB_NAME}}_%A_%a.out
#SBATCH --error=./error/{{JOB_NAME}}_%A_%a.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=800G
#SBATCH --time=72:00:00
#SBATCH --gpus-per-node=1
#SBATCH --array=1-{{NUM_JOBS}}%30  # Define the array with a maximum of 30 jobs running concurrently

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
CONFIG_PATH="./cfg/config/replogle_eval_reconstruct.yaml"

# Calculate the seed and obsm_key based on SLURM_ARRAY_TASK_ID
seed=$(({{BASE_SEED}} + (SLURM_ARRAY_TASK_ID - 1) / {{NUM_OBSM_KEYS}}))
obsm_key_index=$(( (SLURM_ARRAY_TASK_ID - 1) % {{NUM_OBSM_KEYS}} ))

# Get the actual obsm_key from the list
obsm_keys=(
    "random_scramble_PCA"
    "PCA"
    "X_geneformer"
    "X_joung_overexpression_scVI_hvg_15000"
    "X_random_joung_overexpression_scVI_hvg_15000"
    "X_scGPT"
    "X_scGPT_finetuned"
    "X_scVI_hvg_8000"
    "X_uce_33_layers"
    "X_uce_4_layers"
    "cell_PLM"
    "center_scaled_PCA"
    "center_scaled_X_geneformer"
    "center_scaled_X_joung_overexpression_scVI_hvg_15000"
    "center_scaled_X_random_joung_overexpression_scVI_hvg_15000"
    "center_scaled_X_scGPT"
    "center_scaled_X_scGPT_finetuned"
    "center_scaled_X_scVI_hvg_8000"
    "center_scaled_X_uce_33_layers"
    "center_scaled_X_uce_4_layers"
    "center_scaled_cell_PLM"
    "centered_PCA"
    "centered_X_geneformer"
    "centered_X_joung_overexpression_scVI_hvg_15000"
    "centered_X_random_joung_overexpression_scVI_hvg_15000"
    "centered_X_scGPT"
    "centered_X_scGPT_finetuned"
    "centered_X_scVI_hvg_8000"
    "centered_X_uce_33_layers"
    "centered_X_uce_4_layers"
    "centered_cell_PLM"
    "tvn_PCA"
    "tvn_X_geneformer"
    "tvn_X_joung_overexpression_scVI_hvg_15000"
    "tvn_X_random_joung_overexpression_scVI_hvg_15000"
    "tvn_X_scGPT"
    "tvn_X_scGPT_finetuned"
    "tvn_X_scVI_hvg_8000"
    "tvn_X_uce_33_layers"
    "tvn_X_uce_4_layers"
    "tvn_cell_PLM"
)

obsm_key=${obsm_keys[$obsm_key_index]}

# Execute the Python script with the specified configuration and any necessary overrides
python main_eval.py --config $CONFIG_PATH --seed $seed --run_name {{JOB_NAME}} --eval_method {{JOB_TYPE}} --obsm_key $obsm_key --distance_fx euclidean --use_pca 
