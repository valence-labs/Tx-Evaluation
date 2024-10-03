#!/bin/bash

base_seed=42
train_name="baseline_tx"
template_path="./cfg/scripts/templates/l1000_array_eval_template.slurm"
job_type="reconstruct"

# Number of seeds and obsm keys
num_seeds=5
num_obsm_keys=60  # Number of different obsm_key versions
num_jobs=$((num_seeds * num_obsm_keys))

# Read template and replace placeholders
script_content=$(cat "$template_path" | \
    sed "s/{{JOB_NAME}}/${train_name}_eval_${job_type}/g" | \
    sed "s/{{JOB_TYPE}}/$job_type/g" | \
    sed "s/{{BASE_SEED}}/$base_seed/g" | \
    sed "s/{{NUM_JOBS}}/$num_jobs/g" | \
    sed "s/{{NUM_OBSM_KEYS}}/$num_obsm_keys/g")

# Create a unique SLURM script for the array job
script_name="./cfg/scripts/tmp/${train_name}_eval_${job_type}.slurm"
echo "$script_content" > "$script_name"

# Submit the job array
sbatch "$script_name"
