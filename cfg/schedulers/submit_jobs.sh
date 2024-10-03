#!/bin/bash

base_seed=42
train_name="baseline_tx"
template_path="./cfg/scripts/templates/replogle_eval_template.slurm"
job_type="linear"
#obsm_key="tvn_X_scVI_hvg_5000"

# Array of different obsm_key versions
obsm_keys=(
    "random"
    "PCA"
    "X_geneformer"
    "X_joung_overexpression_scVI_hvg_15000"
    "X_random_joung_overexpression_scVI_hvg_15000"
    "X_scGPT"
    "X_scGPT_finetuned"
    "X_scVI_0.0005_new_pert_hvg_8000"
    "X_scVI_0.005_new_pert_hvg_8000"
    "X_scVI_05_hvg_300"
    "X_scVI_05_hvg_8000"
    "X_scVI_05_new_pert_hvg_8000"
    "X_scVI_hvg_1000"
    "X_scVI_hvg_300"
    "X_scVI_hvg_3000"
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
    "center_scaled_X_scVI_0.0005_new_pert_hvg_8000"
    "center_scaled_X_scVI_0.005_new_pert_hvg_8000"
    "center_scaled_X_scVI_05_hvg_300"
    "center_scaled_X_scVI_05_hvg_8000"
    "center_scaled_X_scVI_05_new_pert_hvg_8000"
    "center_scaled_X_scVI_hvg_1000"
    "center_scaled_X_scVI_hvg_300"
    "center_scaled_X_scVI_hvg_3000"
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
    "centered_X_scVI_0.0005_new_pert_hvg_8000"
    "centered_X_scVI_0.005_new_pert_hvg_8000"
    "centered_X_scVI_05_hvg_300"
    "centered_X_scVI_05_hvg_8000"
    "centered_X_scVI_05_new_pert_hvg_8000"
    "centered_X_scVI_hvg_1000"
    "centered_X_scVI_hvg_300"
    "centered_X_scVI_hvg_3000"
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
    "tvn_X_scVI_0.0005_new_pert_hvg_8000"
    "tvn_X_scVI_0.005_new_pert_hvg_8000"
    "tvn_X_scVI_05_hvg_300"
    "tvn_X_scVI_05_hvg_8000"
    "tvn_X_scVI_05_new_pert_hvg_8000"
    "tvn_X_scVI_hvg_1000"
    "tvn_X_scVI_hvg_300"
    "tvn_X_scVI_hvg_3000"
    "tvn_X_scVI_hvg_8000"
    "tvn_X_uce_33_layers"
    "tvn_X_uce_4_layers"
)

for obsm_key in "${obsm_keys[@]}"; do
    # Loop over k
    for i in $(seq 0 4); do
        seed=$((base_seed + i))
        job_name="${train_name}_eval_${job_type}_seed_${seed}_${obsm_key}_"
        
        # Read template and replace placeholders
        script_content=$(cat "$template_path" | \
            sed "s/{{JOB_NAME}}/$job_name/g" | \
            sed "s/{{JOB_TYPE}}/$job_type/g" | \
            sed "s/{{OBSM_KEY}}/$obsm_key/g" | \
            sed "s/{{SEED}}/$seed/g")

        # Create a unique SLURM script for this particular job
        script_name="./cfg/scripts/tmp/${job_name}.slurm"
        echo "$script_content" > "$script_name"

        # Submit the job
        sbatch "$script_name"
    done
done
