### Installation

This code is currently tested with Python 3.9 and 3.10. No other Python version have been tested yet.

#### Pytorch Installation

Current code is run with Pytorch 2.0.1 for CUDA 11.7. Other versions not tested yet.

Install pytorch using : 

```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
# Or 
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

Except for pytorch, please install all dependencies using poetry.

Use `poetry install` on the root of repository.


### Quickstart

### Download data

All datasets are available through the following links : [Part 1](https://zenodo.org/records/14037432), [Part 2](https://zenodo.org/records/14037465), [Part 3](https://zenodo.org/records/14039051)

`replogle_2022` dataset has been split into 3 parts, it would need to be merged first through the following python code :

```python
import anndata as ad

# Read the three split AnnData files
adata_part1 = ad.read_h5ad('replogle_2022_part1.h5ad')
adata_part2 = ad.read_h5ad('replogle_2022_part2.h5ad')
adata_part3 = ad.read_h5ad('replogle_2022_part3.h5ad')

# Concatenate the AnnData objects in the original order
adata_full = ad.concat([adata_part1, adata_part2, adata_part3], axis=0, join='outer')

# Save the concatenated AnnData object
adata_full.write_h5ad('replogle_2022.h5ad')
```


#### Prepare data

Create a `datasets` folder in the root of repository. 

Run :

```
mkdir datasets/eval
mkdir datasets/train
mkdir datasets/test1
mkdir datasets/test2
```

Put full data anndata h5ad file in `datasets/eval`. You can use replogle_2022 or l1000_crispr, or use your own perturbation data for evaluation.

#### Prepare embeddings to evaluate

Extract the embeddings of your adata in  `datasets/eval`  using the models you want to evaluate.

Save the embeddings in `adata.obsm['your_model_key']` of the `datasets/eval` adata. Shape and order should be the same as original data in `adata.X` and `adata.obs`.


#### Optional : Post-process embeddings to be evaluated 

Post process the embeddings saved in the anndata file, if you want to explore post processing effect. 

Three post processing approaches are handled : Centering, Center Scaling, and TVN.

In `./biomodalities/data/post_process.py`, modify the following values to fit your data : 

```python
folder_path = './datasets/eval'
file_name = 'your_anndata_file.h5ad' # File to post-process

pert_col = "is_control" # Perturbation column in .obs
batch_col = "dataset_batch_num" # Batch column in .obs
control_key = True # Control value in perturbation column
obsm_keys = ["your_model_key"] # List of embedding obsm keys to post-process
```


Run post processing with the following command from root :

```
python -m biomodalities.data.post_process
```

#### Split data into train and test

We split data into train and test. We will have two test sets : `test1` is for linear probing and knn, it shares the same perturbations as training, but has distinct batches. `test2` is for reconstruction, it has distinct perturbations and distinct batches from training.

In `./biomodalities/data/data_split.py`, modify the following to fit your data :

```python
file_path = "./datasets/eval/crispr_l1000.h5ad" # your file path
batch = "gem_group" # Batch column
perturbation = "gene_id" # Perturbation column
control_column = "is_control" # Control column, can be the same as perturbation column
control_key = True # Control samples key
```

You can split your adata with the following command from root :

```
python -m biomodalities.data.data_split
```


#### Optional : Create config file

If you are using an evaluation dataset separate from `replogle_2022` and `l1000_cripr` of the article, you will have to create a new configuration file for the evaluation dataset. Please refer to `./cfg/config/replogle_eval.yaml` and `./cfg/config/replogle_eval_reconstruct.yaml` for a template for creating your config file.

#### Wandb 

You need to have wandb setup properly on your terminal and machine. Specify in your config yaml file the wandb project name and entity.


#### Running code


Run from root : 

```
python main_eval.py --config ./cfg/config/replogle_eval.yaml --seed {{SEED}} --run_name {{JOB_NAME}} --eval_method {{JOB_TYPE}} --obsm_key {{OBSM_KEY}}
```

Replace `{{JOB_TYPE}}` with the evaluation of choice (`bmdb`, `bmdb_precision`, `reconstruct`, `linear`, `knn`, `ilisi`)

If you use SLURM for job management, run from root :

```
bash cfg/schedulers/submit_jobs.sh
# OR 
bash cfg/schedulers/submit_arrays.sh
```
