import argparse
import warnings
warnings.filterwarnings("ignore")
import anndata
import scvi
import scanpy as sc
import torch
import logging

import pandas as pd

def expand_categories(adata):
    # Get the current categories
    current_categories = adata.var_names.categories

    # Generate new potential categories by appending '-1', '-2', etc.
    # This assumes duplicates won't go beyond '-9'
    new_categories = [f"{name}-{i}" for name in current_categories for i in range(1, 10)]

    # Combine old and new categories
    all_categories = pd.CategoricalIndex(current_categories.tolist() + new_categories)

    # Set the new categories back to var_names
    adata.var_names = pd.Categorical(adata.var_names, categories=all_categories)

    # Now, make the names unique
    adata.var_names_make_unique()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_process_adata(file_path, args, highly_variable_genes=True):
    adata = anndata.read_h5ad(file_path)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if highly_variable_genes:
        sc.pp.highly_variable_genes(adata, n_top_genes=args.nb_hvg, subset=True)
    return adata

def setup_and_train_model(adata, args):
    scvi.model.SCVI.setup_anndata(adata, batch_key=args.batch_key)
    model = scvi.model.SCVI(adata, n_hidden=args.n_hidden, n_latent=int(args.n_latent), n_layers=int(args.n_layers), dropout_rate=float(args.dropout_rate), dispersion=args.dispersion, gene_likelihood=args.gene_likelihood, latent_distribution=args.latent_distribution)
    model.train()
    return model

def main(args):
    # Unpack arguments
    train_path = args.train_path
    test_path1 = args.test_path1
    test_path2 = args.test_path2
    n_latent = args.n_latent
    n_layers = args.n_layers
    dropout_rate = args.dropout_rate

    logging.info("Starting the training process")

    # Load and process the training and testing data
    adata_train = load_and_process_adata(train_path, args=args, highly_variable_genes=True)
    adata_test1 = load_and_process_adata(test_path1, args=args, highly_variable_genes=False)
    adata_test2 = load_and_process_adata(test_path2, args=args, highly_variable_genes=False)

    # Train model
    model = setup_and_train_model(adata_train, args)

    latent_representation_train = model.get_latent_representation(adata_train)

    # Save trained model and load for test data
    model_filename = "./scvi_model.pt"
    model.save(model_filename, overwrite=True)

    expand_categories(adata_test1)


    genes_to_use = adata_train.var_names.intersection(adata_test1.var_names)
    adata_test1 = adata_test1[:, genes_to_use].copy()
    scvi.model.SCVI.prepare_query_anndata(adata_test1, model_filename)
    model = scvi.model.SCVI.load_query_data(adata_test1, model_filename)
    model.is_trained = True

    # Get latent representations for first test set
    latent_representation_test1 = model.get_latent_representation(adata_test1)

    expand_categories(adata_test2)

    genes_to_use = adata_train.var_names.intersection(adata_test2.var_names)
    adata_test2 = adata_test2[:, genes_to_use].copy()

    # Prepare and load model for second test set
    scvi.model.SCVI.prepare_query_anndata(adata_test2, model_filename)
    model = scvi.model.SCVI.load_query_data(adata_test2, model_filename)
    model.is_trained = True

    # Get latent representations for second test set
    latent_representation_test2 = model.get_latent_representation(adata_test2)

    # load again all datasets
    adata_train = anndata.read_h5ad(train_path)
    adata_test1 = anndata.read_h5ad(test_path1)
    adata_test2 = anndata.read_h5ad(test_path2)

    # Store latent representations
    adata_train.obsm['X_scVI_hvg_'+ str(args.nb_hvg)] = latent_representation_train
    adata_test1.obsm['X_scVI_hvg_'+ str(args.nb_hvg)] = latent_representation_test1
    adata_test2.obsm['X_scVI_hvg_'+ str(args.nb_hvg)] = latent_representation_test2

    # Save to disk
    adata_train.write(train_path)
    adata_test1.write(test_path1)
    adata_test2.write(test_path2)

    # Final cleanup
    logging.info("Training completed successfully")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train scVI model with specified hyperparameters.')
    parser.add_argument("--train_path", type=str, required=True, help="File path to training Anndata.")
    parser.add_argument("--test_path1", type=str, required=True, help="File path to first testing Anndata.")
    parser.add_argument("--test_path2", type=str, required=True, help="File path to second testing Anndata.")
    parser.add_argument("--batch_key", type=str, required=True, help="Key of batch in anndata")
    parser.add_argument('--n_latent', type=int, default=256)
    parser.add_argument('--n_hidden', type=int, default=512)
    parser.add_argument('--n_layers', type=int,  default=1)
    parser.add_argument('--dropout_rate', type=float,  default=0.0)
    parser.add_argument('--nb_hvg', type=int,  default=8000)
    parser.add_argument('--dispersion', type=str, choices=['gene', 'gene-batch', 'gene-label', 'gene-cell'], default='gene')
    parser.add_argument('--gene_likelihood', type=str, choices=['zinb', 'nb', 'poisson'], default='zinb')
    parser.add_argument('--latent_distribution', type=str, choices=['normal', 'ln'], default='normal')

    args = parser.parse_args()

    main(args)
