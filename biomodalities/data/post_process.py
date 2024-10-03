from typing import Optional

import numpy as np
import pandas as pd
from scipy import linalg, sparse
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch
import os
import anndata
from tqdm import tqdm



def center_embeddings(
    embeddings: np.ndarray, 
    metadata: pd.DataFrame,
    pert_col: str, 
    control_key: str,
    batch_col: Optional[str] = None
) -> np.ndarray:
    """
    Compute centered embeddings for each observation by subtracting the mean
    embedding of control samples within each batch from all embeddings in that batch.

    Args:
        embeddings (np.ndarray): The embeddings to be aligned.
        metadata (pd.DataFrame): The metadata containing information about the embeddings.
        pert_col (str): The column in the metadata containing perturbation information.
        control_key (str): The key for non-targeting controls in the metadata.
        batch_col (str, optional): Column name in the metadata representing the batch labels.

    Returns:
        np.ndarray: The centered embeddings.
    """
    # Copy embeddings to avoid modifying the original data
    centered_embeddings = embeddings.copy()

    # If batch information is provided, perform the operation within each batch
    if batch_col is not None:
        batches = metadata[batch_col].unique()
        for batch in batches:
            batch_indices = metadata[batch_col] == batch
            control_indices = batch_indices & (metadata[pert_col] == control_key)
            
            # Compute mean embedding for non-targeting controls in this batch
            mean_embedding = embeddings[control_indices].mean(axis=0)
            
            # Subtract mean embedding from all embeddings in this batch
            centered_embeddings[batch_indices] -= mean_embedding
    else:
        # Compute the global mean of control embeddings if no batch information is provided
        control_indices = metadata[pert_col] == control_key
        mean_embedding = embeddings[control_indices].mean(axis=0)
        centered_embeddings -= mean_embedding

    return centered_embeddings

def centerscale_on_controls(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    pert_col: str,
    control_key: str,
    batch_col: Optional[str] = None,
) -> np.ndarray:
    """
    Center and scale the embeddings on the control perturbation units in the metadata.
    If batch information is provided, the embeddings are centered and scaled by batch.

    Args:
        embeddings (numpy.ndarray): The embeddings to be aligned.
        metadata (pandas.DataFrame): The metadata containing information about the embeddings.
        pert_col (str, optional): The column in the metadata containing perturbation information.
        control_key (str, optional): The key for non-targeting controls in the metadata.
        batch_col (str, optional): Column name in the metadata representing the batch labels.
            Defaults to None.
    Returns:
        numpy.ndarray: The aligned embeddings.
    """
    embeddings = embeddings.copy()
    if batch_col is not None:
        batches = metadata[batch_col].unique()
        for batch in batches:
            batch_ind = metadata[batch_col] == batch
            batch_control_ind = batch_ind & (metadata[pert_col] == control_key)
            embeddings[batch_ind] = StandardScaler().fit(embeddings[batch_control_ind]).transform(embeddings[batch_ind])
        return embeddings

    control_ind = metadata[pert_col] == control_key
    return StandardScaler().fit(embeddings[control_ind]).transform(embeddings)


def tvn_on_controls(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    pert_col: str,
    control_key: str,
    batch_col: Optional[str] = None,
) -> np.ndarray:
    """
    Apply TVN (Typical Variation Normalization) to the data based on the control perturbation units.
    Note that the data is first centered and scaled based on the control units.

    Args:
        embeddings (np.ndarray): The embeddings to be normalized.
        metadata (pd.DataFrame): The metadata containing information about the samples.
        pert_col (str): The column name in the metadata DataFrame that represents the perturbation labels.
        control_key (str): The control perturbation label.
        batch_col (str, optional): Column name in the metadata DataFrame representing the batch labels
            to be used for CORAL normalization. Defaults to None.

    Returns:
        np.ndarray: The normalized embeddings.
    """
    embeddings = embeddings.copy()
    embeddings = centerscale_on_controls(embeddings, metadata, pert_col, control_key)
    ctrl_ind = metadata[pert_col] == control_key
    embeddings = PCA().fit(embeddings[ctrl_ind]).transform(embeddings)
    embeddings = centerscale_on_controls(embeddings, metadata, pert_col, control_key)
    if batch_col is not None:
        batches = metadata[batch_col].unique()
        for batch in batches:
            batch_ind = metadata[batch_col] == batch
            batch_control_ind = batch_ind & (metadata[pert_col] == control_key)
            source_cov = np.cov(embeddings[batch_control_ind], rowvar=False, ddof=1) + 0.5 * np.eye(embeddings.shape[1])
            source_cov_half_inv = linalg.fractional_matrix_power(source_cov, -0.5)
            embeddings[batch_ind] = np.matmul(embeddings[batch_ind], source_cov_half_inv)
    return embeddings



if __name__ == '__main__' :

    folder_path = './datasets/eval'
    file_name = 'crispr_l1000.h5ad' # File to post-process

    pert_col = "is_control" # Perturbation column in .obs
    batch_col = "dataset_batch_num"
    control_key = True # Control value in perturbation column
    obsm_keys = ["X_scVI_poisson_hvg_12000"] # List of obsm keys to center


    full_data_path = os.path.join(folder_path,file_name)
    # Read anndata object from full_data_path
    adata = anndata.read_h5ad(full_data_path)
    for obsm_key in obsm_keys:
        print(obsm_key)
        
        print("Centering...")
        centered_embeddings = center_embeddings(
                        embeddings=adata.obsm[obsm_key],
                        metadata=adata.obs,
                        pert_col=pert_col,
                        control_key=control_key,
                        batch_col=batch_col)
        adata.obsm['centered_' + obsm_key] = centered_embeddings
        
        if folder == 'test2':
            continue

        print("Center Scaling...")
        center_scaled_embeddings = centerscale_on_controls(
                        embeddings=adata.obsm[obsm_key],
                        metadata=adata.obs,
                        pert_col=pert_col,
                        control_key=control_key,
                        batch_col=batch_col)
        adata.obsm['center_scaled_' + obsm_key] = center_scaled_embeddings
        
        print("Applying TVN...")
        tvn_embeddings = tvn_on_controls(
                        embeddings=adata.obsm[obsm_key],
                        metadata=adata.obs,
                        pert_col=pert_col,
                        control_key=control_key,
                        batch_col=batch_col)
        adata.obsm['tvn_' + obsm_key] = tvn_embeddings
        
    print("writing file to disk...")
    adata.write_h5ad(full_data_path)
    print("Done.")
