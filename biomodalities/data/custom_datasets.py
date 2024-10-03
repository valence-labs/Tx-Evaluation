import numpy as np
import torch
from torch.utils.data import Dataset
import weakref
from multiprocessing.pool import ThreadPool
from queue import Queue,Empty
import h5py
import threading
import time
import anndata 
import pandas as pd
import scanpy as sc


# TODO : Add handling of obsm data like dataloader
# TODO : Inherit from dataset of pytorch
class AnnDataset:
    """
    A custom dataset class that wraps around an .h5ad file to provide an interface
    compatible with data-loading in  PyTorch.
    
    Attributes:
        adata (anndata.AnnData): An AnnData object loaded in backed mode, which allows
                                 for accessing large datasets without loading them entirely into memory.
        n_obs (int): The number of observations (samples) in the dataset.
        var (pandas.DataFrame): A shared copy of the variable (features) metadata from the .h5ad file.
    """
    
    def __init__(self, filename, mode='r', chunk_size=None, control_key=None, control_label=None, hvg=False):
        """
        Initializes the AnnDataset by loading data from an .h5ad file in backed mode.
        
        Parameters:
            filename (str): The file path to the .h5ad file.
            mode (str): The file mode for the .h5ad file ('r' for read-only; 'r+' for read and write).
                        Default is 'r'.
            chunk_size (int, optional): The chunk size for reading the data. This parameter
                                        helps optimize read performance by setting the amount
                                        of data that is read into memory at once. Defaults to None,
                                        which means no chunking.
        """
        
        if hvg :
            self.adata = anndata.read_h5ad(filename)
            self.adata.X.data = np.nan_to_num(self.adata.X.data, nan=0.0)
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            sc.pp.log1p(self.adata)
        else :
            self.adata = anndata.read_h5ad(filename, backed=mode, chunk_size=chunk_size)


        if control_key is not None:
            # filter out the control data
            self.adata = self.adata[self.adata.obs[control_key] != control_label]
        self.n_obs = self.adata.n_obs
        self.var = self.adata.var.copy()

    def __len__(self):
        """
        Returns the total number of observations in the dataset.
        
        Returns:
            int: The number of observations in the dataset.
        """
        return self.n_obs

    def __getitem__(self, idx):
        """
        Retrieve the i-th sample from the dataset, formatted as a tuple containing feature data
        and corresponding metadata.
        
        This method makes the class compatible with PyTorch's DataLoader class, allowing it to
        be used in a loop that fetches batches of data.
        
        Parameters:
            idx (int): The index of the observation to retrieve.
        
        Returns:
            tuple: A tuple containing a tensor of features (X_tensor) and a pandas Series of observation
                   metadata (obs_data).
        """
        # Access the data for the specific index.
        X_data = self.adata.X[idx]
        # If the data supports the toarray method (sparse matrix), convert it; otherwise, use it as is.
        X_tensor = torch.tensor(X_data.toarray() if hasattr(X_data, 'toarray') else X_data, dtype=torch.float32)
        # Retrieve the observation metadata
        obs_data = self.adata.obs.iloc[idx]
        return X_tensor, obs_data

    def get_labels(self, label_key):
        """
        Retrieve the unique labels directly from the specified column in the dataset's DataFrame.

        Parameters:
            label_key (str): The key to access the labels in adata.obs.

        Returns:
            np.ndarray: An array containing unique label values as they appear in the dataset.
        """
        labels = self.adata.obs[label_key].unique()
        return labels


