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
from biomodalities.data.custom_datasets import AnnDataset
import sys

# TODO : Inherit from dataloader of pytorch
# TODO : Adapt output to be exactly type of pytorch dataloader if it doesn't happen when inheriting
class AnnLoader:
    """
    A data loader for loading batches of data from an AnnDataset. This loader supports
    multi-threading to prefetch data and can optionally shuffle data each epoch, making it
    suitable for machine learning training loops.
    
    Attributes:
        dataset (AnnDataset): The dataset from which to load data.
        adata (anndata.AnnData): Direct access to the AnnData object for low-level operations.
        var (DataFrame): Variable information from the dataset.
        batch_size (int): Number of samples per batch.
        n_obs (int): Total number of observations in the dataset.
        initialization_time (float): Time taken to initialize the loader.
        num_workers (int): Number of worker threads for loading data.
        shuffle (bool): Whether to shuffle data at the beginning of each epoch.
        indices (np.array): Array of indices representing the data.
        pool (ThreadPool): Pool of worker threads.
        queue (Queue): Queue for holding prefetched data batches.
        next_batch_idx (int): Index of the next batch to be loaded.
        condition (threading.Condition): Condition variable to synchronize prefetching.
        task (str) : Task undergone.
    """
    def __init__(self,  dataset, batch_size=1000, num_workers=1, shuffle=False, data_source='X', obsm_key=None, task=None):
        """
        Initializes the AnnLoader with specified parameters.
        
        Parameters:
            dataset (AnnDataset): The dataset to load.
            batch_size (int, optional): The number of samples per batch.
            num_workers (int, optional): The number of worker threads to use for loading data.
            shuffle (bool, optional): If True, shuffle data indices at the start of each epoch.
            data_source (str, optional): Specifies whether to load data from 'X' or 'obsm'.
            obsm_key (str, optional): The key to access a specific column in adata.obsm if data_source is 'obsm'.
            task (str, optional) : Task undergone.
        """
        start_time = time.time()
        # Initialize AnnData in backed mode to not load everything into memory
        self.dataset = dataset
        self.adata = dataset.adata
        self.var = dataset.var
        self.batch_size = batch_size
        self.n_obs = self.adata.n_obs
        self.initialization_time = time.time() - start_time
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.indices = np.arange(self.n_obs)
        self.pool = ThreadPool(num_workers)
        self.queue = Queue(maxsize=num_workers * 2)  # To hold pre-fetched batches
        self.next_batch_idx = 0
        self.condition = threading.Condition()
        self.data_source = data_source
        self.obsm_key = obsm_key
        self.task = task
        self.reset_epoch()  # Initial setup and shuffling
        # Setup finalizer to ensure resources are cleaned up when object is garbage collected
        weakref.finalize(self, self.close)

    def reset_epoch(self):
        """
        Resets indices for a new epoch, shuffles if necessary, and clears prefetch queue.
        """
        with self.condition:
            if self.shuffle:
                np.random.shuffle(self.indices)
            self.next_batch_idx = 0
            self.queue.queue.clear()  # Clear the current queue
            self._prefetch()


    def _prefetch(self):
        """
        Prefetches data batches asynchronously using worker threads.
        """
        with self.condition:
            while not self.queue.full() and self.next_batch_idx < len(self): 
                end_idx = min((self.next_batch_idx + 1) * self.batch_size, len(self.indices)) 
                batch_indices = self.indices[end_idx-self.batch_size:end_idx]
                self.queue.put(self.pool.apply_async(self._load_batch, args=(batch_indices,)))
                self.next_batch_idx += 1 
            if self.next_batch_idx >= len(self): 
                self.next_batch_idx = 0  # Reset for the next epoch
            self.condition.notify_all()  # Notify any waiting threads that new data may be available


    def __len__(self):
        """
        Returns the total number of full batches available in the dataset.
        """
        return self.n_obs // self.batch_size

    def _load_batch(self, indices):
        """
        Loads a batch of data given specific indices, ensuring data is sorted by index.
        
        Parameters:
            indices (array-like): Indices for the batch to load.
            
        Returns:
            tuple: A batch of data as (feature tensor, metadata).
        """
        ground_truth_X = None
        sorted_indices = np.sort(indices)  # Sort indices to ensure proper ordering
        if self.data_source == 'X':
            # TODO : Benchmark against AnnData.chunk_X function
            X_data = self.adata.X[sorted_indices].toarray() if hasattr(self.adata.X[sorted_indices], 'toarray') else self.adata.X[sorted_indices]
        elif self.data_source == 'obsm' and self.obsm_key :
            if self.task == "reconstruct" :
                ground_truth_X = self.adata.X[sorted_indices].toarray() if hasattr(self.adata.X[sorted_indices], 'toarray') else self.adata.X[sorted_indices]
                ground_truth_X = torch.from_numpy(ground_truth_X).float()
            if self.obsm_key == 'random':
                X_data = emb = np.random.rand(len(sorted_indices), 512)
            elif self.obsm_key == 'random_scramble_PCA' :
                X_data = self.adata.obsm['PCA'][sorted_indices].copy()
                # randomize order of np array X_data in dimension 0
                np.random.shuffle(X_data)
            else :
                X_data = self.adata.obsm[self.obsm_key][sorted_indices]
        else:
            raise ValueError("Invalid data source or obsm_key not specified when required.")
        obs_data = self.adata.obs.iloc[sorted_indices]
        obs_data = [ground_truth_X,obs_data] if self.task == "reconstruct" else obs_data

        X_tensor = torch.from_numpy(X_data).float()
        return X_tensor, obs_data
            

    def __getitem__(self, idx):
        """
        Retrieves a batch by index, waiting if necessary for the batch to be available.
        
        Parameters:
            idx (int): Index of the batch to retrieve.
        
        Returns:
            tuple: The requested batch of data.
        
        Raises:
            IndexError: If the index is out of the range of total batches.
            RuntimeError: If waiting for a batch times out.
        """
        if idx >= len(self) or idx < 0:
            raise IndexError("Batch index out of total number of batches.")
        while True:
            try:
                with self.condition:
                    while self.queue.empty():
                        self._prefetch()
                    future = self.queue.get()
                    X_tensor, batch = future.get() 
                    if idx == len(self) - 1 :  # Last batch of the epoch
                        self.reset_epoch()  # Reset for the next epoch
                    if X_tensor is not None:
                        return X_tensor, batch
            except Empty:
                continue  # If the queue was empty, loop around and check again
            except TimeoutError:
                raise RuntimeError("Timeout waiting for batch to be available.")

    def close(self):
        """
        Closes the thread pool and notifies all threads to avoid deadlocks.
        """
        if self.pool:
            self.pool.close()
            self.pool.join()
        with self.condition:
            self.condition.notify_all()  # Ensure no threads are left waiting if we're closing down








if __name__ == "__main__" :
    #filename = "/rxrx/data/valence/biomodality_sc/replogle_2022/replogle_2022.h5ad"
    filename = "/mnt/ps/home/CORP/ihab.bendidi/sandbox/biomodal_codebase/biomodalities/trivia/shawn_data_dense/rpe1_raw_singlecell_01.h5ad"
    anndataset = AnnDataset(filename, mode='r', chunk_size=128)
    data_loader = AnnLoader(anndataset, batch_size=2048, shuffle=True,num_workers=4)                                     
    print(f"Total batches: {len(data_loader)}")
    print(f"Total observations: {len(anndataset)}")


    # Testing with obsm data
    data_loader_obsm = AnnLoader(anndataset, batch_size=2048, shuffle=True, num_workers=4, data_source='obsm', obsm_key='X_uce_4_layers')
    print(f"Total batches (obsm data): {len(data_loader_obsm)}")


    print("\n####### Testing anndata obsm loader:")

    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}:")
        # Iterate over the first few batches and print some details
        for i in range(min(300, len(data_loader_obsm))):  # Just the first  batches for brevity
            X_tensor, batch_ann_data = data_loader_obsm[i]
            print(f"\nBatch {i + 1}:")
            print(f"Shape of X_obsm tensor: {X_tensor.shape}")
            print(f"Shape of obs frame: {batch_ann_data.shape}")
            print(f"Shape of var frame: {data_loader.var.shape}")

    print("\n####### Testing anndata X loader:")

    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}:")
        # Iterate over the first few batches and print some details
        for i in range(min(300, len(data_loader))):  # Just the first  batches for brevity
            X_tensor, batch_ann_data = data_loader[i]
            print(f"\nBatch {i + 1}:")
            print(f"Shape of X tensor: {X_tensor.shape}")
            print(f"Shape of obs frame: {batch_ann_data.shape}")
            print(f"Shape of var frame: {data_loader.var.shape}")
    print("\nTesting completed successfully.")
