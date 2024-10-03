import numpy as np
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split


from biomodalities.data.custom_datasets import AnnDataset
from biomodalities.data.custom_dataloaders import AnnLoader

def make_fun_get_x(n_most_expressed=5000):
    """
    Creates and returns a function that, when called with an Anndata object, extracts a specified
    number of the most expressed features from its `.X` attribute. The `.X` attribute typically
    contains gene expression data in a matrix format, where rows represent samples and columns 
    represent features (e.g., gene expressions).

    The returned function sorts the features based on their total expression across all samples,
    selecting the top `n_most_expressed` features.

    Parameters
    ----------
    n_most_expressed : int, optional
        The number of top-expressed features to select. Default is 5000.

    Returns
    -------
    function
        A function that takes an Anndata object and returns a sub-matrix of the `.X` attribute,
        containing only the columns corresponding to the top `n_most_expressed` most expressed features.

    Example
    -------
    >>> import anndata
    >>> import numpy as np
    >>> data = anndata.AnnData(X=np.random.randn(100, 10000))
    >>> fun_get_x = make_fun_get_x(100)
    >>> X_top_100 = fun_get_x(data)
    >>> X_top_100.shape
    (100, 100)

    Notes
    -----
    The function assumes that the `.X` attribute of the Anndata object is a NumPy array or a similar
    array-like structure that supports indexing. The function does not handle cases where `.X` is
    sparse or structured in a non-conventional format without dense array support.
    """

    def default_fun_get_x(anndata):
        """Extracts a subarray from an Anndata object's .X attribute containing the top `n_most_expressed` features based on total expression.

        Parameters
        ----------
        anndata : anndata.AnnData
            An Anndata object containing at least a `.X` attribute, which holds the gene expression matrix.

        Returns
        -------
        numpy.ndarray
            A subarray of the original `.X` matrix, now containing only the most expressed features.
        """
        # Extract the gene expression matrix
        X = anndata.X

        # Calculate the sum of expressions across samples for each feature and sort the features by this sum in descending order.
        expression_argsort = np.array(np.sum(X, axis=0)).argsort().flatten()[::-1]

        # Select columns corresponding to the top `n_most_expressed` most expressed features
        X = X[:, expression_argsort[:n_most_expressed]]

        return X
    
    return default_fun_get_x

def make_fun_get_y(columns):
    """
    Creates and returns a function that extracts and one-hot encodes target columns from the `.obs` 
    attribute of an Anndata object. This function is typically used in machine learning workflows 
    where categorical labels need to be transformed into a format suitable for model training.

    The `columns` parameter specifies which columns in the Anndata's `.obs` attribute should be
    processed. The returned function will handle both single column names and lists of column names,
    allowing for flexible usage depending on the required targets.

    Parameters
    ----------
    columns : str or list of str
        The name or list of names of the column(s) in the Anndata object's `.obs` attribute 
        that should be extracted and one-hot encoded when the returned function is called.

    Returns
    -------
    function
        A function that accepts an Anndata object, extracts the specified column(s) from its `.obs`
        attribute, one-hot encodes these columns, and returns a dictionary mapping column names 
        to their encoded arrays.

    Example
    -------
    >>> import anndata
    >>> import numpy as np
    >>> data = anndata.AnnData(obs=pd.DataFrame({'cell_type': ['T-cell', 'B-cell', 'T-cell']}))
    >>> fun_get_y = make_fun_get_y('cell_type')
    >>> targets = fun_get_y(data)
    >>> targets['cell_type']
    array([[1., 0.],
           [0., 1.],
           [1., 0.]], dtype=float32)

    Notes
    -----
    This function assumes that all specified columns exist in the `.obs` attribute of the Anndata
    object and that these columns contain categorical data suitable for one-hot encoding.
    """
    if isinstance(columns, str):
        columns = [columns]  # Ensure columns is a list even if a single column name is provided.

    def fun_get_y(anndata):
        """Function to extract and one-hot encode specified columns from an Anndata object.

        Parameters
        ----------
        anndata : anndata.AnnData
            An Anndata object that contains the data in its `.obs` attribute.

        Returns
        -------
        dict of numpy.ndarray
            A dictionary where keys are column names and values are the one-hot encoded matrices of the data in these columns.
        """
        encoded_data = {}
        for column in columns:
            # Retrieve the column data from the Anndata object's `.obs` attribute.
            text_y = np.array(anndata.obs[column])
            if text_y.ndim == 1:
                text_y = text_y.reshape(-1, 1)  # Ensure 2D array for one-hot encoding.

            # Apply one-hot encoding to the column data.
            encoder = OneHotEncoder(sparse_output=False, dtype=np.float32)
            y = encoder.fit_transform(text_y)
            
            # Store the one-hot encoded data in a dictionary using the column name as the key.
            encoded_data[column] = y

        return encoded_data
    
    return fun_get_y



def balance_batches_for_ilisi(dataset, batch_key):
    # Extract labels and find the unique batches and their counts
    batches = np.array(dataset.adata.obs[batch_key])
    unique_batches, batch_counts = np.unique(batches, return_counts=True)
    min_batch_size = np.min(batch_counts)

    # Sample an equal number of indices from each batch
    balanced_indices = []
    for batch in unique_batches:
        batch_indices = np.where(batches == batch)[0]
        np.random.shuffle(batch_indices)
        balanced_indices.extend(batch_indices[:min_batch_size])

    # Create a new balanced dataset using the selected indices
    dataset.adata = dataset.adata[balanced_indices]
    return dataset



def downsample_data(dataset, label_key, seed, sampled_size=100000):
    labels = dataset.adata.obs[label_key].values.astype(str)
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Find single-instance classes
    single_instance_labels = unique_labels[counts == 1]
    multi_instance_labels = unique_labels[counts > 1]

    # Separate indices for single and multiple instance classes
    single_indices = np.flatnonzero(np.isin(labels, single_instance_labels))
    multi_indices = np.flatnonzero(np.isin(labels, multi_instance_labels))

    # Perform stratified sampling on multi-instance indices
    if len(multi_indices) > sampled_size:
        _, idx_resampled_multi = train_test_split(
            multi_indices,
            train_size=sampled_size - len(single_indices),  # adjust size to account for single instances
            stratify=labels[multi_indices],
            random_state=seed
        )
        resampled_indices = np.concatenate([single_indices, idx_resampled_multi])
    else:
        resampled_indices = np.concatenate([single_indices, multi_indices])

    # Subset the original Anndata object
    dataset.adata = dataset.adata[resampled_indices]

    return dataset



if __name__ == "__main__":
    import pandas as pd
    import anndata
    import numpy as np

    # Create data for Anndata
    n_samples = 100
    cell_types = np.random.choice(['T-cell', 'B-cell'], size=n_samples)
    data = anndata.AnnData(X=np.random.randn(n_samples, 10000), obs=pd.DataFrame({'cell_type': cell_types}))

    # Testing make_fun_get_x
    fun_get_x = make_fun_get_x(100)
    X_top_100 = fun_get_x(data)
    print("Shape of X_top_100:", X_top_100.shape)

    # Testing make_fun_get_y
    fun_get_y = make_fun_get_y('cell_type')
    targets = fun_get_y(data)
    print("One-hot encoded 'cell_type':", targets['cell_type'])