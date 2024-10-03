import torch

def log_normalize(X_tensor):
    """
    Applies logarithmic normalization to the data.

    Parameters:
    - X_tensor (torch.Tensor): The input data tensor.

    Returns:
    - torch.Tensor: Logarithmically normalized data tensor.
    """
    return torch.log1p(X_tensor)

def zscore_normalize(X_tensor):
    """
    Applies Z-score normalization to the data.

    Parameters:
    - X_tensor (torch.Tensor): The input data tensor.

    Returns:
    - torch.Tensor: Z-score normalized data tensor.
    """
    mean = torch.mean(X_tensor, dim=0, keepdim=True)
    std = torch.std(X_tensor, dim=0, keepdim=True)
    return (X_tensor - mean) / (std + 1e-6)  # Added epsilon to avoid division by zero

def normalize_data(X_tensor, method='log'):
    """
    Chooses the normalization method and applies it to the data.

    Parameters:
    - X_tensor (torch.Tensor): The input data tensor.
    - method (str): The normalization method ('log' or 'zscore').

    Returns:
    - torch.Tensor: The normalized data tensor.
    """
    if method == 'log':
        return log_normalize(X_tensor)
    elif method == 'zscore':
        return zscore_normalize(X_tensor)
    else:
        raise ValueError("Unsupported normalization method")

    
if __name__ == "__main__":
    # Generate sample data
    data = torch.rand(10, 5) * 100  # Random data tensor
    print("Original Data:\n", data)

    # Log normalization
    log_normalized_data = normalize_data(data, 'log')
    print("Log Normalized Data:\n", log_normalized_data)

    # Z-score normalization
    zscore_normalized_data = normalize_data(data, 'zscore')
    print("Z-score Normalized Data:\n", zscore_normalized_data)

