import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib.figure import Figure
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_gene_distribution(data_loader: DataLoader, X_index: int = 1) -> Figure:
    """
    Computes the average gene expression across all batches in a data loader and plots the distribution,
    including a shaded area representing the standard deviation. Sorts gene indices to visually approximate
    a Gaussian distribution of the mean values.
    
    Args:
        data_loader (DataLoader): A PyTorch DataLoader object that yields batches of data.
        X_index (int): Index in the batch tuple that refers to the gene counts data.

    Returns:
        Figure: A matplotlib figure object with the plotted sorted average gene expression distribution and
                standard deviation shading.
    """
    logging.info("Initializing sum and squared sum for averaging and variance.")
    sum_gene_expression = None
    sum_squares = None
    total_samples = 0

    # Loop through all batches in the loader
    for i, batch in enumerate(data_loader):
        gene_counts = batch[X_index].to(dtype=torch.float32)

        logging.info(f"Processing batch {i+1}: Converting data for memory efficiency.")

        if sum_gene_expression is None:
            sum_gene_expression = torch.zeros_like(gene_counts[0])
            sum_squares = torch.zeros_like(gene_counts[0])

        sum_gene_expression += gene_counts.sum(dim=0)
        sum_squares += torch.pow(gene_counts, 2).sum(dim=0)
        total_samples += gene_counts.size(0)

        logging.info(f"Updated sum and squared sum: total_samples = {total_samples}")

    # Compute the average and standard deviation expression per gene
    avg_gene_expression = sum_gene_expression / total_samples
    std_deviation = torch.sqrt(sum_squares / total_samples - torch.pow(avg_gene_expression, 2))

    # Sorting genes by average expression
    sorted_indices = torch.argsort(avg_gene_expression)
    avg_gene_expression = avg_gene_expression[sorted_indices]
    std_deviation = std_deviation[sorted_indices]

    # Convert to numpy for plotting
    avg_gene_expression = avg_gene_expression.numpy()
    std_deviation = std_deviation.numpy()

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Sorted Average Gene Expression Distribution with Standard Deviation")
    ax.set_xlabel("Sorted Gene Index")
    ax.set_ylabel("Average Expression")
    ax.plot(avg_gene_expression, label='Sorted Average Gene Expression')
    ax.fill_between(range(len(avg_gene_expression)), 
                    avg_gene_expression - std_deviation, 
                    avg_gene_expression + std_deviation, color='gray', alpha=0.5)
    ax.legend()

    logging.info("Plotting complete.")

    return fig  # Return the figure object for further use

if __name__ == "__main__":
    logging.info("Creating a dummy DataLoader to simulate input.")
    batch_size = 10
    gene_count = 100
    num_samples = 50
    
    data = torch.randn(num_samples, gene_count)
    loader = DataLoader([(data[i], data[i]) for i in range(num_samples)], batch_size=batch_size)

    logging.info("Plotting the gene expression distribution.")
    fig = plot_gene_distribution(loader)
    fig.savefig("gene_distribution.png")
    logging.info("Saved the plot to 'gene_distribution.png'.")
