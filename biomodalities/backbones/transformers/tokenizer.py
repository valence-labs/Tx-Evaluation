import math
from functools import partial
from typing import Optional, Union, Callable

import torch
import torch.nn as nn

class TokenLearner(nn.Module):
    """
    A parent class for token learners that handles gene sequence and gene expression data,
    encoding them into a fixed-size token representation for use in models such as transformers.
    This class is now adapted to handle inputs where sequences can include multiple tokens per gene.

    Attributes:
        nb_genes (int): The number of genes.
        embed_dim (int): The size of the embedding dimension for the output token.
        seq_embed (nn.Embedding): Embedding layer for gene sequences.
        expr_proj (nn.Linear): Linear projection for gene expression values.
        expr_conditioned (bool): Flag to determine if expression projections should be conditioned on sequence embeddings.

    Args:
        nb_genes (int, optional): Number of genes. Defaults to 200.
        embed_dim (int, optional): Embedding dimensionality. Defaults to 64.
        expr_conditioned (bool, optional): Conditions expression projection on sequence embeddings. Defaults to False.
    """
    def __init__(self, embed_dim=64, nb_unique_genes=1000, expr_conditioned=False):
        super().__init__()
        self.nb_unique_genes = nb_unique_genes
        self.embed_dim = embed_dim
        self.expr_conditioned = expr_conditioned
        self.seq_embed = nn.Embedding(self.nb_unique_genes + 1, embed_dim)  # Embedding for sequence data
        if expr_conditioned:
            self.expr_proj = nn.Linear(embed_dim + 1, embed_dim)  # Conditioned on both sequence and expression
        else:
            self.expr_proj = nn.Linear(1, embed_dim)  # Only on expression

    def forward(self, seq_indices, expr_values):
        """
        Processes combined gene sequence indices and gene expression values to produce embeddings.
        Handles both single-token and multi-token gene sequences.

        Args:
            seq_indices (Tensor): Tensor of shape (batch_size * nb_genes, nb_tokens) with gene sequence indices.
            expr_values (Tensor): Tensor of shape (batch_size * nb_genes, 1) with gene expression levels.

        Returns:
            Tensor: Combined embeddings of shape (batch_size * nb_genes, embed_dim).
        """
        # Handle multiple tokens by summing or averaging embeddings
        seq_embeddings = self.seq_embed(seq_indices)  # Shape: (batch_size * nb_genes, nb_tokens, embed_dim)
        seq_embeddings = seq_embeddings.mean(dim=1)  # Take the mean across the token dimension

        if self.expr_conditioned:
            combined_input = torch.cat([seq_embeddings, expr_values], dim=-1)  # Concatenate along the feature dimension
            expr_embeddings = self.expr_proj(combined_input)  # Shape: (batch_size * nb_genes, embed_dim)
            return expr_embeddings
        else:
            expr_embeddings = self.expr_proj(expr_values)  # Shape: (batch_size * nb_genes, embed_dim)
            combined_embeddings = seq_embeddings + expr_embeddings  # Sum embeddings
            return combined_embeddings



if __name__ == "__main__":
    # Example usage
    token_learner = TokenLearner()
    

    batch_size = 64
    nb_genes = 200
    nb_tokens = 30  # number of tokens per gene

    # Generate seq_indices with each gene having 30 tokens
    seq_indices = torch.randint(0, 200, (batch_size * nb_genes, nb_tokens))
    
    # Generate expr_values
    expr_values = torch.randn(batch_size * nb_genes, 1)

    # Test the model without expression conditioning
    output = token_learner(seq_indices, expr_values)
    print(f"Output shape without expr_conditioned: {output.shape}")

    # Re-initialize the token learner with expression conditioning
    token_learner = TokenLearner(expr_conditioned=True)

    # Test the model with expression conditioning
    output = token_learner(seq_indices, expr_values)
    print(f"Output shape with expr_conditioned: {output.shape}")