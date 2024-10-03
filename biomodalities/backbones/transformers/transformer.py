import math
from functools import partial
from typing import Optional, Union, Callable

import torch
import torch.nn as nn

from biomodalities.backbones.transformers.tokenizer import TokenLearner

from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

from biomodalities.utils.misc import trunc_normal_


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class TransformerEncoderLayer(Module):
    r"""
    Mostly copied from torch.nn.TransformerEncoderLayer, but with the following changes:
    - Added the possibility to retrieve the attention weights
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu:
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu:
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super(TransformerEncoderLayer, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, return_attention = False) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        x = src
        if self.norm_first:
            attn, attn_weights = self._sa_block(x = self.norm1(x), attn_mask = src_mask, key_padding_mask = src_key_padding_mask, return_attention = return_attention)
            if return_attention:
                return attn_weights
            x = x + attn
            x = x + self._ff_block(self.norm2(x))
        else:
            attn, attn_weights = self._sa_block(x = self.norm1(x), attn_mask = src_mask, key_padding_mask = src_key_padding_mask, return_attention = return_attention)
            if return_attention:
                return attn_weights
            x = self.norm1(x + attn)
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], return_attention: bool = False) -> Tensor:
        x, attn_weights = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=return_attention,
                            average_attn_weights=False)
        return self.dropout1(x), attn_weights

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)




class Transcriptoformer(nn.Module):
    """Generalist Transformer Encoder for gene expression data of any sequencing technique & depth and number of genes"""
    def __init__(self, embed_dim=32, num_classes=0, depth=12, max_nb_genes=200,
                 num_heads=12, drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, 
                 return_all_tokens=True,expr_conditioned=True,nb_unique_genes=1000, **kwargs):
        super().__init__()

        # Embeddings dimension
        self.num_features = self.embed_dim = embed_dim

        self.nb_genes=max_nb_genes
        self.nb_unique_genes = nb_unique_genes

        # Tokenization module
        self.token_learner = TokenLearner(embed_dim=self.embed_dim,nb_unique_genes=self.nb_unique_genes, expr_conditioned=expr_conditioned)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) 
        # TODO : Do we add a Chromosome Embedding?
        self.pos_drop = nn.Dropout(p=drop_rate)

        # TransformerEncoder block
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(d_model=self.embed_dim, nhead=num_heads, dim_feedforward=512, dropout=dpr[i], batch_first=True)
            for i in range(depth)
        ])
        self.norm = norm_layer(self.embed_dim)

        # Classifier head
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Return only the [CLS] token or all tokens
        self.return_all_tokens = return_all_tokens

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # TODO : Do we need to handle the case where we have more genes than the max nb of genes allowed? is it done automatically?
    def gene_aware_tokenization(self, x, index):
        """
        Processes gene data and prepares it for model input by tokenizing and handling variable gene counts with padding.

        Args:
            x (Tensor): Input data of shape (batch_size * nb_genes, 1+tokendim) where the first column is gene expression level and the rest are gene sequence tokens.
            index (Tensor or list): A list or 1D tensor indicating the number of genes for each item in the batch.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the token embeddings and a mask indicating which tokens are padding. Shape : (B, self.nb_genes, embed_dim), (B, self.nb_genes)
        """
        Bg, t = x.shape
        assert t >= 2, "The input tensor should have at least size 2 in the last dimension."

        expr_values = x[:, 0].unsqueeze(1)  # Extract and reshape expression levels
        seq_indices = x[:, 1:]  # Extract sequence tokens
        # convert to long
        seq_indices = seq_indices.long()

        #print(seq_indices.shape) # torch.Size([290, 20])
        #print(expr_values.shape) # torch.Size([290, 1])

        tokens_per_gene = self.token_learner(seq_indices=seq_indices, expr_values=expr_values)  # Tokenize

        # Calculate cumulative index positions for slicing
        cum_index = [0] + torch.cumsum(torch.tensor(index), 0).tolist()

        # Pre-allocate padded tokens and mask
        padded_tokens = torch.zeros((len(index), max(index), tokens_per_gene.shape[-1]), dtype=tokens_per_gene.dtype, device=tokens_per_gene.device)
        mask = torch.ones((len(index), max(index)), dtype=torch.bool, device=tokens_per_gene.device)

        # Efficient batch filling using advanced indexing
        for i in range(len(index)):
            padded_tokens[i, :index[i]] = tokens_per_gene[cum_index[i]:cum_index[i+1]]
            mask[i, :index[i]] = False

        # padded_tokens shape : (B, self.nb_genes, embed_dim)
        # mask shape : (B, self.nb_genes)
        return padded_tokens, mask

    def forward(self, x, index):
        """
        Processes the input tensor containing gene sequence and expression data through the tokenization, self-attention, and normalization layers, to generate embeddings for each gene or a specific [CLS] token embedding for tasks like classification.

        Args:
            x (Tensor): The input tensor of shape (B * nb_genes, 1 + tokendim), where the first dimension includes both the batch size and the number of genes, concatenated. The second dimension includes one column for gene expression levels and remaining columns for gene sequence tokens.
            index (Tensor or list): Specifies the actual number of genes present for each element in the batch. This is used for correctly padding and masking the input data to ensure consistent tensor dimensions across different batches.

        Returns:
            Tensor: Depending on the configuration of the model (whether self.return_all_tokens is True or False), this method returns either:
                - Non-masked token embeddings (excluding the [CLS] token), shaped dynamically based on the input and the number of actual (non-padded) tokens.
                - Only the [CLS] token embedding, which is typically used for summarization or classification purposes, shaped as (B, embed_dim) where B is the batch size.
        
        The method primarily serves to convert raw gene sequence data into a form suitable for further processing or classification within a deep learning model, taking into account variable gene counts across samples through padding and selective attention masking.
        """
        # x shape : 
        Bg, t = x.shape
        # Apply the TokenLearner module to obtain learnable tokens
        x, gene_mask = self.gene_aware_tokenization(x, index) # output x shape : (B, self.nb_genes, embed_dim), # mask shape : (B, self.nb_genes)

        # Apply the self-attention layers with masked self-attention
        for blk in self.blocks:
            x = blk(x, src_key_padding_mask=gene_mask)  # Use src_key_padding_mask to mask out padded tokens

        # Normalize
        x = self.norm(x)

        if self.return_all_tokens:
            # Create a mask to select non-masked tokens (excluding CLS token)
            non_masked_tokens_mask = ~gene_mask[:, 1:]  # ignoring potential [CLS] token at index 0
            non_masked_tokens = x[:, 1:][non_masked_tokens_mask]
            return non_masked_tokens  # return non-masked tokens (excluding CLS token)
        else:
            return x[:, 0]  # return only the [CLS] token


    def get_last_selfattention(self, x,index):
        """
        Retrieves the last self-attention weights from the final transformer encoder block for the provided input.

        Args:
            x (Tensor): The input tensor of shape (B * nb_genes, 1 + tokendim).
            index (Tensor or list): Specifies the actual number of genes for each batch item.

        Returns:
            Tensor: The self-attention weights from the last transformer block.
        """
        x, gene_mask = self.gene_aware_tokenization(x, index)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, src_key_padding_mask=gene_mask)
            else:
                # Fetch and return attention from the last block
                return blk(x, src_key_padding_mask=gene_mask, return_attention=True)

    def get_intermediate_layers(self, x, index, n=1):
        """
        Retrieves outputs from the last `n` transformer encoder blocks.

        Args:
            x (Tensor): The input tensor of shape (B * nb_genes, 1 + tokendim).
            index (Tensor or list): Specifies the actual number of genes for each batch item.
            n (int): Number of last layers from which to retrieve outputs.

        Returns:
            list: A list containing the outputs of the last `n` blocks.
        """
        x, gene_mask = self.gene_aware_tokenization(x, index)
        output = []
        total_layers = len(self.blocks)
        for i, blk in enumerate(self.blocks):
            x = blk(x, src_key_padding_mask=gene_mask)
            if total_layers - i <= n:
                output.append(self.norm(x))
        return output




if __name__ == "__main__":
    # Parameters
    embed_dim = 4
    num_classes = 0  # Adjust as needed, set to 0 if no classification layer is desired
    depth = 3
    max_nb_genes = 160
    num_heads = 2
    batch_size = 4
    tokendim = 20  # Number of token dimensions after the gene expression level
    max_unique_genes = 1000  # Maximum index for embedding (maximum nb of unique genes available). This normally will e computed when creating the library of genes in train set. New genes would have their own token indice of UNKNOWN

    # Index indicating the actual number of genes for each element in the batch with different values
    index = [120, 50, 20, 70]  # Example where first element has 150 genes, second has 50

    # Create dummy gene expression data (floating point numbers)
    expr_values = torch.randn(sum(index), 1)  # Expression levels are real numbers

    # Create dummy gene sequence token indices (long integers)
    seq_tokens = torch.randint(0, max_unique_genes+1, (sum(index), tokendim), dtype=torch.long)  # Random integers as token indices

    # Combine into a single tensor by concatenating along the last dimension
    # Note: this will convert the entire tensor to a float by default because of the floating-point nature of expr_values
    x = torch.cat([expr_values, seq_tokens.float()], dim=1)  # Convert seq_tokens to float to concatenate

    # Initialize the model
    model = Transcriptoformer(embed_dim=embed_dim, num_classes=num_classes, depth=depth,
                              max_nb_genes=max_nb_genes, num_heads=num_heads, return_all_tokens=False,nb_unique_genes=max_unique_genes)

    # Test the forward pass
    output = model.forward(x, index)
    print("Output from the forward pass:")
    print(output.shape)

    # Test getting last self-attention weights
    last_attention_weights = model.get_last_selfattention(x, index)
    print("Last self-attention weights:")
    print(last_attention_weights.shape)

    # Test getting outputs from the last N layers
    n = 2
    intermediate_outputs = model.get_intermediate_layers(x, index, n)
    print(f"Outputs from the last {n} layers:")
    for layer_output in intermediate_outputs:
        print(layer_output.shape)