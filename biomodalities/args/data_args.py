from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def ann_dataset_args(parser: ArgumentParser):
    """Adds arguments for the AnnDataset."""
    parser.add_argument("--dataset_name", type=str, help="Name of the .h5ad dataset file")
    parser.add_argument("--chunk_size", type=int, default=2048, help="Chunk size for reading the data")
    return parser

def ann_loader_args(parser: ArgumentParser):
    """Adds arguments for the AnnLoader."""
    parser.add_argument("--batch_size", type=int, default=2048, help="Number of samples per batch")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker threads for loading data")
    parser.add_argument("--shuffle", action="store_true", default=False,  help="Whether to shuffle data at the beginning of each epoch")
    parser.add_argument("--data_source", type=str, choices=['X', 'obsm'], default='X', help="Specifies whether to load data from 'X' or 'obsm'")
    parser.add_argument("--obsm_key", type=str, default=None, help="The key to access a specific column in adata.obsm if data_source is 'obsm'")
    parser.add_argument("--use_test2_as_test", action="store_true", default=False, help="Whether to use test2 as the test dataset")
    parser.add_argument("--use_bulk_as_test", action="store_true", default=False, help="Whether to use bulk as the test dataset")
    return parser

def get_data_args(parser: ArgumentParser):
    parser = ann_dataset_args(parser)
    parser = ann_loader_args(parser)
    return parser

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = get_data_args(parser)
    args = parser.parse_args()
    print(args)
