from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def linear_model_args(parser: ArgumentParser):
    """Adds arguments specific to LinearModel."""
    parser.add_argument('--optimizer_name', type=str, choices=['sgd', 'adam', 'adamw'], help='Name of the optimizer')
    parser.add_argument('--lr', type=float,  help='Learning rate')
    parser.add_argument('--weight_decay', type=float,  help='Weight decay for optimizer')
    parser.add_argument('--scheduler_name', type=str, choices=['reduce', 'warmup_cosine', 'step', 'exponential', 'none'], help='Name of the scheduler')
    parser.add_argument('--min_lr', type=float, default=0.0, help='Minimum learning rate for warmup scheduler')
    parser.add_argument('--warmup_start_lr', type=float, default=3e-5, help='Initial learning rate for warmup scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs')
    parser.add_argument('--lr_decay_steps', type=int, nargs='+', default=None, help='Steps to decay the learning rate if scheduler is step')
    parser.add_argument('--scheduler_interval', type=str, default='step', choices=['step', 'epoch'], help='Interval to update the lr scheduler')
    parser.add_argument('--seed', type=int, default=42, help='Seed for initializing the linear layer')
    return parser

def decoder_model_args(parser: ArgumentParser):
    """Adds arguments specific to DecoderModel."""
    parser.add_argument('--model_depth', type=int, default=3, help='Decoder Model Layer depth')
    parser.add_argument('--control_key',type=str, default="gene",help="Key to the labels to be used in obs of anndata")
    return parser

def knn_classifier_args(parser: ArgumentParser):
    """Adds arguments specific to WeightedKNNClassifier."""
    parser.add_argument('--k_initial', type=int, default=None, help='Number of neighbors for k-NN')
    parser.add_argument('--T', type=float, default=0.07, help='Temperature for the exponential (cosine distance)')
    parser.add_argument('--max_distance_matrix_size', type=int, default=int(5e6), help='Maximum number of elements in the distance matrix')
    parser.add_argument('--distance_fx', type=str, choices=['cosine', 'euclidean'], default='cosine', help='Distance function')
    parser.add_argument('--epsilon', type=float, default=0.00001, help='Small value for numerical stability (euclidean distance)')
    return parser

def ilisi_args(parser: ArgumentParser):
    """Adds arguments specific to WeightedKNNClassifier."""
    parser.add_argument('--batch_key',type=str, default="gem_group",help="Key to the labels to be used in obs of anndata")
    parser.add_argument("--use_pca", action="store_true", default=False, help="Whether to use PCA for computing iLISI score")
    return parser

def bmdb_args(parser: ArgumentParser):
    """Adds arguments specific to WeightedKNNClassifier."""
    parser.add_argument("--bmdb_path", type=str, default="./datasets/eval/replogle_2022.h5ad", help="path for dataset used for bmdb evaluation")
    parser.add_argument('--recall_threshold', type=float, default=0.05, help='top and smallest Threshold for bmdb retrieval threshold')
    parser.add_argument('--bmdb_pert_col',type=str, default="gene_name",help="Key to the labels of perturbations used for bmdb evaluation")
    parser.add_argument('--bmdb_ctrl_col',type=str, default="is_control",help="Key to the labels of control used for bmdb evaluation")
    return parser

def main_eval_args(parser: ArgumentParser):
    parser.add_argument("--label_key", type=str, default="ensembl_gene_id",help="Key to the labels to be used in obs of anndata")
    parser.add_argument("--control_label", type=str, default="non-targetting",help="Key to the labels to be used in obs of anndata")
    parser.add_argument('--eval_method', type=str, choices=['knn', 'linear', 'ilisi', 'viz', 'reconstruct', 'bmdb','bmdb_precision','all'], default="all", help='Name of the evaluation method')
    return parser

def get_eval_args(parser: ArgumentParser):
    parser = linear_model_args(parser)
    parser = knn_classifier_args(parser)
    parser = ilisi_args(parser)
    parser = decoder_model_args(parser)
    parser = bmdb_args(parser)
    parser = main_eval_args(parser)
    return parser

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = get_eval_args(parser)
    args = parser.parse_args()
    print(args)
