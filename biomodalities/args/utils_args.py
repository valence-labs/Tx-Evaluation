from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def utils_args(parser: ArgumentParser):
    """Adds training-related arguments to the parser."""
    # General training arguments
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs to train')
    parser.add_argument('--gpus', type=int, nargs='+', default=None, help='List of GPUs to use, e.g., 0 1 2')
    parser.add_argument('--accelerator', type=str, default='gpu', choices=['gpu', 'cpu', 'tpu'], help='Type of accelerator to use')
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32], help='Precision level to use')
    parser.add_argument('--distributed_backend', type=str, default=None, choices=['dp', 'ddp', 'ddp2', 'horovod'], help='Distributed backend to use for multi-GPU training')

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_monitor', type=str, default='val_loss', help='Metric to monitor for checkpointing')
    parser.add_argument('--checkpoint_save_top_k', type=int, default=1, help='Number of top checkpoints to save')
    parser.add_argument('--checkpoint_mode', type=str, default='min', choices=['min', 'max'], help='Mode for monitoring metric')

    # Logging
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--log_every_n_steps', type=int, default=50, help='Log every n steps')

    # Early Stopping
    parser.add_argument('--early_stop_monitor', type=str, default='val_loss', help='Metric to monitor for early stopping')
    parser.add_argument('--early_stop_patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--early_stop_mode', type=str, default='min', choices=['min', 'max'], help='Mode for monitoring metric')

    # WandB integration
    parser.add_argument('--wandb_project_name', type=str, default='default_project', help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default='default_entity', help='Weights & Biases entity (user or team)')
    parser.add_argument('--run_name', type=str, default='run', help='Weights & Biases run name')

    parser.add_argument("--debug", action="store_true", default=False, help="Whether we store results in wandb debug project or main project")
    
    return parser

def get_utils_args(parser: ArgumentParser):
    parser = utils_args(parser)
    return parser

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = get_utils_args(parser)
    args = parser.parse_args()
    print(args)
