from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import yaml
import os 


from .data_args import get_data_args
from .eval_args import get_eval_args
from .utils_args import get_utils_args

def get_args_parser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    # Incorporate all argument definitions
    get_data_args(parser)
    get_eval_args(parser)
    get_utils_args(parser)
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file")
    return parser


def parse_config_and_args():
    """
    Parses command line arguments and configuration file settings.
    
    This function first initializes defaults from a YAML configuration file if specified,
    then parses the command line to allow for overrides of these settings.
    
    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = get_args_parser()
    args = parser.parse_args()  # Parse arguments initially to get the config file path

    # Load configuration from YAML file if specified and exists
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
        # Update the parser defaults with values from the YAML file
        parser.set_defaults(**config)

    # Reparse args to apply command line overrides to the updated defaults
    args = parser.parse_args()
    return args
