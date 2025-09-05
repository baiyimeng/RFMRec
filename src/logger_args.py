import os
import time
import argparse
import logging
import yaml


def setup_logging_and_args():
    train_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    config = load_config("config.yaml")
    args = cmdline_args()
    args = merge_config_with_args(config, args)

    log_dir = os.path.join(args.log_file, args.model, args.dataset)
    os.makedirs(log_dir, exist_ok=True)

    log_file_path = os.path.join(log_dir, f"{train_time}{args.description}.log")
    print(f"Log file path: {log_file_path}")
    reset_log(log_file_path)

    return logging.getLogger(__name__), args


def reset_log(log_path):
    fileh = logging.FileHandler(log_path, "a")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fileh.setFormatter(formatter)

    log = logging.getLogger()
    for hdlr in log.handlers[:]:
        log.removeHandler(hdlr)
    log.addHandler(fileh)
    log.setLevel(logging.DEBUG)


def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def merge_config_with_args(config, args):
    config_namespace = argparse.Namespace(**config)
    for key, value in vars(args).items():
        if value is not None:
            setattr(config_namespace, key, value)
    return config_namespace


def str2bool(value):
    if value.lower() in ("true", "1", "t", "y", "yes"):
        return True
    elif value.lower() in ("false", "0", "f", "n", "no"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def cmdline_args():
    """Parse command line arguments with organized grouping"""
    parser = argparse.ArgumentParser()

    # Basic experiment settings
    parser.add_argument(
        "--dataset", help="Dataset name: toys, amazon_beauty, steam, ml-1m"
    )
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--description", type=str, help="Model brief introduction")
    parser.add_argument("--random_seed", type=int, help="Random seed")
    parser.add_argument("--log_file", help="Log directory path")

    # Data processing parameters
    parser.add_argument("--max_len", type=int, help="The max length of sequence")
    parser.add_argument("--batch_size", type=int, help="Batch Size")

    # Model architecture parameters
    parser.add_argument("--hidden_size", type=int, help="Hidden size of model")
    parser.add_argument("--num_blocks", type=int, help="Number of encoder blocks")
    parser.add_argument("--num_heads", type=int, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, help="Dropout of representation")
    parser.add_argument("--emb_dropout", type=float, help="Dropout of item embedding")
    parser.add_argument("--is_causal", type=str2bool, help="Use causal attention")
    parser.add_argument("--pre_norm", type=str2bool, help="Use prenorm transformer")
    parser.add_argument(
        "--pretrained", type=str2bool, help="Use pretrained embedding weight"
    )
    parser.add_argument("--freeze_emb", type=str2bool, help="Freezing embedding weight")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, help="L2 regularization")
    parser.add_argument("--decay_step", type=int, help="Decay step for StepLR")
    parser.add_argument("--gamma", type=float, help="Gamma for StepLR")

    # Evaluation settings
    parser.add_argument("--eval_interval", type=int, help="The number of epoch to eval")
    parser.add_argument(
        "--patience", type=int, help="The number of epoch to wait before early stop"
    )

    # Hardware and deployment
    parser.add_argument("--device", type=str, help="Device: cpu, cuda:0, cuda:1")
    parser.add_argument("--sample_steps", type=int, help="Sample step")

    return parser.parse_args()
