import pickle
import time
from logger_args import setup_logging_and_args
from utils import DataTrain, DataVal, DataTest
from utils import setup_seed, format_arguments, format_training_time
from trainer import train_and_evaluate


def set_item_num(args):
    dataset_item_nums = {
        "ml-100k": 1008,
        "yelp": 64669,
        "sports": 12301,
        "baby": 4731,
        "toys": 7309,
        "beauty": 6086,
    }

    if args.dataset in dataset_item_nums:
        args.item_num = dataset_item_nums[args.dataset]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    return args


def load_data(args):
    data_path = f"../datasets/data/{args.dataset}/dataset.pkl"

    with open(data_path, "rb") as file:
        data_raw = pickle.load(file)

    train_dataset = DataTrain(data_raw["train"], args.max_len, args.batch_size)
    val_dataset = DataVal(
        data_raw["train"], data_raw["val"], args.max_len, args.batch_size
    )
    test_dataset = DataTest(
        data_raw["train"],
        data_raw["val"],
        data_raw["test"],
        args.max_len,
        args.batch_size,
    )

    train_loader = train_dataset.get_dataloader()
    val_loader = val_dataset.get_dataloader()
    test_loader = test_dataset.get_dataloader()

    return train_loader, val_loader, test_loader


def main():
    """Main function for the training pipeline"""
    # Record start time for tracking training duration
    start_time = time.time()

    # Initialize logging system and parse command line arguments
    logger, args = setup_logging_and_args()
    args.start_time = start_time
    logger.info("Starting training process...")

    # Set random seed for reproducibility
    setup_seed(args.random_seed)
    logger.info(f"Set random seed to: {args.random_seed}")

    # Set the number of items based on dataset name
    args = set_item_num(args)
    logger.info(f"Dataset: {args.dataset}, Item count: {args.item_num}")

    # Load and prepare training, validation, and test datasets
    logger.info("Loading datasets...")
    train_loader, val_loader, test_loader = load_data(args)
    logger.info("Datasets loaded successfully")

    # Log all configuration parameters for reference
    logger.info("Training arguments:\n%s", format_arguments(args))

    # Execute the main training process
    logger.info("Starting model training...")
    train_and_evaluate(args, train_loader, val_loader, test_loader, logger)

    # Calculate and log total training duration
    training_duration = time.time() - start_time
    logger.info("Training completed in: %s", format_training_time(training_duration))


if __name__ == "__main__":
    main()
