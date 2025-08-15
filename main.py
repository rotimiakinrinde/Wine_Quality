import sys
from sklearn.model_selection import train_test_split

from config import DEFAULT_ALPHA, DEFAULT_L1_RATIO, DATASET_URL
from logger import init_root_logger, get_logger
from data import load_data
from model import train_and_log

# Initialize root logger before anything else
init_root_logger()
logger = get_logger("main")

def main(alpha: float = DEFAULT_ALPHA, l1_ratio: float = DEFAULT_L1_RATIO):
    logger.info("Starting pipeline")

    # Load data (DATASET_URL comes from .env/config)
    df = load_data()

    # Train/test split (reproducible using RANDOM_STATE set in config)
    train, test = train_test_split(df, random_state=None)  # using default random split; set random_state if desired

    train_x = train.drop(columns=["quality"])
    test_x = test.drop(columns=["quality"])
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    logger.info(f"Data prepared. train shape: {train_x.shape}, test shape: {test_x.shape}")

    # Train + log to MLflow — this function will switch logging to run-specific file
    run_id = train_and_log(train_x, train_y, test_x, test_y, alpha=alpha, l1_ratio=l1_ratio)

    logger.info(f"Pipeline finished. MLflow run id: {run_id}")

if __name__ == "__main__":
    # CLI overrides
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ALPHA
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_L1_RATIO
    main(alpha, l1_ratio)
