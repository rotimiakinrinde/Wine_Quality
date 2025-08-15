from typing import Tuple
import pandas as pd
from logger import get_logger
from config import DATASET_URL

logger = get_logger("data")

def load_data(csv_url: str = DATASET_URL) -> pd.DataFrame:
    """
    Load dataset from CSV URL and return a pandas DataFrame.
    """
    logger.info(f"Loading dataset from: {csv_url}")
    try:
        df = pd.read_csv(csv_url, sep=";")
        logger.info(f"Dataset loaded with shape: {df.shape}")
        return df
    except Exception as e:
        logger.exception("Failed to load dataset.")
        raise
