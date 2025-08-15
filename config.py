import os
from dotenv import load_dotenv

load_dotenv()  # loads .env in project root

# MLflow tracking
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# Model registry name (optional)
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "ElasticnetWineModel")

# Dataset URL (kept in .env to satisfy "no hardcoding in scripts")
DATASET_URL = os.getenv(
    "DATASET_URL",
    "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
)

# Defaults for training
DEFAULT_ALPHA = float(os.getenv("DEFAULT_ALPHA", 0.5))
DEFAULT_L1_RATIO = float(os.getenv("DEFAULT_L1_RATIO", 0.5))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))

# AWS (optional)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
