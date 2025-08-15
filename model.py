from typing import Tuple
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse

from logger import get_logger, switch_to_run_logger
from config import MLFLOW_TRACKING_URI, MLFLOW_MODEL_NAME, RANDOM_STATE

logger = get_logger("model")

def eval_metrics(actual, pred) -> Tuple[float, float, float]:
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    mae = float(mean_absolute_error(actual, pred))
    r2 = float(r2_score(actual, pred))
    return rmse, mae, r2

def train_and_log(train_x, train_y, test_x, test_y, alpha: float, l1_ratio: float) -> str:
    """
    Train ElasticNet, log params/metrics/model to MLflow.
    Returns MLflow run_id.
    """
    # Configure MLflow tracking (read from env via config)
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"Set MLflow tracking URI from config.")

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Switch logging file to include run_id
        switch_to_run_logger(run_id)

        logger.info(f"Started MLflow run id: {run_id} — training ElasticNet(alpha={alpha}, l1_ratio={l1_ratio})")

        # Train
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=RANDOM_STATE)
        model.fit(train_x, train_y.values.ravel())

        # Predict & evaluate
        preds = model.predict(test_x)
        rmse, mae, r2 = eval_metrics(test_y, preds)

        logger.info(f"Metrics — RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

        # Log to MLflow
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        tracking_scheme = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_scheme != "file":
            mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=MLFLOW_MODEL_NAME)
            logger.info("Model logged to MLflow registry.")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")
            logger.info("Model logged as artifact (local file store).")

        logger.info(f"Run completed: {run_id}")
        return run_id
