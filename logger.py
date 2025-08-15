import logging
import os
from datetime import datetime
from typing import Optional

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

_ROOT_LOGGER_NAME = "mlflow_wine_root"

def _create_file_handler(filepath: str, level=logging.INFO) -> logging.Handler:
    fh = logging.FileHandler(filepath)
    fh.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    return fh

def init_root_logger(level: int = logging.INFO) -> None:
    """
    Initialize root logger with both console handler and a timestamped file handler.
    Call this once at program start.
    """
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    if logger.handlers:
        return  # already initialized

    logger.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(ch)

    # Initial file handler (timestamped) until run_id is known
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    init_log_path = os.path.join(LOG_DIR, f"run_{timestamp}.log")
    fh = _create_file_handler(init_log_path, level=level)
    logger.addHandler(fh)

    # propagate False so child loggers don't duplicate to root handler
    logger.propagate = False
    logger.info(f"Root logger initialized. File: {init_log_path}")

def switch_to_run_logger(run_id: str, level: int = logging.INFO) -> None:
    """
    Replace the root file handler with a run-specific file handler (logs/run_<run_id>.log).
    Keeps console handler unchanged.
    """
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    if not logger.handlers:
        init_root_logger(level=level)

    target_logfile = os.path.join(LOG_DIR, f"run_{run_id}.log")

    # Remove existing FileHandlers attached to root logger
    handlers_to_remove = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    for h in handlers_to_remove:
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    # Add new file handler for this run
    fh = _create_file_handler(target_logfile, level=level)
    logger.addHandler(fh)
    logger.info(f"Switched logging to run-specific file: {target_logfile}")

def get_logger(name: str) -> logging.Logger:
    """
    Return a module logger that is a child of the root logger.
    Use this in all modules: from logger import get_logger; logger = get_logger(__name__)
    """
    root_logger = logging.getLogger(_ROOT_LOGGER_NAME)
    if not root_logger.handlers:
        init_root_logger()

    module_logger = logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
    # Ensure module logger won't add duplicate handlers; it will propagate to root
    module_logger.propagate = True
    module_logger.setLevel(root_logger.level)
    return module_logger
