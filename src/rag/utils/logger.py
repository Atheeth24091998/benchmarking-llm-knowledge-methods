import logging
from pathlib import Path
from typing import Optional


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "rag_pipeline.log"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Create and return a configured logger.

    Args:
        name (str): Logger name (usually __name__)

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate logs if logger already configured
    if logger.handlers:
        return logger

    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
