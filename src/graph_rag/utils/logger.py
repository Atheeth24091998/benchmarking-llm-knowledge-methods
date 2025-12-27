import logging
from pathlib import Path
from datetime import datetime

def get_logger(name: str):
    root = Path(__file__).resolve().parents[4]
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"graph_rag_{datetime.now().strftime('%Y%m%d')}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
