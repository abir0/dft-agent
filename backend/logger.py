import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from backend.utils.date import get_datetime


def setup_logger(
    name: str,
    log_dir: str = "logs",
    log_file: str = "scraping.log",
    interval: int = 10,
):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create parent dirs for logs with current date
    logs_path = Path(log_dir) / get_datetime("%Y-%m-%d")
    logs_path.mkdir(parents=True, exist_ok=True)

    # Create a handler for rotating log files
    filename = logs_path / log_file
    handler = TimedRotatingFileHandler(filename, when="H", interval=interval)
    handler.setLevel(logging.INFO)

    # Set the log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    logging.captureWarnings(True)

    return logger


logger = setup_logger(__name__)
