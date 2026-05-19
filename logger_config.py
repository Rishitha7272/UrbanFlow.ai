import logging
import os
from config import LOG_FILE

def setup_logger():
    """Sets up a structured logger pointing to traffic_system.log."""
    logger = logging.getLogger("TrafficSystem")
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if setup_logger is called multiple times
    if logger.handlers:
        return logger

    # Create directory for log file if it doesn't exist
    log_dir = os.path.dirname(os.path.abspath(LOG_FILE))
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Formatter for structured logs: Timestamp - Level - Module - Message
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File Handler
    file_handler = logging.FileHandler(LOG_FILE, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler (optional but good for real-time visibility in console)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

# Initialize the global logger instance
logger = setup_logger()
