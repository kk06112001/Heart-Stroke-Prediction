import os
import sys
import logging

# Log format
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Log directory and file path
log_dir = "logs"
log_filepath = os.path.join(log_dir, "logging.log")

# Create log directory if not exist
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

# Get the logger instance
logger = logging.getLogger("Logger")

# Now you can use the logger instance to log messages
logger.info("This is an info message.")
logger.error("This is an error message.")
