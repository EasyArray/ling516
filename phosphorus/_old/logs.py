"""Custom logging setup for Phosphorus."""

import logging
from logging.handlers import MemoryHandler

# ANSI escape codes for colors
COLORS = {
    "DEBUG": "\033[90m",  # Gray
    "INFO": "",            # Default color
    "WARNING": "\033[38;5;208m",  # Orange
    "ERROR": "\033[91m",   # Red
    "CRITICAL": "\033[41m\033[97m",  # White on Red Background
    "RESET": "\033[0m"    # Reset color
}

class ColorFormatter(logging.Formatter):
  """Custom formatter to add color based on log level."""
  def format(self, record):
    log_color = COLORS.get(record.levelname, COLORS["RESET"])
    message = super().format(record)
    return f"{log_color}{message}{COLORS['RESET']}"  # Wrap message in color codes


# Create a stream handler (prints logs to console)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(ColorFormatter("%(message)s"))

# Create a memory handler to buffer logs below WARNING level
memory_handler = MemoryHandler(capacity=1000, target=console_handler, flushLevel=logging.CRITICAL)
memory_handler.setLevel(logging.INFO)

# Set up root logger
logger = logging.getLogger("BufferedLogger")
logger.setLevel(logging.DEBUG)  # Capture all logs internally
logger.addHandler(console_handler)
logger.propagate = False # Prevent double logging in Colab
