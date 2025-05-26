import logging
import logging.handlers
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import queue
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.config import AppConfig  # Assuming AppConfig structure

class JSONFormatter(logging.Formatter):
    """Formats log records as JSONL."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "lineno": record.lineno,
            "threadName": record.threadName,
            "process": record.process,
        }
        
        # Add extra fields if they exist in the record
        if hasattr(record, 'extra') and isinstance(record.extra, dict):
            log_entry.update(record.extra)
        
        # Also check for any extra attributes in __dict__ that aren't standard LogRecord attributes
        standard_attrs = set(logging.LogRecord('', 0, '', 0, '', (), None, None).__dict__.keys())
        extra_attrs = set(record.__dict__.keys()) - standard_attrs
        for attr in extra_attrs:
            # Skip any private attributes or special attributes
            if not attr.startswith('_'):
                log_entry[attr] = getattr(record, attr)
                
        # Handle the case where extra fields were passed directly to the log method
        if 'extra_fields' in record.__dict__ and isinstance(record.__dict__['extra_fields'], dict):
            log_entry.update(record.__dict__['extra_fields'])

        return json.dumps(log_entry)


def setup_logging(app_config: 'AppConfig') -> logging.handlers.QueueListener:
    """
    Sets up the JSONL logging system with asynchronous handling and rotation.

    Args:
        app_config: The application configuration object.

    Returns:
        The QueueListener instance, which should be stopped on application shutdown.
    """
    log_dir = Path(app_config.log_dir) / "LightChat" / "logs"
    os.makedirs(log_dir, exist_ok=True)

    active_log_file = log_dir / "lightchat.log"

    # TODO: Make maxBytes and backupCount configurable via AppConfig
    max_bytes = 10 * 1024 * 1024  # 10MB
    backup_count = 5

    # Create formatter and handler
    json_formatter = JSONFormatter()
    rotating_file_handler = logging.handlers.RotatingFileHandler(
        filename=active_log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    rotating_file_handler.setFormatter(json_formatter)

    # Setup asynchronous logging
    log_queue = queue.Queue(-1)  # Infinite queue size
    queue_handler = logging.handlers.QueueHandler(log_queue)

    # The listener consumes logs from the queue and sends them to the file handler
    listener = logging.handlers.QueueListener(
        log_queue,
        rotating_file_handler,
        respect_handler_level=True
    )

    # Configure the application's logger
    app_logger = logging.getLogger("lightchat")
    app_logger.setLevel(logging.INFO)  # TODO: Make log level configurable via AppConfig
    app_logger.addHandler(queue_handler)
    app_logger.propagate = False # Avoid duplicating logs if root logger is also configured

    listener.start()
    # Pass extra fields using the 'extra' kwarg for logging calls
    app_logger.info("Logging system initialized.", extra={'setup_phase': 'logging'})

    return listener


# Example usage (typically in main.py or app initialization)
if __name__ == "__main__":
    # This is a mock AppConfig for demonstration purposes.
    class MockAppConfig:
        def __init__(self, log_dir):
            self.log_dir = log_dir

    import tempfile
    import time
    import shutil
    import sys

    temp_log_root = Path(tempfile.mkdtemp()) # Create a temp dir for logs
    mock_config = MockAppConfig(log_dir=temp_log_root)

    print(f"Temporary log directory: {temp_log_root / 'LightChat' / 'logs'}")

    listener = setup_logging(mock_config)
    logger = logging.getLogger("lightchat") # Get the same logger instance

    logger.info("This is an info message from example.")
    logger.warning("This is a warning message from example.")
    logger.error("This is an error message from example.", extra={'custom_key': 'custom_value'})

    # Simulate some activity
    for i in range(5):
        logger.info(f"Log entry {i}", extra={'iteration': i})
        time.sleep(0.01)
    
    try:
        1/0
    except ZeroDivisionError:
        logger.exception("A handled ZeroDivisionError occurred.", extra={'error_type': 'ZeroDivisionError'})

    print(f"Check logs at: {temp_log_root / 'LightChat' / 'logs' / 'lightchat.log'}")

    # Stop the listener (important for graceful shutdown)
    listener.stop()
    print("Logging listener stopped.")

    # To inspect logs, comment out the rmtree line below
    # shutil.rmtree(temp_log_root)
    # print(f"Cleaned up temporary directory: {temp_log_root}")
