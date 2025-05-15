import logging
import os
import sys

class DefaultLogger:
    def __init__(self, path: str = "log", name: str = None, level: int = logging.DEBUG):
        if name is None:
            raise ValueError("Logger name must be provided")
        
        # Ensure the directory exists
        os.makedirs(path, exist_ok=True)
        # Log file named after the logger
        log_file = os.path.join(path, f"{name}.log")
        
        # Create and configure the internal logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Formatter for log messages
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        
        # Add both handlers to the logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    # Override logging methods
    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)