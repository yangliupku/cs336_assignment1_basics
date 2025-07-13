import logging
import pprint
import sys
import os
from datetime import datetime
from typing import Any
from pathlib import Path


class Logger:
    """
    A logger class that uses pprint for formatting complex data structures.
    Each instance can have its own name and configuration.
    By default logs to both console and file with timestamped filenames.
    """

    # Class-level shared configuration
    LOG_FOLDER = (
        Path(__file__).resolve().parent.parent
    ) / "logs"  # Shared logging folder for the whole project
    LOG_FILE_PATTERN = "app_{timestamp}.txt"  # Naming pattern for log files
    SESSION_TIMESTAMP = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )  # Shared timestamp for this session
    _log_file_path = None  # Cached log file path for this session

    @classmethod
    def set_log_folder(cls, folder_path: str):
        """Set the shared logging folder for all logger instances."""
        cls.LOG_FOLDER = folder_path
        cls._log_file_path = None  # Reset cached path

    @classmethod
    def set_log_file_pattern(cls, pattern: str):
        """
        Set the naming pattern for log files.
        Pattern should include {timestamp} placeholder.
        Example: "myapp_{timestamp}.log"
        """
        if "{timestamp}" not in pattern:
            raise ValueError("Log file pattern must include {timestamp} placeholder")
        cls.LOG_FILE_PATTERN = pattern
        cls._log_file_path = None  # Reset cached path

    @classmethod
    def get_log_file_path(cls) -> str:
        """Get the current session's log file path."""
        if cls._log_file_path is None:
            # Create logs directory if it doesn't exist
            log_dir = Path(cls.LOG_FOLDER)
            log_dir.mkdir(exist_ok=True)

            # Generate filename with timestamp
            filename = cls.LOG_FILE_PATTERN.format(timestamp=cls.SESSION_TIMESTAMP)
            cls._log_file_path = str(log_dir / filename)

        return cls._log_file_path

    def __init__(
        self,
        caller: str = None,
        level: int = logging.INFO,
        enable_console: bool = True,
        enable_file: bool = True,
    ):
        """
        Initialize the logger with configuration options.

        Args:
            caller: Caller identifier (typically __file__ or module name)
            level: Logging level (default: INFO)
            enable_console: Whether to log to console (default: True)
            enable_file: Whether to log to file (default: True)
        """
        # Clean up the caller name if it's a file path
        if caller:
            if os.path.sep in caller:
                # Extract just the filename without extension
                caller = os.path.splitext(os.path.basename(caller))[0]
        else:
            caller = "MyPrettyLogger"

        self.caller = caller
        self.level = level
        self.logger = logging.getLogger(caller)
        self.logger.setLevel(level)

        # Only add handlers if this logger doesn't have them yet
        if not self.logger.handlers:
            if enable_console:
                self._setup_console_handler(level)
            if enable_file:
                self._setup_file_handler(level)

        # Configure pprint
        self.pp = pprint.PrettyPrinter(
            indent=2, width=80, depth=None, compact=False, sort_dicts=True
        )

    def _setup_console_handler(self, level: int):
        """Set up console handler with formatting."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def _setup_file_handler(self, level: int):
        """Set up file handler with formatting."""
        log_file_path = self.get_log_file_path()

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(level)

        # Create formatter (slightly more detailed for file logging)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _format_complex_data(self, data: Any) -> str:
        """Format complex data structures using pprint."""
        if isinstance(data, dict | list | set | tuple) and data:
            return f"\n{self.pp.pformat(data)}"
        return str(data)

    def debug(self, message: str, data: Any = None):
        """Log debug message with optional pretty-printed data."""
        if data is not None:
            message = f"{message}: {self._format_complex_data(data)}"
        self.logger.debug(message)

    def info(self, message: str, data: Any = None):
        """Log info message with optional pretty-printed data."""
        if data is not None:
            message = f"{message}: {self._format_complex_data(data)}"
        self.logger.info(message)

    def warning(self, message: str, data: Any = None):
        """Log warning message with optional pretty-printed data."""
        if data is not None:
            message = f"{message}: {self._format_complex_data(data)}"
        self.logger.warning(message)

    def error(self, message: str, data: Any = None):
        """Log error message with optional pretty-printed data."""
        if data is not None:
            message = f"{message}: {self._format_complex_data(data)}"
        self.logger.error(message)

    def critical(self, message: str, data: Any = None):
        """Log critical message with optional pretty-printed data."""
        if data is not None:
            message = f"{message}: {self._format_complex_data(data)}"
        self.logger.critical(message)

    def log_dict(self, level: int, message: str, data: dict):
        """Specifically for logging dictionaries with pretty formatting."""
        formatted_message = f"{message}:\n{self.pp.pformat(data)}"
        self.logger.log(level, formatted_message)

    def log_list(self, level: int, message: str, data: list):
        """Specifically for logging lists with pretty formatting."""
        formatted_message = f"{message}:\n{self.pp.pformat(data)}"
        self.logger.log(level, formatted_message)

    def set_level(self, level: int):
        """Change the logging level for this logger instance."""
        self.level = level
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)

    def add_custom_handler(self, handler: logging.Handler):
        """Add a custom handler to this logger."""
        self.logger.addHandler(handler)

    def get_log_info(self) -> dict[str, Any]:
        """Get information about this logger instance."""
        return {
            "caller": self.caller,
            "level": logging.getLevelName(self.level),
            "level_numeric": self.level,
            "handlers_count": len(self.logger.handlers),
            "log_file": self.get_log_file_path(),
            "log_folder": self.LOG_FOLDER,
            "session_timestamp": self.SESSION_TIMESTAMP,
        }

    def __repr__(self):
        return f"MyPrettyLogger(caller='{self.caller}', level={logging.getLevelName(self.level)})"


# Convenience function to configure global logging settings
def configure_global_logging(
    log_folder: str = "logs",
    log_file_pattern: str = "app_{timestamp}.log",
    default_level: int = logging.INFO,
):
    """
    Configure global logging settings for all MyPrettyLogger instances.

    Args:
        log_folder: Folder to store log files
        log_file_pattern: Pattern for log filenames (must include {timestamp})
        default_level: Default logging level
    """
    Logger.set_log_folder(log_folder)
    Logger.set_log_file_pattern(log_file_pattern)

    # Set root logger level
    logging.getLogger().setLevel(default_level)

    print(f"Logging configured:")
    print(f"  - Log folder: {log_folder}")
    print(f"  - Log file: {Logger.get_log_file_path()}")
    print(f"  - Default level: {logging.getLevelName(default_level)}")


# Example usage and testing
if __name__ == "__main__":
    # Configure global logging settings
    # configure_global_logging(
    #     log_folder="project_logs",
    #     log_file_pattern="myapp_{timestamp}.log",
    #     default_level=logging.DEBUG,
    # )

    # Test the logger with different configurations
    logger1 = Logger(caller=__file__, level=logging.DEBUG)
    logger2 = Logger(caller="test_module", level=logging.INFO)
    logger3 = Logger(
        caller="console_only", level=logging.WARNING, enable_file=False
    )  # Console only
    logger4 = Logger(caller="file_only", level=logging.ERROR, enable_console=False)  # File only

    # Test data
    sample_dict = {
        "application": {
            "name": "MyApp",
            "version": "1.0.0",
            "config": {
                "debug": True,
                "features": ["logging", "pretty_print", "file_output"],
                "database": {
                    "host": "localhost",
                    "port": 5432,
                    "credentials": {"username": "admin", "password": "***hidden***"},
                },
            },
        },
        "runtime": {
            "startup_time": "2025-07-12T10:30:00Z",
            "memory_usage": "45.2MB",
            "active_connections": 12,
        },
    }

    sample_list = [
        {"id": 1, "name": "Task 1", "priority": "high", "tags": ["urgent", "bug"]},
        {"id": 2, "name": "Task 2", "priority": "medium", "tags": ["feature"]},
        {"id": 3, "name": "Task 3", "priority": "low", "tags": ["improvement", "nice-to-have"]},
    ]

    sample_set = {"error", "warning", "info", "debug", "critical"}

    # Test different loggers
    logger1.info("Logger 1 initialized - full logging")
    logger1.debug("Debug information from logger 1", sample_dict)
    logger1.info("Sample list data", sample_list)

    logger2.info("Logger 2 initialized - info level")
    logger2.debug("This debug won't show due to INFO level")
    logger2.warning("Warning with set data", sample_set)

    logger3.warning("Console-only logger warning")
    logger3.error("Console-only error message", {"console": True, "file": False})

    logger4.error("File-only logger error")
    logger4.critical("File-only critical message", {"console": False, "file": True})

    # Show logger information
    print("\nLogger Information:")
    for i, logger in enumerate([logger1, logger2, logger3, logger4], 1):
        info = logger.get_log_info()
        print(f"Logger {i}: {info}")

    # Test specific logging methods
    logger1.log_dict(logging.INFO, "Application configuration", sample_dict)
    logger1.log_list(logging.DEBUG, "Task list", sample_list)

    print(f"\nCheck the log file at: {Logger.get_log_file_path()}")
    print("All loggers share the same log file for this session.")
