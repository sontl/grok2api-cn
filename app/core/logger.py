"""Global logging module"""

import sys
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

from app.core.config import setting


# Filter patterns
FILTER_PATTERNS = [
    "chunk: b'",  # SSE raw bytes
    "Got event:",  # SSE events
    "Closing",  # SSE closing
]


class MCPLogFilter(logging.Filter):
    """MCP log filter - Filters out DEBUG logs that contain large amounts of data"""

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter logs"""
        # Filter out SSE DEBUG logs
        if record.name == "sse_starlette.sse" and record.levelno == logging.DEBUG:
            msg = record.getMessage()
            return not any(p in msg for p in FILTER_PATTERNS)

        # Filter out some DEBUG logs from MCP streamable_http
        if "mcp.server.streamable_http" in record.name and record.levelno == logging.DEBUG:
            return False

        return True


class LoggerManager:
    """Logger manager (Singleton)"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize logging system"""
        if LoggerManager._initialized:
            return

        # Log configuration
        log_dir = Path(__file__).parents[2] / "logs"
        log_dir.mkdir(exist_ok=True)
        log_level = setting.global_config.get("log_level", "INFO").upper()
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        log_file = log_dir / "app.log"

        # Configure root logger
        self.logger = logging.getLogger()
        self.logger.setLevel(log_level)

        # Avoid duplicate handler addition
        if self.logger.handlers:
            return

        # Create formatter
        formatter = logging.Formatter(log_format)

        # Create log filter
        mcp_filter = MCPLogFilter()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(mcp_filter)

        # File handler (10MB, 5 backups)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(mcp_filter)

        # Add handlers to root logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        # Configure third-party library log levels
        self._configure_third_party()

        LoggerManager._initialized = True
    
    def _configure_third_party(self):
        """Configure third-party library log levels"""
        config = {
            "asyncio": logging.WARNING,
            "uvicorn": logging.INFO,
            "fastapi": logging.INFO,
            "aiomysql": logging.WARNING,
            "mcp": logging.CRITICAL,
            "fastmcp": logging.CRITICAL,
        }
        
        for name, level in config.items():
            logging.getLogger(name).setLevel(level)

    def debug(self, msg: str) -> None:
        """Debug log"""
        self.logger.debug(msg)

    def info(self, msg: str) -> None:
        """Info log"""
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        """Warning log"""
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        """Error log"""
        self.logger.error(msg)

    def critical(self, msg: str) -> None:
        """Critical error log"""
        self.logger.critical(msg)


# Global logger instance
logger = LoggerManager()
