"""Global logging module"""

import sys
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
from app.core.config import setting


class MCPLogFilter(logging.Filter):
    """MCP log filter - Filters out DEBUG logs that contain large amounts of data"""

    def filter(self, record):
        # Filter out SSE logs that contain raw byte data
        if record.name == "sse_starlette.sse" and "chunk: b'" in record.getMessage():
            return False

        # Filter out some redundant SSE logs
        if record.name == "sse_starlette.sse" and record.levelno == logging.DEBUG:
            msg = record.getMessage()
            if any(x in msg for x in ["Got event:", "Closing", "chunk:"]):
                return False

        # Filter out some DEBUG logs from MCP streamable_http
        if "mcp.server.streamable_http" in record.name and record.levelno == logging.DEBUG:
            return False

        return True


class LoggerManager:
    """Logger manager"""

    _initialized = False

    def __init__(self):
        """Initialize logging"""
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

        # File handler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(mcp_filter)

        # Add handlers to root logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        # Configure third-party library log levels to avoid excessive debug information
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        logging.getLogger("uvicorn").setLevel(logging.INFO)
        logging.getLogger("fastapi").setLevel(logging.INFO)
        logging.getLogger("aiomysql").setLevel(logging.WARNING)

        # FastMCP related logs - Turn off
        logging.getLogger("mcp").setLevel(logging.CRITICAL)
        logging.getLogger("fastmcp").setLevel(logging.CRITICAL)

        LoggerManager._initialized = True

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
