"""Exception handlers - OpenAI-compatible error response"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException


# HTTP Error Map
HTTP_ERROR_MAP = {
    400: ("invalid_request_error", "Request format error or missing required parameters."),
    401: ("invalid_request_error", "Token authentication failed."),
    403: ("permission_error", "No permission to access this resource."),
    404: ("invalid_request_error", "Requested resource does not exist."),
    429: ("rate_limit_error", "Request frequency exceeds limit, please try again later."),
    500: ("api_error", "Internal server error."),
    503: ("api_error", "Service temporarily unavailable."),
}

# Grok Error Code Map
GROK_STATUS_MAP = {
    "NO_AUTH_TOKEN": status.HTTP_401_UNAUTHORIZED,
    "INVALID_TOKEN": status.HTTP_401_UNAUTHORIZED,
    "HTTP_ERROR": status.HTTP_502_BAD_GATEWAY,
    "NETWORK_ERROR": status.HTTP_503_SERVICE_UNAVAILABLE,
    "JSON_ERROR": status.HTTP_502_BAD_GATEWAY,
    "API_ERROR": status.HTTP_502_BAD_GATEWAY,
    "STREAM_ERROR": status.HTTP_502_BAD_GATEWAY,
    "NO_RESPONSE": status.HTTP_502_BAD_GATEWAY,
    "TOKEN_SAVE_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "NO_AVAILABLE_TOKEN": status.HTTP_503_SERVICE_UNAVAILABLE,
}

GROK_TYPE_MAP = {
    "NO_AUTH_TOKEN": "authentication_error",
    "INVALID_TOKEN": "authentication_error",
    "HTTP_ERROR": "api_error",
    "NETWORK_ERROR": "api_error",
    "JSON_ERROR": "api_error",
    "API_ERROR": "api_error",
    "STREAM_ERROR": "api_error",
    "NO_RESPONSE": "api_error",
    "TOKEN_SAVE_ERROR": "api_error",
    "NO_AVAILABLE_TOKEN": "api_error",
}


class GrokApiException(Exception):
    """Grok API Business Exception"""

    def __init__(self, message: str, error_code: str = None, details: dict = None, context: dict = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.context = context or {}
        super().__init__(self.message)


def build_error_response(message: str, error_type: str, code: str = None, param: str = None) -> dict:
    """Build OpenAI-compatible error response"""
    error = {
        "message": message,
        "type": error_type,
    }
    
    if code:
        error["code"] = code
    if param:
        error["param"] = param

    return {"error": error}


async def http_exception_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle HTTP exceptions"""
    error_type, default_message = HTTP_ERROR_MAP.get(exc.status_code, ("api_error", str(exc.detail)))
    message = str(exc.detail) if exc.detail else default_message

    return JSONResponse(
        status_code=exc.status_code,
        content=build_error_response(message, error_type)
    )


async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle validation errors"""
    errors = exc.errors()
    param = errors[0]["loc"][-1] if errors and errors[0].get("loc") else None
    message = errors[0]["msg"] if errors and errors[0].get("msg") else "Request parameter error."

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=build_error_response(message, "invalid_request_error", param=param)
    )


async def grok_api_exception_handler(_: Request, exc: GrokApiException) -> JSONResponse:
    """Handle Grok API exceptions"""
    http_status = GROK_STATUS_MAP.get(exc.error_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
    error_type = GROK_TYPE_MAP.get(exc.error_code, "api_error")

    return JSONResponse(
        status_code=http_status,
        content=build_error_response(exc.message, error_type, exc.error_code)
    )


async def global_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught exceptions"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=build_error_response(
            "Server encountered an unexpected error, please try again.",
            "api_error"
        )
    )


def register_exception_handlers(app) -> None:
    """Register exception handlers"""
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(GrokApiException, grok_api_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)