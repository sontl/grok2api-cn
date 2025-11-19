"""FastAPI Application Main Entry Point"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.core.logger import logger
from app.core.exception import register_exception_handlers
from app.core.storage import storage_manager
from app.core.config import setting
from app.services.grok.token import token_manager
from app.api.v1.chat import router as chat_router
from app.api.v1.models import router as models_router
from app.api.v1.images import router as images_router
from app.api.admin.manage import router as admin_router

# Import MCP server (authentication configuration is completed in server.py)
from app.services.mcp import mcp

# Create MCP's FastAPI application instance
# Using streaming HTTP transport, supporting efficient bidirectional streaming communication
mcp_app = mcp.http_app(stateless_http=True, transport="streamable-http")

# 2. Define application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup sequence:
    1. Initialize core services (storage, settings, token_manager)
    2. Start MCP service lifecycle

    Shutdown sequence (LIFO):
    1. Close MCP service lifecycle
    2. Close core services
    """
    # --- Startup process ---
    # 1. Initialize core services
    await storage_manager.init()

    # Set storage to config and token manager
    storage = storage_manager.get_storage()
    setting.set_storage(storage)
    token_manager.set_storage(storage)

    # Reload config and token data
    await setting.reload()
    token_manager._load_data()
    logger.info("[Grok2API] Core services initialization completed")

    # 2. Manage MCP service lifecycle
    mcp_lifespan_context = mcp_app.lifespan(app)
    await mcp_lifespan_context.__aenter__()
    logger.info("[MCP] MCP services initialization completed")

    logger.info("[Grok2API] Application startup successful")

    try:
        yield
    finally:
        # --- Shutdown process ---
        # 1. Exit MCP service lifecycle
        await mcp_lifespan_context.__aexit__(None, None, None)
        logger.info("[MCP] MCP service closed")

        # 2. Close core services
        await storage_manager.close()
        logger.info("[Grok2API] Application shutdown successful")


# Initialize logger
logger.info("[Grok2API] Application is starting...")

# Create FastAPI application
app = FastAPI(
    title="Grok2API",
    description="Grok API Translation Service",
    version="1.3.1",
    lifespan=lifespan
)

# Register global exception handlers
register_exception_handlers(app)

# Register routers
app.include_router(chat_router, prefix="/v1")
app.include_router(models_router, prefix="/v1")
app.include_router(images_router)
app.include_router(admin_router)

# Mount static files
app.mount("/static", StaticFiles(directory="app/template"), name="template")

@app.get("/")
async def root():
    """Root path"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/login")


@app.get("/health")
async def health_check():
    """Health check interface"""
    return {
        "status": "healthy",
        "service": "Grok2API",
        "version": "1.0.3"
    }

# Mount MCP server
app.mount("", mcp_app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)