"""FastAPI Application Main Entry Point"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
from app.services.mcp import mcp

# 0. Compatibility detection
try:
    if sys.platform != 'win32':
        import uvloop
        uvloop.install()
        logger.info("[Grok2API] Enabled uvloop high-performance event loop")
    else:
        logger.info("[Grok2API] Windows system, using default asyncio event loop")
except ImportError:
    logger.info("[Grok2API] uvloop not installed, using default asyncio event loop")

# 1. Create MCP FastAPI application instance
mcp_app = mcp.http_app(stateless_http=True, transport="streamable-http")

# 2. Define application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup sequence:
    1. Initialize core services (storage, settings, token_manager)
    2. Asynchronously load token data
    3. Start batch save task
    4. Start MCP service lifecycle

    Shutdown sequence (LIFO):
    1. Close MCP service lifecycle
    2. Close batch save task and flush data
    3. Close core services
    """
    # --- Startup process ---
    # 1. Initialize core services
    await storage_manager.init()

    # Set storage to config and token manager
    storage = storage_manager.get_storage()
    setting.set_storage(storage)
    token_manager.set_storage(storage)

    # 2. Reload config
    await setting.reload()
    logger.info("[Grok2API] Core services initialization completed")
    
    # 2.5. Initialize proxy pool
    from app.core.proxy_pool import proxy_pool
    proxy_url = setting.grok_config.get("proxy_url", "")
    proxy_pool_url = setting.grok_config.get("proxy_pool_url", "")
    proxy_pool_interval = setting.grok_config.get("proxy_pool_interval", 300)
    proxy_pool.configure(proxy_url, proxy_pool_url, proxy_pool_interval)
    
    # 3. Asynchronously load token data
    await token_manager._load_data()
    logger.info("[Grok2API] Token data loading completed")

    # 4. Start batch save task
    await token_manager.start_batch_save()

    # 5. Manage MCP service lifecycle
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

        # 2. Close batch save task and flush data
        await token_manager.shutdown()
        logger.info("[Token] Token manager closed")

        # 3. Close core services
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
    import os
    
    # Read worker count from environment variable, default is 1
    workers = int(os.getenv("WORKERS", "1"))
    
    # Hint for multi-process mode
    if workers > 1:
        logger.info(
            f"[Grok2API] Multi-process mode enabled (workers={workers}). "
            f"Recommended to use Redis/MySQL storage for best performance."
        )
    
    # Determine event loop type
    loop_type = "auto"
    if workers == 1 and sys.platform != 'win32':
        try:
            import uvloop
            loop_type = "uvloop"
        except ImportError:
            pass
    
    uvicorn.run(
        "main:app",  # Use string to support multi-worker
        host="0.0.0.0",
        port=8001,
        workers=workers,
        loop=loop_type
    )