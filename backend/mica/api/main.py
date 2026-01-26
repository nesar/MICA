"""
MICA FastAPI Application

Main entry point for the MICA backend API server.
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..config import config
from .routes import router
from .websocket import websocket_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.logging.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting MICA API server...")

    # Ensure required directories exist
    config.ensure_directories()

    # Validate configuration
    issues = config.validate()
    if issues:
        logger.warning(f"Configuration issues: {issues}")

    logger.info(f"LLM Provider: {config.llm.llm_provider}")
    logger.info(f"Default Model: {config.llm.default_model}")
    logger.info(f"Session Directory: {config.server.session_dir}")

    yield

    # Shutdown
    logger.info("Shutting down MICA API server...")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI app
    """
    app = FastAPI(
        title="MICA API",
        description="Materials Intelligence Co-Analyst - AI-powered supply chain analysis",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(router, prefix="/api/v1")
    app.include_router(websocket_router, prefix="/ws")

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "version": "0.1.0",
            "llm_provider": config.llm.llm_provider,
        }

    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "MICA API",
            "description": "Materials Intelligence Co-Analyst",
            "version": "0.1.0",
            "docs_url": "/docs",
            "health_url": "/health",
        }

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if config.logging.log_level == "DEBUG" else None,
            },
        )

    return app


# Create app instance
app = create_app()


def run(
    host: Optional[str] = None,
    port: Optional[int] = None,
    reload: bool = False,
):
    """
    Run the MICA API server.

    Args:
        host: Host to bind to (default from config)
        port: Port to bind to (default from config)
        reload: Enable auto-reload for development
    """
    host = host or config.server.host
    port = port or config.server.port

    logger.info(f"Starting MICA API on {host}:{port}")

    uvicorn.run(
        "mica.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=config.logging.log_level.lower(),
    )


if __name__ == "__main__":
    run(reload=True)
