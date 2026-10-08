"""
TruthLens - FastAPI Ingestion Gateway Application
Author: Pratham Yadav & Backend Gateway Team
Sprint 1: Test-framework setup for the ingestion gateway
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import health, ingest, manifest, validate


def create_app() -> FastAPI:
    """Factory function for instantiating the TruthLens Ingestion Gateway FastAPI application."""
    app = FastAPI(
        title="TruthLens Ingestion Gateway API",
        description="High-throughput forensic ingestion, temporal video extraction, and benchmark validation API.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Enable CORS for TruthLens React Frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register Routers
    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(validate.router)
    app.include_router(manifest.router)

    from backend.ingestion.routes import router as ingestion_router
    app.include_router(ingestion_router)

    return app


app = create_app()
