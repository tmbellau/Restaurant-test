"""FastAPI application factory."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import locations, predictions, metrics


def create_app() -> FastAPI:
    app = FastAPI(
        title="Wagamama Forecast API",
        version="0.2.0",
        description="Restaurant demand prediction platform",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(locations.router, prefix="/api/v1", tags=["locations"])
    app.include_router(predictions.router, prefix="/api/v1", tags=["predictions"])
    app.include_router(metrics.router, prefix="/api/v1", tags=["metrics"])

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
