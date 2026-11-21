"""FastAPI application factory, lifespan management, routers, and static file serving."""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError

from webapp.config import settings
from webapp.routers import system, correct, batch, analytics
from webapp.services.model_manager import model_manager
from webapp.services.batch_jobs import batch_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    # Startup: ensure clean state without loading model
    assert model_manager.state == "unloaded", "Model must start in 'unloaded' state."
    # Resume any pending or interrupted batch jobs
    batch_manager.start_worker()
    yield
    # Shutdown: cleanly release GPU allocations
    await model_manager.unload()



def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="OCR LoRA Correction Service",
        version="0.1.0",
        description="Local evaluation and inference API for fine-tuned Qwen2.5 OCR LoRA adapters.",
        lifespan=lifespan,
    )

    # JSON exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": exc.errors()},
        )

    # Register routers
    app.include_router(system.router)
    app.include_router(correct.router)
    app.include_router(batch.router)
    app.include_router(analytics.router)


    # Mount static assets
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Serve placeholder index.html at root
    @app.get("/", include_in_schema=False)
    async def get_index():
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse({"status": "ok", "message": "OCR LoRA Backend Ready"})

    return app


app = create_app()
