"""FastAPI application factory with lifespan management."""
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.v1.routes import batch, health, jobs, keys, predict
from app.core.config import get_settings
from app.core.model import load_model

# ── Structured Logging Setup ──────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan: load model on startup, cleanup on shutdown.
    Preferred over deprecated @app.on_event.
    """
    settings = get_settings()
    logger.info("startup.begin", model_path=settings.model_path)

    try:
        app.state.bundle_v1 = load_model(settings.model_path, version_tag="v1")
        app.state.model_loaded = True
        logger.info("startup.model_loaded", version="v1")
    except RuntimeError as e:
        logger.error("startup.model_load_failed", error=str(e))
        app.state.bundle_v1 = None
        app.state.model_loaded = False

    # Optional second model for A/B testing
    app.state.bundle_v2 = None
    if settings.model_path_v2:
        try:
            app.state.bundle_v2 = load_model(
                settings.model_path_v2, version_tag="v2"
            )
            logger.info("startup.model_loaded", version="v2")
        except RuntimeError as e:
            logger.warning("startup.v2_load_failed", error=str(e))

    # Initialize drift detection DB
    try:
        from app.services.drift import init_db

        init_db()
    except Exception:
        pass

    # Initialize async jobs DB
    try:
        from app.api.v1.routes.jobs import init_jobs_db

        init_jobs_db()
    except Exception:
        pass

    # Optional: APScheduler for drift checks
    scheduler = None
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler

        from app.services.drift import run_drift_check

        scheduler = AsyncIOScheduler()
        scheduler.add_job(run_drift_check, "interval", hours=24)
        scheduler.start()
        app.state.scheduler = scheduler
        logger.info("startup.scheduler_started")
    except ImportError:
        pass

    yield  # Application runs here

    logger.info("shutdown.begin")
    if scheduler:
        scheduler.shutdown()
    if app.state.bundle_v1:
        del app.state.bundle_v1
    if app.state.bundle_v2:
        del app.state.bundle_v2
    logger.info("shutdown.complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Sentiment Analysis API",
        description=(
            "Fine-tuned DistilBERT model for product review sentiment "
            "classification. Trained on Amazon Reviews dataset with "
            "92%+ accuracy."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Rate Limiter ──────────────────────────────────────────────────
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── CORS ──────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request ID Middleware ─────────────────────────────────────────
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # ── Request Logging Middleware ────────────────────────────────────
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=round(elapsed_ms, 2),
            request_id=getattr(request.state, "request_id", "unknown"),
        )
        return response

    # ── Global Exception Handler ──────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(
            "unhandled_exception", error=str(exc), request_id=request_id
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred.",
                    "request_id": request_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            },
        )

    # ── Optional Prometheus Metrics ───────────────────────────────────
    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    except ImportError:
        pass

    # ── Routers ───────────────────────────────────────────────────────
    app.include_router(health.router, tags=["Health"])
    app.include_router(predict.router, prefix="/api/v1", tags=["Predictions"])
    app.include_router(batch.router, prefix="/api/v1", tags=["Predictions"])
    app.include_router(jobs.router, prefix="/api/v1", tags=["Async Jobs"])
    app.include_router(keys.router, prefix="/api/v1", tags=["Authentication"])

    return app


app = create_app()
