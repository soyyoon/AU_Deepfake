"""
FastAPI entry. Loads the pipeline once at startup (lifespan).
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.detect import router as detect_router
from app.config import settings
from app.pipeline.orchestrator import Pipeline
from app.utils.cache import ResultCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading pipeline ...")
    app.state.pipeline = Pipeline()
    app.state.cache = ResultCache()
    logger.info("Pipeline ready.")
    yield
    logger.info("Shutting down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Deepfake Guard API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(detect_router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
