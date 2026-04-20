"""FastAPI application entry point."""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.tasks import router
from .api.dramas import router as drama_router
from .core.detector import get_detector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load NudeNet model on startup
    get_detector()
    yield


app = FastAPI(title="Video Censor Service", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(drama_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
