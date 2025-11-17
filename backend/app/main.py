from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .core.logger import configure_logging
from .dependencies import get_predictor
from .routers import inference
from .schemas import HealthResponse

configure_logging(settings.log_level)
app = FastAPI(title=settings.app_name, version=settings.version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(inference.router)


@app.on_event("startup")
async def warmup_model() -> None:
    get_predictor()


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    return HealthResponse(name=settings.app_name, version=settings.version)
