from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from power_plant.routes import router as power_router, load_all_models as load_power_models
from steel_plant.routes import router as steel_router, load_all_models as load_steel_models

logger = logging.getLogger("main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models
    logger.info("🚀 Loading Power Plant models …")
    try:
        await load_power_models()
        logger.info("✅ Power Plant models ready.")
    except Exception as e:
        logger.error("❌ Power Plant model loading failed: %s", e)

    logger.info("🚀 Loading Steel Plant models …")
    try:
        await load_steel_models()
        logger.info("✅ Steel Plant models ready.")
    except Exception as e:
        logger.error("❌ Steel Plant model loading failed: %s", e)

    yield
    # Shutdown: (if needed)

app = FastAPI(
    title="iFactory AI Backend",
    description="Power Plant + Steel Plant AI inference APIs",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping", tags=["meta"])
def ping():
    return {"status": "ok", "message": "pong"}

@app.get("/", tags=["meta"])
def root():
    return {
        "service": "iFactory AI Backend",
        "version": "1.0.0",
        "routes": ["/power", "/steel", "/ping", "/docs"],
    }

app.include_router(power_router, prefix="/power")
app.include_router(steel_router, prefix="/steel")