"""
Main application entry point for SMS Transaction Classifier.
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .config.base_config import BaseConfig
from .api.routes import router as api_router
from .persistence.version_control import ModelVersionControl
from .models.classifier import TransactionClassifier

# Configure logging
logging.basicConfig(
    level=getattr(logging, BaseConfig.LOG_LEVEL),
    format=BaseConfig.LOG_FORMAT
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for the FastAPI application.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting up SMS Transaction Classifier...")

    # Create necessary directories
    BaseConfig.create_directories()

    # Initialize version control
    version_control = ModelVersionControl(BaseConfig)
    app.state.version_control = version_control

    # Load latest model version
    try:
        latest_version = await version_control.get_current_active_version()
        if latest_version:
            model = await version_control.get_version(latest_version)
            app.state.model = TransactionClassifier.from_pretrained(model)
        else:
            logger.warning("No active model version found. Loading base model...")
            app.state.model = TransactionClassifier.from_pretrained(BaseConfig.MODEL_NAME)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    logger.info("Startup complete.")
    yield

    # Shutdown
    logger.info("Shutting down SMS Transaction Classifier...")
    # Cleanup resources
    app.state.model = None
    logger.info("Shutdown complete.")

def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    app = FastAPI(
        title=BaseConfig.PROJECT_NAME,
        description="SMS Transaction Classifier with continuous learning capabilities",
        version="1.0.0",
        lifespan=lifespan
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(
        api_router,
        prefix=BaseConfig.API_V1_PREFIX
    )

    return app

app = create_application()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=BaseConfig.DEBUG
    )