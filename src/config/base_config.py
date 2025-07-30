"""
Base configuration for the SMS Transaction Classifier system.
"""
from typing import Dict
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class BaseConfig:
    # Project Paths
    ROOT_DIR = Path(__file__).parent.parent.parent
    MODEL_DIR = ROOT_DIR / "models"
    DATA_DIR = ROOT_DIR / "data"
    LOG_DIR = ROOT_DIR / "logs"

    # Model Configuration
    MODEL_NAME = "bert-base-uncased"
    MAX_SEQ_LENGTH = 128
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    NUM_TRAIN_EPOCHS = 3
    WARMUP_STEPS = 500

    # Persistence Configuration
    MODEL_BACKUP_COUNT = 3
    BACKUP_INTERVAL_HOURS = 24
    CHECKPOINT_BEFORE_TRAINING = True

    # Monitoring Configuration
    PERFORMANCE_THRESHOLD = 0.95
    MAX_PERFORMANCE_DEGRADATION = 0.02
    ALERT_ON_DEGRADATION = True

    # API Configuration
    API_V1_PREFIX = "/api/v1"
    PROJECT_NAME = "SMS Transaction Classifier"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/sms_classifier")

    # Redis Configuration
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

    # Security Configuration
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30

    # Logging Configuration
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def create_directories(cls) -> None:
        """Create necessary project directories."""
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_model_path(cls, version: str) -> Path:
        """Get path for a specific model version."""
        return cls.MODEL_DIR / f"model-{version}"

    @classmethod
    def to_dict(cls) -> Dict:
        """Convert configuration to dictionary."""
        return {
            key: value for key, value in cls.__dict__.items() 
            if not key.startswith('__') and not callable(value)
        }