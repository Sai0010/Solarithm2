import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Database settings
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./solarithm.db")

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_RELOAD = os.getenv("API_RELOAD", "True").lower() == "true"

# ML model settings
ML_MODEL_PATH = BASE_DIR / "ml_model" / "trained_lstm.h5"
ML_SCALER_PATH = BASE_DIR / "ml_model" / "scaler.pkl"
ML_CONFIG_PATH = BASE_DIR / "ml_model" / "train_config.json"

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"