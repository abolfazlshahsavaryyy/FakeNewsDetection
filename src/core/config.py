import os
from pathlib import Path

class Settings:
    PROJECT_NAME: str = "Fake News Detection API"
    VERSION: str = "1.0.0"

    # Model paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    SVM_MODEL_PATH: Path = BASE_DIR / "model" / "model.pkl"
    LOGISTIC_MODEL_PATH: Path = BASE_DIR / "model" / "model2.pkl"

    # Environment
    DEBUG: bool = True
    ENV: str = os.getenv("ENV", "development")

    # Logging, DB, or CORS config can go here too
    # DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./test.db")

settings = Settings()
