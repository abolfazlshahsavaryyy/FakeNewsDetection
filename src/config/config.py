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
    PARAM_GRID_LOGISTIC = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "liblinear",
    "max_iter": 100
    }
    logistic_regression_paramater={
        'max_iter':1000
    }
    svc_parameter={
        'max_iter':1000,
        'C':0.01
    }

settings = Settings()
