"""
Configuration de l'application
"""
from pydantic_settings import BaseSettings
from typing import List
import os
from pathlib import Path


class Settings(BaseSettings):
    """Configuration globale de l'application"""
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    API_DEBUG: bool = True
    API_VERSION: str = "v1"
    
    # Database Laravel
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "waste_management"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = ""
    
    @property
    def DATABASE_URL(self) -> str:
        """URL de connexion à la base de données"""
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    MODEL_PATH: Path = BASE_DIR / "trained_models"
    DATASET_PATH: Path = BASE_DIR / "datasets"
    
    # Models
    CLASSIFICATION_MODEL: str = "waste_classifier_v1.h5"
    RISK_MODEL: str = "risk_predictor.pkl"
    
    # Image Processing
    MAX_IMAGE_SIZE: int = 10485760  # 10MB
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png"]
    CONFIDENCE_THRESHOLD: float = 0.7
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost",
    ]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Instance globale des settings
settings = Settings()