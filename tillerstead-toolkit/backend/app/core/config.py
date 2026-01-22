"""
BarberX Legal Case Management Pro Suite
Application Configuration
"""
from pydantic_settings import BaseSettings
from typing import Optional, List
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application
    APP_NAME: str = "BarberX Legal Case Management Pro Suite"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./barberx_legal.db"
    
    # Security
    SECRET_KEY: str = "dev-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    # Redis (for background jobs)
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # File storage
    UPLOAD_DIR: str = "./uploads"
    DOCUMENTS_DIR: str = "./uploads/documents"
    EVIDENCE_DIR: str = "./uploads/evidence"
    BWC_DIR: str = "./uploads/bwc"
    EXPORT_DIR: str = "./exports"
    INDEX_DIR: str = "./index"
    
    # BWC Processing
    BWC_ROOT: Optional[str] = "./private-core-barber-cam/.bwc"
    BWC_FILENAME_PATTERN: str = r"^([A-Za-z]+)_(\d{12})_([A-Z0-9]+-?\d*)-?(\d+)?\.mp4$"
    BWC_SYNC_TOLERANCE_MS: int = 5000
    
    # Document Processing
    TESSERACT_CMD: Optional[str] = None
    OCR_LANGUAGE: str = "eng"
    OCR_DPI: int = 300
    MAX_UPLOAD_SIZE_MB: int = 500
    
    # NJ Civil Pleadings
    PLEADINGS_TEMPLATES_DIR: str = "./njcivil_terminal_tool/templates"
    DEFAULT_COURT_COUNTY: str = "Atlantic"
    
    # Analysis
    NLP_MODEL: str = "en_core_web_sm"
    SIMILARITY_THRESHOLD: float = 0.7
    
    # OpenAI / GPT-5.2 Integration
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-5.2"
    OPENAI_MAX_TOKENS: int = 4000
    OPENAI_TEMPERATURE: float = 0.3
    AI_BATCH_CONCURRENCY: int = 3
    
    # CORS Origins
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:4000",
        "https://barberx.info",
        "https://www.barberx.info"
    ]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()


# Create required directories
def init_directories():
    """Create required directories on startup"""
    dirs = [
        settings.UPLOAD_DIR,
        settings.DOCUMENTS_DIR,
        settings.EVIDENCE_DIR,
        settings.BWC_DIR,
        settings.EXPORT_DIR,
        settings.INDEX_DIR,
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
