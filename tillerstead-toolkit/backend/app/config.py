"""
BarberX Legal Case Management Pro Suite
Application Configuration
"""
import os
from typing import List, Optional
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "BarberX Legal Case Management Pro Suite"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./barberx_legal.db"
    
    # File Storage
    UPLOAD_DIR: str = "./uploads"
    DOCUMENTS_DIR: str = "./uploads/documents"
    EVIDENCE_DIR: str = "./uploads/evidence"
    BWC_DIR: str = "./uploads/bwc"
    EXPORTS_DIR: str = "./exports"
    
    # Processing
    MAX_UPLOAD_SIZE_MB: int = 500  # For large BWC files
    ALLOWED_DOCUMENT_TYPES: List[str] = [".pdf", ".doc", ".docx", ".txt"]
    ALLOWED_VIDEO_TYPES: List[str] = [".mp4", ".mov", ".avi", ".mkv"]
    ALLOWED_AUDIO_TYPES: List[str] = [".mp3", ".wav", ".m4a"]
    
    # OCR Settings
    TESSERACT_CMD: Optional[str] = None  # Auto-detect
    OCR_LANGUAGE: str = "eng"
    OCR_DPI: int = 300
    
    # Video Processing
    VIDEO_THUMBNAIL_SIZE: tuple = (320, 240)
    VIDEO_KEYFRAME_INTERVAL: int = 30  # seconds
    AUDIO_SAMPLE_RATE: int = 44100
    
    # Analysis
    NLP_MODEL: str = "en_core_web_sm"
    SIMILARITY_THRESHOLD: float = 0.7
    
    # API
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "https://barberx.info",
        "https://www.barberx.info"
    ]
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Motorola BWC Settings
    BWC_FILENAME_PATTERN: str = r"^([A-Za-z]+)_(\d{12})_([A-Z0-9]+)-(\d+)\.mp4$"
    BWC_SYNC_TOLERANCE_MS: int = 5000  # 5 seconds sync tolerance
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create required directories"""
        for dir_path in [
            self.UPLOAD_DIR,
            self.DOCUMENTS_DIR,
            self.EVIDENCE_DIR,
            self.BWC_DIR,
            self.EXPORTS_DIR
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()


# Violation types and their base damages
VIOLATION_DAMAGES = {
    "4th_unlawful_search": (15000, 75000, 200000),
    "4th_unlawful_seizure": (20000, 100000, 300000),
    "4th_excessive_force": (50000, 200000, 1000000),
    "4th_warrantless_arrest": (25000, 100000, 300000),
    "5th_miranda": (10000, 50000, 150000),
    "5th_self_incrimination": (25000, 100000, 300000),
    "5th_due_process": (15000, 75000, 200000),
    "6th_right_to_counsel": (20000, 100000, 300000),
    "6th_speedy_trial": (15000, 75000, 200000),
    "8th_cruel_punishment": (25000, 150000, 500000),
    "8th_excessive_bail": (10000, 50000, 150000),
    "14th_equal_protection": (30000, 150000, 500000),
    "14th_due_process": (20000, 100000, 300000),
    "brady_violation": (50000, 250000, 1000000),
    "giglio_material": (25000, 150000, 500000),
    "evidence_tampering": (50000, 200000, 750000),
    "policy_violation": (10000, 50000, 150000),
    "training_failure": (25000, 100000, 300000),
    "supervision_failure": (25000, 100000, 300000),
}

# Document classification rules
DOCUMENT_CLASSIFICATIONS = {
    "police_report": [
        "incident report", "police report", "offense report",
        "arrest report", "booking report"
    ],
    "court_filing": [
        "complaint", "answer", "motion", "brief",
        "memorandum of law", "order"
    ],
    "medical_record": [
        "medical record", "hospital", "emergency room",
        "treatment", "diagnosis", "discharge"
    ],
    "bwc_transcript": [
        "body worn camera", "bwc", "body camera",
        "video transcript"
    ],
    "witness_statement": [
        "witness statement", "affidavit", "declaration",
        "deposition", "testimony"
    ],
    "discovery": [
        "interrogatory", "request for production",
        "request for admission", "subpoena"
    ],
    "correspondence": [
        "letter", "email", "communication",
        "notice", "demand"
    ],
    "evidence": [
        "exhibit", "photograph", "evidence",
        "documentation"
    ]
}

# Legal citation patterns
LEGAL_CITATION_PATTERNS = [
    r'\d+\s+U\.S\.\s+\d+',  # Supreme Court
    r'\d+\s+F\.\s*\d*d\s+\d+',  # Federal Reporter
    r'\d+\s+F\.\s*Supp\.\s*\d*d?\s+\d+',  # Federal Supplement
    r'\d+\s+S\.\s*Ct\.\s+\d+',  # Supreme Court Reporter
    r'\d+\s+N\.J\.\s+\d+',  # New Jersey Reports
    r'\d+\s+N\.J\.\s*Super\.\s+\d+',  # NJ Superior Court
    r'\d+\s+A\.\s*\d*d\s+\d+',  # Atlantic Reporter
]

# Officer behavior patterns
OFFICER_BEHAVIOR_PATTERNS = {
    "aggressive": [
        "yelled", "screamed", "threatened", "aggressive",
        "hostile", "intimidating", "belligerent"
    ],
    "unprofessional": [
        "cursed", "profanity", "inappropriate",
        "unprofessional", "disrespectful"
    ],
    "deceptive": [
        "lied", "false", "misrepresented", "fabricated",
        "planted", "concealed"
    ],
    "excessive": [
        "excessive", "unnecessary", "disproportionate",
        "unreasonable", "continued after"
    ],
    "procedural": [
        "failed to", "did not", "violated",
        "ignored", "disregarded"
    ]
}
