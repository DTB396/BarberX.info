"""
BarberX Legal Case Management Pro Suite
API Routers Module
"""
from app.api.cases import router as cases
from app.api.documents import router as documents
from app.api.evidence import router as evidence
from app.api.analysis import router as analysis
from app.api.exports import router as exports
from app.api.settings import router as settings
from app.api.pleadings import router as pleadings
from app.api.ai import router as ai
from app.api.bwc_analysis import router as bwc_analysis
from app.api.batch_upload import router as batch_upload
from app.api.audio_analysis import router as audio_analysis
from app.api.premium_legal import router as premium_legal
from app.api.firm_management import router as firm_management
from app.api.subscriptions import router as subscriptions

__all__ = [
    "cases",
    "documents",
    "evidence",
    "analysis",
    "exports",
    "settings",
    "pleadings",
    "ai",
    "bwc_analysis",
    "batch_upload",
    "audio_analysis",
    "premium_legal",
    "firm_management",
    "subscriptions",
]
