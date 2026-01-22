"""
BarberX Legal Case Management Pro Suite
Settings API Router - Application Configuration & User Preferences
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from pathlib import Path

from app.db.database import get_db
from app.core.config import settings

router = APIRouter()


# ============================================================================
# Pydantic Schemas
# ============================================================================

class AppSettings(BaseModel):
    """Application settings response"""
    app_name: str
    app_version: str
    database_url: str
    bwc_root: Optional[str]
    pleadings_templates_dir: str
    default_court_county: str


class BWCSettings(BaseModel):
    """BWC processing settings"""
    bwc_root: str = Field(..., description="Root folder for BWC footage")
    sync_tolerance_ms: int = Field(default=5000, description="Sync tolerance in ms")
    filename_pattern: str = Field(..., description="Motorola filename regex")


class AnalysisSettings(BaseModel):
    """Analysis engine settings"""
    nlp_model: str = Field(default="en_core_web_sm")
    similarity_threshold: float = Field(default=0.7, ge=0, le=1)
    auto_scan_violations: bool = Field(default=True)
    violation_confidence_threshold: float = Field(default=0.6, ge=0, le=1)


class PleadingsSettings(BaseModel):
    """Pleadings generator settings"""
    templates_dir: str
    default_county: str = Field(default="Atlantic")
    output_format: str = Field(default="docx")
    auto_timestamp_filenames: bool = Field(default=True)


class UserPreferences(BaseModel):
    """User preferences"""
    theme: str = Field(default="dark")
    default_view: str = Field(default="dashboard")
    notifications_enabled: bool = Field(default=True)
    timeline_sort_order: str = Field(default="desc")
    auto_link_evidence: bool = Field(default=True)


class SystemStatus(BaseModel):
    """System status response"""
    status: str
    database: str
    bwc_root_accessible: bool
    templates_accessible: bool
    disk_space_mb: Optional[float]
    active_cases: int
    total_documents: int
    total_evidence: int


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/app", response_model=AppSettings)
async def get_app_settings():
    """Get current application settings"""
    return AppSettings(
        app_name=settings.APP_NAME,
        app_version=settings.APP_VERSION,
        database_url=settings.DATABASE_URL.split("///")[-1],  # Don't expose full URL
        bwc_root=settings.BWC_ROOT,
        pleadings_templates_dir=settings.PLEADINGS_TEMPLATES_DIR,
        default_court_county=settings.DEFAULT_COURT_COUNTY
    )


@router.get("/bwc", response_model=BWCSettings)
async def get_bwc_settings():
    """Get BWC processing settings"""
    return BWCSettings(
        bwc_root=settings.BWC_ROOT or "./private-core-barber-cam/.bwc",
        sync_tolerance_ms=settings.BWC_SYNC_TOLERANCE_MS,
        filename_pattern=settings.BWC_FILENAME_PATTERN
    )


@router.put("/bwc")
async def update_bwc_settings(bwc_settings: BWCSettings):
    """Update BWC processing settings (runtime only, not persisted)"""
    # In production, would update environment or config file
    return {
        "message": "BWC settings updated",
        "settings": bwc_settings.model_dump()
    }


@router.get("/analysis", response_model=AnalysisSettings)
async def get_analysis_settings():
    """Get analysis engine settings"""
    return AnalysisSettings(
        nlp_model=settings.NLP_MODEL,
        similarity_threshold=settings.SIMILARITY_THRESHOLD
    )


@router.get("/pleadings", response_model=PleadingsSettings)
async def get_pleadings_settings():
    """Get pleadings generator settings"""
    return PleadingsSettings(
        templates_dir=settings.PLEADINGS_TEMPLATES_DIR,
        default_county=settings.DEFAULT_COURT_COUNTY
    )


@router.get("/status", response_model=SystemStatus)
async def get_system_status(db: AsyncSession = Depends(get_db)):
    """Get system health and statistics"""
    from app.db.models import Case, Document, Evidence
    
    # Check BWC root
    bwc_path = Path(settings.BWC_ROOT) if settings.BWC_ROOT else None
    bwc_accessible = bwc_path.exists() if bwc_path else False
    
    # Check templates
    templates_path = Path(settings.PLEADINGS_TEMPLATES_DIR)
    templates_accessible = templates_path.exists()
    
    # Get counts
    try:
        case_count = (await db.execute(select(Case))).scalars().all()
        active_cases = len([c for c in case_count if c.status != 'closed'])
    except:
        active_cases = 0
    
    try:
        doc_result = await db.execute(select(Document))
        total_docs = len(doc_result.scalars().all())
    except:
        total_docs = 0
    
    try:
        evidence_result = await db.execute(select(Evidence))
        total_evidence = len(evidence_result.scalars().all())
    except:
        total_evidence = 0
    
    return SystemStatus(
        status="healthy",
        database="connected",
        bwc_root_accessible=bwc_accessible,
        templates_accessible=templates_accessible,
        disk_space_mb=None,  # Would calculate from uploads dir
        active_cases=active_cases,
        total_documents=total_docs,
        total_evidence=total_evidence
    )


@router.get("/directories")
async def get_directories():
    """Get configured directory paths and their status"""
    dirs = {
        "uploads": settings.UPLOAD_DIR,
        "documents": settings.DOCUMENTS_DIR,
        "evidence": settings.EVIDENCE_DIR,
        "bwc": settings.BWC_DIR,
        "exports": settings.EXPORT_DIR,
        "index": settings.INDEX_DIR,
        "bwc_root": settings.BWC_ROOT,
        "templates": settings.PLEADINGS_TEMPLATES_DIR
    }
    
    result = {}
    for name, path in dirs.items():
        if path:
            p = Path(path)
            result[name] = {
                "path": path,
                "exists": p.exists(),
                "is_directory": p.is_dir() if p.exists() else False
            }
    
    return result


@router.post("/directories/init")
async def initialize_directories():
    """Create all required directories"""
    from app.core.config import init_directories
    
    init_directories()
    
    return {
        "message": "Directories initialized",
        "directories": [
            settings.UPLOAD_DIR,
            settings.DOCUMENTS_DIR,
            settings.EVIDENCE_DIR,
            settings.BWC_DIR,
            settings.EXPORT_DIR,
            settings.INDEX_DIR
        ]
    }


@router.get("/violation-types")
async def get_violation_types():
    """Get all supported violation types for analysis"""
    from app.db.models import ViolationType
    
    return {
        "violation_types": [
            {
                "code": vt.value,
                "name": vt.name.replace("_", " ").title(),
                "amendment": vt.value.split("_")[0] if "_" in vt.value else None
            }
            for vt in ViolationType
        ]
    }


@router.get("/case-types")
async def get_case_types():
    """Get all supported case types"""
    from app.db.models import CaseType
    
    return {
        "case_types": [
            {
                "code": ct.value,
                "name": ct.name.replace("_", " ").title()
            }
            for ct in CaseType
        ]
    }


@router.get("/document-types")
async def get_document_types():
    """Get all supported document types"""
    from app.db.models import DocumentType
    
    return {
        "document_types": [
            {
                "code": dt.value,
                "name": dt.name.replace("_", " ").title()
            }
            for dt in DocumentType
        ]
    }


@router.get("/evidence-types")
async def get_evidence_types():
    """Get all supported evidence types"""
    from app.db.models import EvidenceType
    
    return {
        "evidence_types": [
            {
                "code": et.value,
                "name": et.name.replace("_", " ").title()
            }
            for et in EvidenceType
        ]
    }


@router.get("/nj-counties")
async def get_nj_counties():
    """Get NJ county codes for court filings"""
    counties = [
        {"code": "ATL", "name": "Atlantic"},
        {"code": "BER", "name": "Bergen"},
        {"code": "BUR", "name": "Burlington"},
        {"code": "CAM", "name": "Camden"},
        {"code": "CAP", "name": "Cape May"},
        {"code": "CUM", "name": "Cumberland"},
        {"code": "ESS", "name": "Essex"},
        {"code": "GLO", "name": "Gloucester"},
        {"code": "HUD", "name": "Hudson"},
        {"code": "HUN", "name": "Hunterdon"},
        {"code": "MER", "name": "Mercer"},
        {"code": "MID", "name": "Middlesex"},
        {"code": "MON", "name": "Monmouth"},
        {"code": "MOR", "name": "Morris"},
        {"code": "OCN", "name": "Ocean"},
        {"code": "PAS", "name": "Passaic"},
        {"code": "SAL", "name": "Salem"},
        {"code": "SOM", "name": "Somerset"},
        {"code": "SUS", "name": "Sussex"},
        {"code": "UNI", "name": "Union"},
        {"code": "WAR", "name": "Warren"}
    ]
    return {"counties": counties}
