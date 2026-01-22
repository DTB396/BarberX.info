"""
BarberX Legal Case Management Pro Suite - FastAPI Backend
Constitutional Rights Defense & Evidence Management Platform
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api import cases, documents, evidence, analysis, exports, settings, pleadings, ai, bwc_analysis, batch_upload, audio_analysis, premium_legal, firm_management, subscriptions
from app.db.database import engine, Base
from app.core.config import settings as app_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(
    title="BarberX Legal Case Management Pro Suite",
    description="""
    Constitutional Rights Defense & Evidence Management Platform
    
    ## Features
    - 📄 Batch PDF Upload & OCR Processing
    - 🎥 Body-Worn Camera (BWC) Footage Integration
    - 🔍 Constitutional Violation Analysis
    - 📊 Multi-POV Video Synchronization
    - ⚖️ Liability Assessment Engine
    - 🏛️ Premium Legal Analysis Suite
    - 💼 Firm Management Suite
    
    ## Premium Legal Tools
    - **E-Discovery**: Bates numbering, privilege analysis, document review
    - **Depositions**: Witness management, deposition digests, impeachment tracking
    - **Case Strategy**: Liability assessment, damages calculation, settlement analysis
    - **Deadlines**: FRCP/NJ rules, deadline calculator, litigation timeline
    - **Brady/Giglio**: Exculpatory evidence tracking, officer credibility database
    
    ## Firm Management
    - **Conflict Checking**: Entity matching, corporate family tracking, waiver management
    - **Legal Billing**: Time tracking, LEDES export, budget management
    - **Legal Research**: Citation database, research memos, issue tracking
    
    Built for civil rights litigation and police misconduct cases.
    """,
    version="2.5.0",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:4000",
        "https://barberx.info"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routers
app.include_router(cases.router, prefix="/api/cases", tags=["Cases"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(evidence.router, prefix="/api/evidence", tags=["Evidence & BWC"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
app.include_router(exports.router, prefix="/api/exports", tags=["Exports"])
app.include_router(settings.router, prefix="/api/settings", tags=["Settings"])
app.include_router(ai.router, prefix="/api/ai", tags=["AI / GPT-5.2"])
app.include_router(bwc_analysis.router, prefix="/api/bwc", tags=["BWC Analysis & Sync"])
app.include_router(batch_upload.router, prefix="/api/upload", tags=["Batch Upload & Transcription"])
app.include_router(audio_analysis.router, prefix="/api/audio", tags=["Audio Analysis & Enhancement"])
app.include_router(pleadings.router, prefix="/api/v1", tags=["Pleadings & Filings"])

# Premium Legal Analysis Suite
app.include_router(premium_legal.router, prefix="/api/legal", tags=["Premium Legal Analysis"])

# Firm Management Suite
app.include_router(firm_management.router, prefix="/api/firm", tags=["Firm Management"])

# Subscriptions & Billing
app.include_router(subscriptions.router, prefix="/api/subscriptions", tags=["Subscriptions & Billing"])


@app.get("/")
async def root():
    return {
        "name": "BarberX Legal Case Management Pro Suite",
        "version": "2.5.0",
        "status": "running",
        "description": "Constitutional Rights Defense & Evidence Management Platform",
        "modules": {
            "cases": "Legal case management",
            "documents": "PDF upload, OCR, classification",
            "evidence": "BWC footage, video sync, audio harmonization",
            "analysis": "Constitutional violation scanning",
            "exports": "Reports and evidence binders",
            "pleadings": "NJ Civil pleading generation (complaints, motions, certifications)",
            "premium_legal": {
                "discovery": "E-Discovery, Bates numbering, privilege analysis",
                "depositions": "Witness management, deposition digests, impeachment",
                "strategy": "Liability assessment, damages calculation, settlement analysis",
                "deadlines": "FRCP/NJ court rules, deadline calculator",
                "brady_giglio": "Exculpatory evidence tracking, officer credibility"
            },
            "firm_management": {
                "conflicts": "Conflict of interest checking, waiver management",
                "billing": "Time tracking, LEDES export, budget monitoring",
                "research": "Citation database, research memos, issue tracking"
            }
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "BarberX Legal Suite",
        "version": "2.5.0",
        "components": {
            "database": "connected",
            "pdf_processor": "ready",
            "video_processor": "ready",
            "analysis_engine": "ready",
            "premium_legal": {
                "discovery_service": "ready",
                "deposition_service": "ready",
                "strategy_service": "ready",
                "deadline_calculator": "ready",
                "brady_service": "ready"
            },
            "firm_management": {
                "conflict_service": "ready",
                "billing_service": "ready",
                "research_service": "ready"
            }
        }
    }

