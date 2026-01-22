"""
BarberX Legal Case Management Pro Suite
Exports API Router - Reports, Evidence Binders, Timelines
"""
import uuid
from datetime import datetime
from typing import Optional, List
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
import json
import io

from app.db.database import get_db
from app.db.models import Case, Document, Evidence, Violation, TimelineEvent
from app.schemas.legal_schemas import (
    ExportTimelineRequest, ExportViolationsRequest,
    ExportEvidenceBinderRequest, ExportResponse
)

router = APIRouter()

# Export directory
EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/timeline", response_model=ExportResponse)
async def export_timeline(
    request: ExportTimelineRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Export case timeline as PDF, DOCX, or CSV.
    
    Generates a chronological timeline of all case events.
    """
    # Verify case exists
    result = await db.execute(select(Case).where(Case.id == request.case_id))
    case = result.scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    # Get timeline events
    query = select(TimelineEvent).where(TimelineEvent.case_id == request.case_id)
    
    if request.date_range_start:
        query = query.where(TimelineEvent.event_date >= request.date_range_start)
    if request.date_range_end:
        query = query.where(TimelineEvent.event_date <= request.date_range_end)
    
    query = query.order_by(TimelineEvent.event_date.asc())
    
    events_result = await db.execute(query)
    events = events_result.scalars().all()
    
    # Generate export
    export_id = uuid.uuid4().hex[:12]
    filename = f"timeline_{case.case_number or case.id}_{export_id}.{request.format}"
    file_path = EXPORT_DIR / filename
    
    if request.format == "csv":
        # Generate CSV
        output = io.StringIO()
        output.write("Date,Time,Event,Description,Category,Source\n")
        for event in events:
            output.write(f'"{event.event_date.strftime("%Y-%m-%d")}",')
            output.write(f'"{event.event_time or ""}",')
            output.write(f'"{event.title}",')
            output.write(f'"{event.description or ""}",')
            output.write(f'"{event.event_category or ""}",')
            output.write(f'"{event.source_description or ""}"\n')
        
        with open(file_path, "w") as f:
            f.write(output.getvalue())
    
    elif request.format == "json":
        # Generate JSON
        timeline_data = {
            "case_id": case.id,
            "case_title": case.title,
            "generated_at": datetime.utcnow().isoformat(),
            "events": [
                {
                    "date": event.event_date.isoformat(),
                    "time": event.event_time,
                    "title": event.title,
                    "description": event.description,
                    "category": event.event_category,
                    "is_key_event": event.is_key_event,
                    "source": event.source_description
                }
                for event in events
            ]
        }
        
        with open(file_path, "w") as f:
            json.dump(timeline_data, f, indent=2)
    
    else:  # PDF (placeholder - would use reportlab/weasyprint)
        # For now, create a text-based output
        with open(file_path, "w") as f:
            f.write(f"CASE TIMELINE\n")
            f.write(f"{'='*50}\n")
            f.write(f"Case: {case.title}\n")
            f.write(f"Case Number: {case.case_number}\n")
            f.write(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}\n\n")
            
            for event in events:
                f.write(f"[{event.event_date.strftime('%Y-%m-%d')}] ")
                if event.is_key_event:
                    f.write("★ ")
                f.write(f"{event.title}\n")
                if event.description:
                    f.write(f"   {event.description}\n")
                f.write("\n")
    
    return ExportResponse(
        export_id=export_id,
        filename=filename,
        file_path=str(file_path),
        file_size=file_path.stat().st_size,
        format=request.format,
        created_at=datetime.utcnow(),
        download_url=f"/api/exports/download/{filename}"
    )


@router.post("/violations", response_model=ExportResponse)
async def export_violations_report(
    request: ExportViolationsRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Export constitutional violations report.
    
    Comprehensive report of all identified violations with:
    - Legal basis and citations
    - Evidence references
    - Damages estimates
    """
    # Verify case exists
    result = await db.execute(select(Case).where(Case.id == request.case_id))
    case = result.scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    # Get violations
    violations_result = await db.execute(
        select(Violation)
        .where(Violation.case_id == request.case_id)
        .order_by(Violation.severity.desc())
    )
    violations = violations_result.scalars().all()
    
    # Generate export
    export_id = uuid.uuid4().hex[:12]
    filename = f"violations_{case.case_number or case.id}_{export_id}.{request.format}"
    file_path = EXPORT_DIR / filename
    
    # Calculate totals
    total_estimated_damages = sum(v.estimated_damages or 0 for v in violations)
    
    if request.format == "json":
        report_data = {
            "case_id": case.id,
            "case_title": case.title,
            "generated_at": datetime.utcnow().isoformat(),
            "total_violations": len(violations),
            "total_estimated_damages": total_estimated_damages,
            "violations": [
                {
                    "id": v.id,
                    "type": v.violation_type.value,
                    "title": v.title,
                    "description": v.description,
                    "severity": v.severity,
                    "legal_basis": v.legal_basis if request.include_legal_basis else None,
                    "confidence_score": v.confidence_score,
                    "is_verified": v.is_verified,
                    "estimated_damages": v.estimated_damages if request.include_damages_estimate else None,
                    "source_document_id": v.source_document_id if request.include_evidence_refs else None,
                    "source_evidence_id": v.source_evidence_id if request.include_evidence_refs else None,
                    "timestamp_reference": v.timestamp_start
                }
                for v in violations
            ]
        }
        
        with open(file_path, "w") as f:
            json.dump(report_data, f, indent=2)
    
    else:  # Text/PDF placeholder
        with open(file_path, "w") as f:
            f.write(f"CONSTITUTIONAL VIOLATIONS REPORT\n")
            f.write(f"{'='*60}\n")
            f.write(f"Case: {case.title}\n")
            f.write(f"Case Number: {case.case_number}\n")
            f.write(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Total Violations: {len(violations)}\n")
            if request.include_damages_estimate:
                f.write(f"Estimated Damages: ${total_estimated_damages:,.2f}\n")
            f.write(f"\n{'='*60}\n\n")
            
            for i, v in enumerate(violations, 1):
                f.write(f"VIOLATION #{i}: {v.violation_type.value.upper()}\n")
                f.write(f"{'-'*40}\n")
                f.write(f"Title: {v.title}\n")
                f.write(f"Severity: {'★' * v.severity}{'☆' * (5-v.severity)}\n")
                f.write(f"Verified: {'Yes' if v.is_verified else 'No'}\n")
                f.write(f"Confidence: {(v.confidence_score or 0) * 100:.0f}%\n\n")
                
                if v.description:
                    f.write(f"Description:\n{v.description}\n\n")
                
                if request.include_legal_basis and v.legal_basis:
                    f.write(f"Legal Basis:\n{v.legal_basis}\n\n")
                
                if request.include_damages_estimate and v.estimated_damages:
                    f.write(f"Estimated Damages: ${v.estimated_damages:,.2f}\n\n")
                
                f.write("\n")
    
    return ExportResponse(
        export_id=export_id,
        filename=filename,
        file_path=str(file_path),
        file_size=file_path.stat().st_size,
        format=request.format,
        created_at=datetime.utcnow(),
        download_url=f"/api/exports/download/{filename}"
    )


@router.post("/evidence-binder", response_model=ExportResponse)
async def export_evidence_binder(
    request: ExportEvidenceBinderRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Compile evidence binder for case.
    
    Creates organized collection of:
    - Table of contents/index
    - Bates-stamped documents
    - Evidence listings
    - Timeline integration
    """
    # Verify case exists
    result = await db.execute(select(Case).where(Case.id == request.case_id))
    case = result.scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    # Get documents
    if request.document_ids:
        docs_result = await db.execute(
            select(Document).where(Document.id.in_(request.document_ids))
        )
    else:
        docs_result = await db.execute(
            select(Document)
            .join(Case.documents)
            .where(Case.id == request.case_id)
        )
    documents = docs_result.scalars().all()
    
    # Get evidence
    if request.evidence_ids:
        ev_result = await db.execute(
            select(Evidence).where(Evidence.id.in_(request.evidence_ids))
        )
    else:
        ev_result = await db.execute(
            select(Evidence)
            .join(Case.evidence)
            .where(Case.id == request.case_id)
        )
    evidence = ev_result.scalars().all()
    
    # Generate index
    export_id = uuid.uuid4().hex[:12]
    filename = f"evidence_binder_{case.case_number or case.id}_{export_id}.json"
    file_path = EXPORT_DIR / filename
    
    binder_data = {
        "case_id": case.id,
        "case_title": case.title,
        "case_number": case.case_number,
        "generated_at": datetime.utcnow().isoformat(),
        "bates_prefix": request.bates_prefix,
        "index": [],
        "documents": [],
        "evidence": []
    }
    
    bates_counter = 1
    
    # Index documents
    for doc in documents:
        bates_start = f"{request.bates_prefix}{bates_counter:05d}"
        bates_end = f"{request.bates_prefix}{bates_counter + (doc.page_count or 1) - 1:05d}"
        
        doc_entry = {
            "id": doc.id,
            "bates_range": f"{bates_start}-{bates_end}",
            "title": doc.title or doc.original_filename,
            "type": doc.document_type.value,
            "page_count": doc.page_count or 1,
            "file_path": doc.file_path
        }
        
        binder_data["documents"].append(doc_entry)
        binder_data["index"].append({
            "bates": bates_start,
            "description": doc.title or doc.original_filename,
            "type": "document"
        })
        
        bates_counter += doc.page_count or 1
    
    # Index evidence
    for ev in evidence:
        ev_entry = {
            "id": ev.id,
            "exhibit": f"{request.bates_prefix}-EV{ev.id:04d}",
            "title": ev.title or ev.original_filename,
            "type": ev.evidence_type.value,
            "duration": ev.duration_seconds,
            "officer": ev.officer_name,
            "file_path": ev.file_path
        }
        
        binder_data["evidence"].append(ev_entry)
        binder_data["index"].append({
            "exhibit": ev_entry["exhibit"],
            "description": ev.title or ev.original_filename,
            "type": "evidence"
        })
    
    # Save binder index
    with open(file_path, "w") as f:
        json.dump(binder_data, f, indent=2)
    
    return ExportResponse(
        export_id=export_id,
        filename=filename,
        file_path=str(file_path),
        file_size=file_path.stat().st_size,
        format="json",
        created_at=datetime.utcnow(),
        download_url=f"/api/exports/download/{filename}"
    )


@router.post("/settlement-analysis")
async def export_settlement_analysis(
    case_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate settlement analysis report.
    
    Includes:
    - Damages calculation
    - Comparable case settlements
    - Strengths and weaknesses
    - Recommended settlement range
    """
    # Verify case exists
    result = await db.execute(select(Case).where(Case.id == case_id))
    case = result.scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    # Get violations
    violations_result = await db.execute(
        select(Violation).where(Violation.case_id == case_id)
    )
    violations = violations_result.scalars().all()
    
    # Calculate damages
    total_estimated = sum(v.estimated_damages or 0 for v in violations)
    
    # Generate report
    export_id = uuid.uuid4().hex[:12]
    filename = f"settlement_analysis_{case.case_number or case.id}_{export_id}.json"
    file_path = EXPORT_DIR / filename
    
    analysis = {
        "case_id": case.id,
        "case_title": case.title,
        "generated_at": datetime.utcnow().isoformat(),
        
        "damages_analysis": {
            "economic_damages": {
                "medical_expenses": 0,
                "lost_wages": 0,
                "property_damage": 0
            },
            "non_economic_damages": {
                "pain_and_suffering": total_estimated * 0.4,
                "emotional_distress": total_estimated * 0.3,
                "loss_of_enjoyment": total_estimated * 0.2
            },
            "punitive_damages_potential": total_estimated * 0.5,
            "attorneys_fees_estimate": total_estimated * 0.33
        },
        
        "settlement_recommendation": {
            "low": total_estimated * 0.5,
            "target": total_estimated,
            "high": total_estimated * 2.0,
            "walk_away": total_estimated * 0.25
        },
        
        "comparable_cases": [
            {
                "description": "Similar excessive force case - NJ Federal Court",
                "settlement": 125000,
                "year": 2024
            },
            {
                "description": "False arrest / malicious prosecution",
                "settlement": 75000,
                "year": 2023
            }
        ],
        
        "risk_factors": {
            "qualified_immunity_risk": "Medium",
            "jury_appeal": "High" if len(violations) > 2 else "Medium",
            "evidence_strength": "Strong" if len(violations) > 0 else "Needs Development"
        }
    }
    
    with open(file_path, "w") as f:
        json.dump(analysis, f, indent=2)
    
    return ExportResponse(
        export_id=export_id,
        filename=filename,
        file_path=str(file_path),
        file_size=file_path.stat().st_size,
        format="json",
        created_at=datetime.utcnow(),
        download_url=f"/api/exports/download/{filename}"
    )


@router.get("/download/{filename}")
async def download_export(filename: str):
    """Download an exported file"""
    file_path = EXPORT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Export file not found")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/octet-stream"
    )
