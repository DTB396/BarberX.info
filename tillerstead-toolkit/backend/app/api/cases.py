"""
BarberX Legal Case Management Pro Suite
Cases API Router - Case CRUD Operations
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
from sqlalchemy.orm import selectinload

from app.db.database import get_db
from app.db.models import Case, Party, Document, Evidence, Violation, CaseStatus, CaseType
from app.schemas.legal_schemas import (
    CaseCreate, CaseUpdate, CaseResponse, CaseListResponse,
    PartyCreate, PartyResponse, CaseTimelineResponse
)

router = APIRouter()


@router.get("/", response_model=CaseListResponse)
async def list_cases(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    status: Optional[CaseStatus] = None,
    case_type: Optional[CaseType] = None,
    search: Optional[str] = None,
    is_active: bool = True,
    db: AsyncSession = Depends(get_db)
):
    """List all cases with pagination and filtering"""
    query = select(Case).where(Case.is_active == is_active)
    
    if status:
        query = query.where(Case.status == status)
    if case_type:
        query = query.where(Case.case_type == case_type)
    if search:
        search_filter = or_(
            Case.title.ilike(f"%{search}%"),
            Case.case_number.ilike(f"%{search}%"),
            Case.docket_number.ilike(f"%{search}%"),
            Case.summary.ilike(f"%{search}%")
        )
        query = query.where(search_filter)
    
    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar()
    
    # Paginate
    query = query.order_by(Case.priority.asc(), Case.updated_at.desc())
    query = query.offset((page - 1) * per_page).limit(per_page)
    
    result = await db.execute(query)
    cases = result.scalars().all()
    
    # Enhance with counts
    case_responses = []
    for case in cases:
        case_dict = CaseResponse.model_validate(case).model_dump()
        
        # Get counts
        doc_count = await db.execute(
            select(func.count()).select_from(Document)
            .join(Case.documents).where(Case.id == case.id)
        )
        evidence_count = await db.execute(
            select(func.count()).select_from(Evidence)
            .join(Case.evidence).where(Case.id == case.id)
        )
        violation_count = await db.execute(
            select(func.count()).select_from(Violation)
            .where(Violation.case_id == case.id)
        )
        
        case_dict['document_count'] = doc_count.scalar() or 0
        case_dict['evidence_count'] = evidence_count.scalar() or 0
        case_dict['violation_count'] = violation_count.scalar() or 0
        
        case_responses.append(CaseResponse(**case_dict))
    
    return CaseListResponse(
        items=case_responses,
        total=total,
        page=page,
        per_page=per_page,
        pages=(total + per_page - 1) // per_page
    )


@router.post("/", response_model=CaseResponse)
async def create_case(
    case_data: CaseCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new case"""
    # Generate internal reference
    count_result = await db.execute(select(func.count()).select_from(Case))
    count = count_result.scalar() or 0
    internal_ref = f"BX-{count + 1:06d}"
    
    case = Case(
        **case_data.model_dump(),
        internal_reference=internal_ref
    )
    
    db.add(case)
    await db.commit()
    await db.refresh(case)
    
    return CaseResponse.model_validate(case)


@router.get("/{case_id}", response_model=CaseResponse)
async def get_case(
    case_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific case by ID"""
    result = await db.execute(
        select(Case)
        .options(
            selectinload(Case.parties),
            selectinload(Case.violations)
        )
        .where(Case.id == case_id)
    )
    case = result.scalar_one_or_none()
    
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    return CaseResponse.model_validate(case)


@router.put("/{case_id}", response_model=CaseResponse)
async def update_case(
    case_id: int,
    case_data: CaseUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update a case"""
    result = await db.execute(select(Case).where(Case.id == case_id))
    case = result.scalar_one_or_none()
    
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    update_data = case_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(case, field, value)
    
    await db.commit()
    await db.refresh(case)
    
    return CaseResponse.model_validate(case)


@router.delete("/{case_id}")
async def archive_case(
    case_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Archive a case (soft delete)"""
    result = await db.execute(select(Case).where(Case.id == case_id))
    case = result.scalar_one_or_none()
    
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    case.is_archived = True
    case.is_active = False
    await db.commit()
    
    return {"message": "Case archived", "case_id": case_id}


# ============================================================
# PARTIES
# ============================================================

@router.get("/{case_id}/parties", response_model=List[PartyResponse])
async def list_case_parties(
    case_id: int,
    db: AsyncSession = Depends(get_db)
):
    """List all parties for a case"""
    result = await db.execute(
        select(Party).where(Party.case_id == case_id)
    )
    parties = result.scalars().all()
    return [PartyResponse.model_validate(p) for p in parties]


@router.post("/{case_id}/parties", response_model=PartyResponse)
async def add_party(
    case_id: int,
    party_data: PartyCreate,
    db: AsyncSession = Depends(get_db)
):
    """Add a party to a case"""
    # Verify case exists
    case_result = await db.execute(select(Case).where(Case.id == case_id))
    if not case_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Case not found")
    
    party = Party(**party_data.model_dump())
    party.case_id = case_id
    
    db.add(party)
    await db.commit()
    await db.refresh(party)
    
    return PartyResponse.model_validate(party)


# ============================================================
# TIMELINE
# ============================================================

@router.get("/{case_id}/timeline", response_model=CaseTimelineResponse)
async def get_case_timeline(
    case_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get the complete timeline for a case"""
    from app.db.models import TimelineEvent
    
    result = await db.execute(
        select(Case).where(Case.id == case_id)
    )
    case = result.scalar_one_or_none()
    
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    events_result = await db.execute(
        select(TimelineEvent)
        .where(TimelineEvent.case_id == case_id)
        .order_by(TimelineEvent.event_date.asc())
    )
    events = events_result.scalars().all()
    
    from app.schemas.legal_schemas import TimelineEventResponse
    
    event_responses = [TimelineEventResponse.model_validate(e) for e in events]
    
    date_range_start = min((e.event_date for e in events), default=None)
    date_range_end = max((e.event_date for e in events), default=None)
    
    return CaseTimelineResponse(
        case_id=case_id,
        case_title=case.title,
        events=event_responses,
        total_events=len(events),
        date_range_start=date_range_start,
        date_range_end=date_range_end
    )


# ============================================================
# CASE IMPORT (from existing BarberX structure)
# ============================================================

@router.post("/import/barberx")
async def import_from_barberx_structure(
    folder_path: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Import a case from BarberX folder structure.
    Expects structure like: .bwc/25-41706 Barber, Devon/
    """
    import os
    import re
    from pathlib import Path
    
    path = Path(folder_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Folder not found")
    
    # Parse folder name for case info
    folder_name = path.name
    
    # Try to extract case number and name
    # Format: "25-41706 Barber, Devon" -> case_number: 25-41706, name: Barber, Devon
    match = re.match(r'^([\d-]+)\s+(.+)$', folder_name)
    
    if match:
        case_number = match.group(1)
        case_name = match.group(2)
    else:
        case_number = None
        case_name = folder_name
    
    # Create case
    case = Case(
        title=f"{case_name} v. [Defendants TBD]",
        short_title=case_name,
        case_number=case_number,
        case_type=CaseType.CIVIL_RIGHTS,
        status=CaseStatus.INTAKE,
        internal_reference=f"BX-{case_number}" if case_number else None
    )
    
    db.add(case)
    await db.commit()
    await db.refresh(case)
    
    # Scan for files
    files_found = {
        'pdfs': [],
        'videos': [],
        'images': [],
        'other': []
    }
    
    for file_path in path.iterdir():
        if file_path.is_file():
            ext = file_path.suffix.lower()
            if ext == '.pdf':
                files_found['pdfs'].append(str(file_path))
            elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
                files_found['videos'].append(str(file_path))
            elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                files_found['images'].append(str(file_path))
            else:
                files_found['other'].append(str(file_path))
    
    return {
        "message": "Case imported successfully",
        "case_id": case.id,
        "case_number": case.case_number,
        "title": case.title,
        "files_detected": {
            "pdfs": len(files_found['pdfs']),
            "videos": len(files_found['videos']),
            "images": len(files_found['images']),
            "other": len(files_found['other'])
        },
        "next_steps": [
            f"POST /api/documents/upload with PDFs to process {len(files_found['pdfs'])} documents",
            f"POST /api/evidence/bwc/upload with videos to process {len(files_found['videos'])} BWC files"
        ]
    }
