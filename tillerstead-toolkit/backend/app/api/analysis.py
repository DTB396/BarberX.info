"""
BarberX Legal Case Management Pro Suite
Analysis API Router - Constitutional Violation Scanning, Liability Assessment
"""
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.db.database import get_db
from app.db.models import (
    Case, Document, Evidence, Violation, Officer, AnalysisJob,
    ViolationType, ProcessingStatus
)
from app.schemas.legal_schemas import (
    ViolationCreate, ViolationResponse,
    ConstitutionalAnalysisRequest, ConstitutionalAnalysisResponse,
    LiabilityAssessmentRequest, LiabilityAssessmentResponse,
    OfficerHistoryRequest, OfficerHistoryResponse
)

router = APIRouter()


# ============================================================
# CONSTITUTIONAL VIOLATION PATTERNS
# ============================================================

VIOLATION_KEYWORDS = {
    ViolationType.UNLAWFUL_SEARCH: [
        "without warrant", "warrantless search", "no consent",
        "exceeded scope", "pretextual", "fruit of poisonous tree"
    ],
    ViolationType.UNLAWFUL_SEIZURE: [
        "detained without", "no reasonable suspicion", "prolonged stop",
        "without probable cause", "unlawful arrest"
    ],
    ViolationType.EXCESSIVE_FORCE: [
        "struck", "punched", "tased", "shot", "choked",
        "knee on neck", "excessive", "unreasonable force",
        "continued after", "handcuffed and", "compliant"
    ],
    ViolationType.MIRANDA_VIOLATION: [
        "custodial interrogation", "not advised", "no miranda",
        "continued questioning", "after invoked"
    ],
    ViolationType.RIGHT_TO_COUNSEL: [
        "requested attorney", "asked for lawyer", "invoked counsel",
        "continued interrogation after"
    ],
    ViolationType.BRADY_VIOLATION: [
        "withheld", "concealed evidence", "exculpatory",
        "failed to disclose", "suppressed"
    ],
    ViolationType.EQUAL_PROTECTION: [
        "racial profiling", "discriminatory", "selective enforcement",
        "treated differently"
    ],
    ViolationType.DUE_PROCESS: [
        "denied hearing", "no notice", "procedural violation"
    ]
}

LEGAL_CITATIONS = {
    ViolationType.UNLAWFUL_SEARCH: [
        "Mapp v. Ohio, 367 U.S. 643 (1961)",
        "Katz v. United States, 389 U.S. 347 (1967)",
        "Terry v. Ohio, 392 U.S. 1 (1968)"
    ],
    ViolationType.EXCESSIVE_FORCE: [
        "Graham v. Connor, 490 U.S. 386 (1989)",
        "Tennessee v. Garner, 471 U.S. 1 (1985)",
        "Kingsley v. Hendrickson, 576 U.S. 389 (2015)"
    ],
    ViolationType.MIRANDA_VIOLATION: [
        "Miranda v. Arizona, 384 U.S. 436 (1966)",
        "Berghuis v. Thompkins, 560 U.S. 370 (2010)"
    ],
    ViolationType.RIGHT_TO_COUNSEL: [
        "Gideon v. Wainwright, 372 U.S. 335 (1963)",
        "Edwards v. Arizona, 451 U.S. 477 (1981)"
    ],
    ViolationType.BRADY_VIOLATION: [
        "Brady v. Maryland, 373 U.S. 83 (1963)",
        "Giglio v. United States, 405 U.S. 150 (1972)"
    ]
}

DAMAGES_ESTIMATES = {
    ViolationType.EXCESSIVE_FORCE: (25000, 150000, 500000),
    ViolationType.UNLAWFUL_SEARCH: (10000, 50000, 150000),
    ViolationType.UNLAWFUL_SEIZURE: (15000, 75000, 250000),
    ViolationType.MIRANDA_VIOLATION: (5000, 25000, 75000),
    ViolationType.BRADY_VIOLATION: (50000, 200000, 1000000),
    ViolationType.EQUAL_PROTECTION: (25000, 100000, 500000),
}


def scan_text_for_violations(text: str) -> List[dict]:
    """Scan text for constitutional violation indicators"""
    violations_found = []
    text_lower = text.lower()
    
    for violation_type, keywords in VIOLATION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                # Find context around the keyword
                idx = text_lower.find(keyword)
                start = max(0, idx - 100)
                end = min(len(text), idx + len(keyword) + 100)
                context = text[start:end]
                
                violations_found.append({
                    "type": violation_type,
                    "keyword": keyword,
                    "context": context,
                    "confidence": 0.7 + (0.1 if len(keyword) > 15 else 0)
                })
    
    return violations_found


@router.post("/constitutional", response_model=ConstitutionalAnalysisResponse)
async def run_constitutional_analysis(
    request: ConstitutionalAnalysisRequest,
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Run constitutional violation analysis on case materials.
    
    Scans documents and evidence for potential violations:
    - 4th Amendment: Search & seizure, excessive force
    - 5th Amendment: Miranda, self-incrimination, due process
    - 6th Amendment: Right to counsel
    - 14th Amendment: Equal protection, due process
    - Brady/Giglio violations
    """
    start_time = datetime.utcnow()
    
    # Verify case exists
    result = await db.execute(select(Case).where(Case.id == request.case_id))
    case = result.scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    violations_found = []
    
    # Analyze documents
    if request.document_ids:
        for doc_id in request.document_ids:
            doc_result = await db.execute(select(Document).where(Document.id == doc_id))
            document = doc_result.scalar_one_or_none()
            
            if document and document.extracted_text:
                text_violations = scan_text_for_violations(document.extracted_text)
                
                for v in text_violations:
                    # Check if we should scan this type
                    if request.scan_types and v["type"] not in request.scan_types:
                        continue
                    
                    violation = Violation(
                        case_id=request.case_id,
                        violation_type=v["type"],
                        title=f"{v['type'].value.replace('_', ' ').title()} - {v['keyword']}",
                        description=v["context"],
                        legal_basis="\n".join(LEGAL_CITATIONS.get(v["type"], [])),
                        source_document_id=doc_id,
                        confidence_score=v["confidence"],
                        severity=3,
                        supporting_citations=LEGAL_CITATIONS.get(v["type"], [])
                    )
                    
                    db.add(violation)
                    await db.commit()
                    await db.refresh(violation)
                    
                    violations_found.append(ViolationResponse.model_validate(violation))
    
    # Analyze evidence transcripts
    if request.evidence_ids:
        for eid in request.evidence_ids:
            ev_result = await db.execute(select(Evidence).where(Evidence.id == eid))
            evidence = ev_result.scalar_one_or_none()
            
            if evidence and evidence.transcript:
                text_violations = scan_text_for_violations(evidence.transcript)
                
                for v in text_violations:
                    if request.scan_types and v["type"] not in request.scan_types:
                        continue
                    
                    violation = Violation(
                        case_id=request.case_id,
                        violation_type=v["type"],
                        title=f"{v['type'].value.replace('_', ' ').title()} - BWC Evidence",
                        description=v["context"],
                        legal_basis="\n".join(LEGAL_CITATIONS.get(v["type"], [])),
                        source_evidence_id=eid,
                        confidence_score=v["confidence"],
                        severity=3,
                        supporting_citations=LEGAL_CITATIONS.get(v["type"], [])
                    )
                    
                    db.add(violation)
                    await db.commit()
                    await db.refresh(violation)
                    
                    violations_found.append(ViolationResponse.model_validate(violation))
    
    # Calculate metrics
    severity_breakdown = {}
    type_breakdown = {}
    total_estimated_damages = 0
    
    for v in violations_found:
        # Severity
        sev_key = f"severity_{v.severity}"
        severity_breakdown[sev_key] = severity_breakdown.get(sev_key, 0) + 1
        
        # Type
        type_breakdown[v.violation_type.value] = type_breakdown.get(v.violation_type.value, 0) + 1
        
        # Damages
        if v.violation_type in DAMAGES_ESTIMATES:
            _, mid, _ = DAMAGES_ESTIMATES[v.violation_type]
            total_estimated_damages += mid
    
    # Overall liability score
    liability_score = min(100, len(violations_found) * 15 + (
        sum(DAMAGES_ESTIMATES.get(v.violation_type, (0, 0, 0))[1] for v in violations_found) / 10000
    ))
    
    # Generate recommendations
    recommendations = []
    if ViolationType.EXCESSIVE_FORCE.value in type_breakdown:
        recommendations.append("Document all injuries with medical records and photographs")
        recommendations.append("Request all use-of-force reports from the incident")
    if ViolationType.BRADY_VIOLATION.value in type_breakdown:
        recommendations.append("File motion to compel disclosure of exculpatory evidence")
    if ViolationType.MIRANDA_VIOLATION.value in type_breakdown:
        recommendations.append("Move to suppress statements made without proper warnings")
    if request.include_suggestions:
        recommendations.append("Request complete personnel files for involved officers")
        recommendations.append("OPRA request for similar complaints against same officers")
    
    processing_time = (datetime.utcnow() - start_time).total_seconds()
    
    # Create analysis job record
    job = AnalysisJob(
        job_type="constitutional_analysis",
        status=ProcessingStatus.COMPLETED,
        case_id=request.case_id,
        results={
            "violations_count": len(violations_found),
            "liability_score": liability_score
        },
        completed_at=datetime.utcnow()
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    
    return ConstitutionalAnalysisResponse(
        case_id=request.case_id,
        analysis_id=job.id,
        violations_found=violations_found,
        total_violations=len(violations_found),
        severity_breakdown=severity_breakdown,
        type_breakdown=type_breakdown,
        liability_score=liability_score,
        estimated_total_damages=total_estimated_damages,
        recommendations=recommendations,
        processing_time_seconds=processing_time
    )


@router.post("/liability", response_model=LiabilityAssessmentResponse)
async def assess_liability(
    request: LiabilityAssessmentRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Comprehensive liability assessment for a case.
    
    Evaluates:
    - Constitutional violations identified
    - Strength of evidence
    - Officer history and prior incidents
    - Department patterns
    - Comparable case settlements
    """
    # Verify case exists
    result = await db.execute(select(Case).where(Case.id == request.case_id))
    case = result.scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    
    # Get violations for this case
    violations_result = await db.execute(
        select(Violation).where(Violation.case_id == request.case_id)
    )
    violations = violations_result.scalars().all()
    
    # Calculate scores
    constitutional_score = min(100, len(violations) * 20)
    
    # Evidence strength (based on what we have)
    doc_count = await db.execute(
        select(func.count()).select_from(Document)
        .join(Case.documents).where(Case.id == request.case_id)
    )
    evidence_count = await db.execute(
        select(func.count()).select_from(Evidence)
        .join(Case.evidence).where(Case.id == request.case_id)
    )
    evidence_strength = min(100, (doc_count.scalar() or 0) * 5 + (evidence_count.scalar() or 0) * 15)
    
    # Officer history score (placeholder - would query actual officer history)
    officer_score = 50  # Default middle score
    
    # Department pattern score (placeholder)
    department_score = 40
    
    # Overall score
    overall_score = (
        constitutional_score * 0.35 +
        evidence_strength * 0.30 +
        officer_score * 0.20 +
        department_score * 0.15
    )
    
    # Damages calculation
    base_damages = sum(
        DAMAGES_ESTIMATES.get(v.violation_type, (10000, 50000, 150000))[1]
        for v in violations
    )
    
    damages_low = base_damages * 0.4
    damages_mid = base_damages
    damages_high = base_damages * 2.5
    
    # Comparable settlements (would query actual database)
    comparable_settlements = [
        {"case": "Similar excessive force case", "amount": 75000, "year": 2024},
        {"case": "False arrest settlement", "amount": 50000, "year": 2023},
    ]
    
    # Strengths and weaknesses
    strengths = []
    weaknesses = []
    recommendations = []
    
    if len(violations) > 2:
        strengths.append(f"Multiple constitutional violations identified ({len(violations)})")
    if evidence_count.scalar() > 0:
        strengths.append("BWC footage available as objective evidence")
    
    if len(violations) == 0:
        weaknesses.append("No constitutional violations formally identified yet")
        recommendations.append("Run constitutional analysis on all documents and evidence")
    if evidence_count.scalar() == 0:
        weaknesses.append("No video evidence uploaded")
        recommendations.append("Obtain all available BWC and surveillance footage")
    
    recommendations.append("Depose all involved officers")
    recommendations.append("Request complete discovery including internal affairs files")
    
    return LiabilityAssessmentResponse(
        case_id=request.case_id,
        overall_liability_score=overall_score,
        constitutional_violations_score=constitutional_score,
        evidence_strength_score=evidence_strength,
        officer_history_score=officer_score,
        department_pattern_score=department_score,
        estimated_damages_low=damages_low,
        estimated_damages_mid=damages_mid,
        estimated_damages_high=damages_high,
        comparable_settlements=comparable_settlements,
        strengths=strengths,
        weaknesses=weaknesses,
        recommendations=recommendations
    )


@router.post("/officer/{badge_number}", response_model=OfficerHistoryResponse)
async def get_officer_history(
    badge_number: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get officer history across all cases.
    
    Aggregates:
    - Total incidents
    - Complaints (sustained and unsustained)
    - Lawsuits and settlements
    - Violation patterns
    """
    # Find officer records
    result = await db.execute(
        select(Officer).where(Officer.badge_number == badge_number)
    )
    officer = result.scalar_one_or_none()
    
    if not officer:
        # Create basic record from case data
        return OfficerHistoryResponse(
            officer_name="Unknown",
            badge_number=badge_number,
            department="Unknown",
            total_incidents=0,
            total_complaints=0,
            sustained_complaints=0,
            total_lawsuits=0,
            total_settlements=0,
            violation_patterns={},
            risk_score=50,
            related_case_ids=[],
            incident_timeline=[]
        )
    
    return OfficerHistoryResponse(
        officer_name=officer.name,
        badge_number=officer.badge_number,
        department=officer.department or "Unknown",
        total_incidents=officer.total_complaints,
        total_complaints=officer.total_complaints,
        sustained_complaints=officer.sustained_complaints,
        total_lawsuits=officer.civil_suits,
        total_settlements=officer.settlements_total,
        violation_patterns={},  # Would aggregate from violations
        risk_score=min(100, officer.total_complaints * 10 + officer.sustained_complaints * 20),
        related_case_ids=officer.related_case_ids or [],
        incident_timeline=officer.prior_incidents or []
    )


@router.get("/{case_id}/violations", response_model=List[ViolationResponse])
async def list_case_violations(
    case_id: int,
    violation_type: Optional[ViolationType] = None,
    is_verified: Optional[bool] = None,
    min_severity: Optional[int] = None,
    db: AsyncSession = Depends(get_db)
):
    """List all violations for a case"""
    query = select(Violation).where(Violation.case_id == case_id)
    
    if violation_type:
        query = query.where(Violation.violation_type == violation_type)
    if is_verified is not None:
        query = query.where(Violation.is_verified == is_verified)
    if min_severity:
        query = query.where(Violation.severity >= min_severity)
    
    query = query.order_by(Violation.severity.desc(), Violation.created_at.desc())
    
    result = await db.execute(query)
    violations = result.scalars().all()
    
    return [ViolationResponse.model_validate(v) for v in violations]


@router.post("/violations", response_model=ViolationResponse)
async def create_violation(
    violation_data: ViolationCreate,
    db: AsyncSession = Depends(get_db)
):
    """Manually create a violation record"""
    # Verify case exists
    case_result = await db.execute(select(Case).where(Case.id == violation_data.case_id))
    if not case_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Case not found")
    
    violation = Violation(
        **violation_data.model_dump(),
        is_verified=False  # Manual entries start unverified
    )
    
    db.add(violation)
    await db.commit()
    await db.refresh(violation)
    
    return ViolationResponse.model_validate(violation)


@router.put("/violations/{violation_id}/verify")
async def verify_violation(
    violation_id: int,
    verified_by: str,
    db: AsyncSession = Depends(get_db)
):
    """Mark a violation as verified by human review"""
    result = await db.execute(select(Violation).where(Violation.id == violation_id))
    violation = result.scalar_one_or_none()
    
    if not violation:
        raise HTTPException(status_code=404, detail="Violation not found")
    
    violation.is_verified = True
    violation.verified_by = verified_by
    violation.verified_at = datetime.utcnow()
    
    await db.commit()
    
    return {
        "message": "Violation verified",
        "violation_id": violation_id,
        "verified_by": verified_by,
        "verified_at": violation.verified_at
    }


@router.post("/pattern")
async def analyze_cross_case_patterns(
    officer_name: Optional[str] = None,
    department: Optional[str] = None,
    violation_type: Optional[ViolationType] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze patterns across multiple cases.
    
    Identifies:
    - Repeat offender officers
    - Department-wide issues
    - Common violation patterns
    """
    query = select(Violation)
    
    if violation_type:
        query = query.where(Violation.violation_type == violation_type)
    
    result = await db.execute(query)
    violations = result.scalars().all()
    
    # Aggregate patterns
    by_type = {}
    by_case = {}
    
    for v in violations:
        # By type
        type_key = v.violation_type.value
        by_type[type_key] = by_type.get(type_key, 0) + 1
        
        # By case
        by_case[v.case_id] = by_case.get(v.case_id, 0) + 1
    
    return {
        "total_violations_analyzed": len(violations),
        "violations_by_type": by_type,
        "cases_with_violations": len(by_case),
        "average_violations_per_case": len(violations) / max(1, len(by_case)),
        "most_common_violation": max(by_type.items(), key=lambda x: x[1])[0] if by_type else None
    }
