"""
BarberX Legal Case Management Pro Suite
Premium Legal Analysis API - Discovery, Deposition, Strategy, Deadlines
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.database import get_db
from app.db.models import Case
from app.services.discovery_service import discovery_service, PrivilegeType, ReviewStatus
from app.services.deposition_service import deposition_service, WitnessType, CredibilityFactor
from app.services.case_strategy_service import case_strategy_service, RiskLevel
from app.services.deadline_service import deadline_calculator
from app.services.brady_service import brady_service, EvidenceClassification, ViolationType

router = APIRouter()


# ============================================================
# SCHEMAS
# ============================================================

# Discovery Schemas
class PrivilegeAnalysisRequest(BaseModel):
    document_text: str
    author: str
    recipients: List[str]
    date: str
    subject: str


class DocumentReviewRequest(BaseModel):
    document_text: str
    case_issues: List[str]
    search_terms: List[str]


class ProductionSetRequest(BaseModel):
    name: str
    bates_prefix: str
    produced_to: str
    confidentiality_designation: str = "standard"


class BatesAssignmentRequest(BaseModel):
    document_id: str
    page_count: int
    prefix: str = "DEF"


# Witness Schemas
class WitnessCreateRequest(BaseModel):
    name: str
    witness_type: str
    relationship: str = ""
    favorable_to: str = ""
    phone: str = ""
    email: str = ""
    employer: str = ""


class CredibilityAssessmentRequest(BaseModel):
    witness_id: str
    case_context: str
    statements: List[Dict[str, Any]]


class DepositionDigestRequest(BaseModel):
    transcript_text: str
    witness_name: str
    deposition_date: str
    case_issues: List[str]


class CrossExamRequest(BaseModel):
    witness_id: str
    case_theory: str
    goals: List[str]


# Expert Witness Schemas
class ExpertWitnessRequest(BaseModel):
    name: str
    specialty: str
    retained_by: str
    education: List[str] = []
    certifications: List[str] = []


class DaubertAnalysisRequest(BaseModel):
    expert_id: str
    expert_report: str
    methodology_description: str


# Strategy Schemas
class LiabilityAssessmentRequest(BaseModel):
    case_facts: str
    defendant_name: str
    claim_type: str
    jurisdiction: str
    applicable_law: str


class DamagesCalculationRequest(BaseModel):
    case_id: str
    economic_items: Dict[str, float]
    injury_description: str
    plaintiff_age: int
    jurisdiction: str
    comparable_verdicts: List[Dict[str, Any]] = []


class SettlementAnalysisRequest(BaseModel):
    case_id: str
    liability_assessment_id: str
    damages_analysis_id: str
    litigation_costs: float
    policy_limits: float
    time_to_trial_months: int


class CaseTheoryRequest(BaseModel):
    case_id: str
    party: str
    case_facts: str
    claims: List[str]
    key_evidence: List[str]


class JuryProfileRequest(BaseModel):
    case_id: str
    case_type: str
    case_theory_id: str
    venue: str


# Deadline Schemas
class DeadlineCalculationRequest(BaseModel):
    case_id: str
    rule_id: str
    trigger_date: str
    service_method: str = "personal"
    description: str = ""


class LitigationTimelineRequest(BaseModel):
    case_id: str
    jurisdiction: str
    complaint_filed_date: str
    service_date: str
    trial_date: Optional[str] = None


# Brady Schemas
class BradyClassificationRequest(BaseModel):
    evidence_description: str
    case_facts: str
    defendant_theory: str


class BradyEvidenceRequest(BaseModel):
    case_id: str
    classification: str
    description: str
    source_type: str
    source_description: str
    significance: str = ""


class OfficerRecordRequest(BaseModel):
    officer_name: str
    badge_number: str
    department: str


class DisciplinaryRecordRequest(BaseModel):
    record_id: str
    complaint_date: str
    complaint_type: str
    allegation: str
    finding: str
    discipline_imposed: str = ""


class BradyViolationRequest(BaseModel):
    case_id: str
    violation_type: str
    evidence_id: str
    responsible_party: str
    when_discovered: str
    how_discovered: str
    description: str = ""
    responsible_office: str = ""
    when_should_have_been_disclosed: str = ""


# ============================================================
# DISCOVERY ENDPOINTS
# ============================================================

@router.post("/discovery/privilege/analyze")
async def analyze_privilege(request: PrivilegeAnalysisRequest):
    """
    AI-powered privilege analysis.
    
    Analyzes documents for:
    - Attorney-client privilege
    - Work product doctrine
    - Joint defense privilege
    - Common interest doctrine
    """
    metadata = {
        "author": request.author,
        "recipients": request.recipients,
        "date": request.date,
        "subject": request.subject
    }
    
    result = await discovery_service.analyze_privilege(
        document_text=request.document_text,
        document_metadata=metadata
    )
    
    return result


@router.post("/discovery/review/ai-assist")
async def ai_assisted_review(request: DocumentReviewRequest):
    """
    AI-assisted document review.
    
    Analyzes for:
    - Responsiveness to case issues
    - Key concepts and entities
    - Hot document indicators
    """
    result = await discovery_service.ai_assisted_review(
        document_text=request.document_text,
        case_issues=request.case_issues,
        search_terms=request.search_terms
    )
    
    return result


@router.post("/discovery/bates/assign")
async def assign_bates_numbers(request: BatesAssignmentRequest):
    """Assign Bates numbers to document"""
    bates = discovery_service.assign_bates_numbers(
        document_id=request.document_id,
        page_count=request.page_count,
        prefix=request.prefix
    )
    
    return {
        "document_id": request.document_id,
        "bates_start": bates.start_label,
        "bates_end": bates.end_label,
        "page_count": request.page_count
    }


@router.post("/discovery/production/create")
async def create_production_set(request: ProductionSetRequest):
    """Create new production set"""
    production = discovery_service.create_production_set(
        name=request.name,
        bates_prefix=request.bates_prefix,
        produced_to=request.produced_to,
        confidentiality_designation=request.confidentiality_designation
    )
    
    return asdict(production)


@router.get("/discovery/privilege-log/export")
async def export_privilege_log(format: str = Query("csv")):
    """Export privilege log"""
    content = discovery_service.export_privilege_log(format)
    return {"format": format, "content": content}


# ============================================================
# WITNESS & DEPOSITION ENDPOINTS
# ============================================================

@router.post("/witness/create")
async def create_witness(request: WitnessCreateRequest):
    """Create new witness profile"""
    witness = deposition_service.create_witness(
        name=request.name,
        witness_type=WitnessType(request.witness_type),
        relationship=request.relationship,
        favorable_to=request.favorable_to,
        phone=request.phone,
        email=request.email,
        employer=request.employer
    )
    
    return asdict(witness)


@router.get("/witness/{witness_id}")
async def get_witness(witness_id: str):
    """Get witness profile"""
    witness = deposition_service.witnesses.get(witness_id)
    if not witness:
        raise HTTPException(status_code=404, detail="Witness not found")
    return asdict(witness)


@router.post("/witness/credibility/assess")
async def assess_credibility(request: CredibilityAssessmentRequest):
    """
    AI-powered witness credibility assessment.
    
    Analyzes:
    - Internal consistency
    - Bias indicators
    - Memory/perception issues
    - Impeachment opportunities
    """
    result = await deposition_service.assess_credibility(
        witness_id=request.witness_id,
        all_statements=request.statements,
        case_context=request.case_context
    )
    
    return result


@router.post("/deposition/digest")
async def generate_deposition_digest(request: DepositionDigestRequest):
    """
    Generate comprehensive deposition digest.
    
    Creates:
    - Executive summary
    - Key admissions/denials
    - Impeachment material
    - Follow-up questions
    """
    digest = await deposition_service.generate_deposition_digest(
        transcript_text=request.transcript_text,
        witness_name=request.witness_name,
        deposition_date=request.deposition_date,
        case_issues=request.case_issues
    )
    
    return asdict(digest)


@router.post("/witness/impeachment/find")
async def find_impeachment_opportunities(
    witness_id: str,
    deposition_text: str,
    prior_statements: List[Dict[str, Any]]
):
    """Find impeachment opportunities by comparing statements"""
    result = await deposition_service.find_impeachment_opportunities(
        witness_id=witness_id,
        deposition_text=deposition_text,
        prior_statements=prior_statements
    )
    
    return result


@router.post("/witness/cross-examination/outline")
async def generate_cross_outline(request: CrossExamRequest):
    """Generate cross-examination outline"""
    # Get witness and deposition digest
    witness = deposition_service.witnesses.get(request.witness_id)
    if not witness:
        raise HTTPException(status_code=404, detail="Witness not found")
    
    # Find deposition digest for this witness
    digest = None
    for d in deposition_service.depositions.values():
        if d.witness_name == witness.name:
            digest = d
            break
    
    if not digest:
        raise HTTPException(status_code=404, detail="No deposition digest found")
    
    result = await deposition_service.generate_cross_examination_outline(
        witness_id=request.witness_id,
        deposition_digest=digest,
        case_theory=request.case_theory,
        goals=request.goals
    )
    
    return result


# ============================================================
# EXPERT WITNESS ENDPOINTS
# ============================================================

@router.post("/expert/create")
async def create_expert_witness(request: ExpertWitnessRequest):
    """Create expert witness profile"""
    expert = deposition_service.create_expert_witness(
        name=request.name,
        specialty=request.specialty,
        retained_by=request.retained_by,
        education=request.education,
        certifications=request.certifications
    )
    
    return asdict(expert)


@router.post("/expert/daubert/analyze")
async def analyze_daubert_factors(request: DaubertAnalysisRequest):
    """
    Analyze expert testimony under Daubert factors.
    
    Evaluates:
    - Testability
    - Peer review
    - Error rate
    - Standards
    - General acceptance
    """
    result = await deposition_service.analyze_daubert_factors(
        expert_id=request.expert_id,
        expert_report=request.expert_report,
        methodology_description=request.methodology_description
    )
    
    return result


# ============================================================
# CASE STRATEGY ENDPOINTS
# ============================================================

@router.post("/strategy/liability/assess")
async def assess_liability(request: LiabilityAssessmentRequest):
    """
    AI-powered liability assessment.
    
    Analyzes:
    - Elements of each claim
    - Available defenses
    - Qualified immunity
    - Comparative fault
    """
    assessment = await case_strategy_service.assess_liability(
        case_facts=request.case_facts,
        defendant_name=request.defendant_name,
        claim_type=request.claim_type,
        jurisdiction=request.jurisdiction,
        applicable_law=request.applicable_law
    )
    
    return asdict(assessment)


@router.post("/strategy/damages/calculate")
async def calculate_damages(request: DamagesCalculationRequest):
    """
    Comprehensive damages calculation.
    
    Calculates:
    - Economic damages
    - Non-economic damages
    - Punitive damages potential
    - Applicable caps
    """
    analysis = await case_strategy_service.calculate_damages(
        case_id=request.case_id,
        economic_items=request.economic_items,
        injury_description=request.injury_description,
        plaintiff_age=request.plaintiff_age,
        jurisdiction=request.jurisdiction,
        comparable_verdicts=request.comparable_verdicts
    )
    
    return asdict(analysis)


@router.post("/strategy/settlement/analyze")
async def analyze_settlement(request: SettlementAnalysisRequest):
    """Calculate settlement value and negotiation strategy"""
    liability = case_strategy_service.liability_assessments.get(request.liability_assessment_id)
    damages = case_strategy_service.damages_analyses.get(request.damages_analysis_id)
    
    if not liability:
        raise HTTPException(status_code=404, detail="Liability assessment not found")
    if not damages:
        raise HTTPException(status_code=404, detail="Damages analysis not found")
    
    analysis = await case_strategy_service.analyze_settlement(
        case_id=request.case_id,
        liability_assessment=liability,
        damages_analysis=damages,
        litigation_costs=request.litigation_costs,
        policy_limits=request.policy_limits,
        time_to_trial_months=request.time_to_trial_months
    )
    
    return asdict(analysis)


@router.post("/strategy/theory/develop")
async def develop_case_theory(request: CaseTheoryRequest):
    """Develop comprehensive case theory"""
    theory = await case_strategy_service.develop_case_theory(
        case_id=request.case_id,
        party=request.party,
        case_facts=request.case_facts,
        claims=request.claims,
        key_evidence=request.key_evidence
    )
    
    return asdict(theory)


@router.post("/strategy/jury/profile")
async def develop_jury_profile(request: JuryProfileRequest):
    """Develop ideal juror profile for voir dire"""
    theory = case_strategy_service.case_theories.get(request.case_theory_id)
    if not theory:
        raise HTTPException(status_code=404, detail="Case theory not found")
    
    profile = await case_strategy_service.develop_jury_profile(
        case_id=request.case_id,
        case_type=request.case_type,
        case_theory=theory,
        venue=request.venue
    )
    
    return asdict(profile)


# ============================================================
# DEADLINE ENDPOINTS
# ============================================================

@router.post("/deadline/calculate")
async def calculate_deadline(request: DeadlineCalculationRequest):
    """Calculate deadline based on court rules"""
    deadline = deadline_calculator.calculate_deadline(
        case_id=request.case_id,
        rule_id=request.rule_id,
        trigger_date=request.trigger_date,
        service_method=request.service_method,
        description=request.description
    )
    
    return asdict(deadline)


@router.post("/deadline/timeline/create")
async def create_litigation_timeline(request: LitigationTimelineRequest):
    """Create complete litigation timeline with all key deadlines"""
    chain = deadline_calculator.create_litigation_timeline(
        case_id=request.case_id,
        jurisdiction=request.jurisdiction,
        complaint_filed_date=request.complaint_filed_date,
        service_date=request.service_date,
        trial_date=request.trial_date
    )
    
    return {
        "chain_id": chain.chain_id,
        "chain_name": chain.chain_name,
        "deadlines": [asdict(dl) for dl in chain.deadlines]
    }


@router.get("/deadline/upcoming")
async def get_upcoming_deadlines(
    case_id: Optional[str] = None,
    days_ahead: int = Query(30, ge=1, le=365)
):
    """Get upcoming deadlines"""
    deadlines = deadline_calculator.get_upcoming_deadlines(
        case_id=case_id,
        days_ahead=days_ahead
    )
    
    return {
        "count": len(deadlines),
        "deadlines": [asdict(dl) for dl in deadlines]
    }


@router.get("/deadline/rules")
async def get_available_rules(jurisdiction: Optional[str] = None):
    """Get available court rules"""
    rules = deadline_calculator.rules.values()
    
    if jurisdiction:
        rules = [r for r in rules if r.jurisdiction == jurisdiction]
    
    return {
        "count": len(list(rules)),
        "rules": [
            {
                "rule_id": r.rule_id,
                "jurisdiction": r.jurisdiction,
                "rule_number": r.rule_number,
                "rule_title": r.rule_title,
                "days": r.days
            }
            for r in rules
        ]
    }


# ============================================================
# BRADY/GIGLIO ENDPOINTS
# ============================================================

@router.post("/brady/classify")
async def classify_brady_evidence(request: BradyClassificationRequest):
    """
    AI-powered Brady/Giglio classification.
    
    Classifies evidence for:
    - Direct exculpatory value
    - Impeachment potential
    - Witness credibility impact
    """
    result = await brady_service.classify_evidence(
        evidence_description=request.evidence_description,
        case_facts=request.case_facts,
        defendant_theory=request.defendant_theory
    )
    
    return result


@router.post("/brady/evidence/create")
async def create_brady_evidence(request: BradyEvidenceRequest):
    """Create Brady evidence record"""
    evidence = brady_service.create_brady_evidence(
        case_id=request.case_id,
        classification=EvidenceClassification(request.classification),
        description=request.description,
        source_type=request.source_type,
        source_description=request.source_description,
        significance=request.significance
    )
    
    return asdict(evidence)


@router.post("/brady/officer/create")
async def create_officer_record(request: OfficerRecordRequest):
    """Create officer credibility record for Giglio tracking"""
    record = brady_service.create_officer_record(
        officer_name=request.officer_name,
        badge_number=request.badge_number,
        department=request.department
    )
    
    return asdict(record)


@router.post("/brady/officer/discipline")
async def add_disciplinary_record(request: DisciplinaryRecordRequest):
    """Add disciplinary record to officer"""
    brady_service.add_disciplinary_record(
        record_id=request.record_id,
        complaint_date=request.complaint_date,
        complaint_type=request.complaint_type,
        allegation=request.allegation,
        finding=request.finding,
        discipline_imposed=request.discipline_imposed
    )
    
    record = brady_service.officer_records.get(request.record_id)
    return asdict(record) if record else {"error": "Record not found"}


@router.get("/brady/officer/{record_id}/giglio")
async def analyze_officer_giglio(record_id: str):
    """Generate comprehensive Giglio analysis for officer"""
    result = await brady_service.analyze_officer_giglio(record_id)
    return result


@router.post("/brady/violation/document")
async def document_brady_violation(request: BradyViolationRequest):
    """Document a Brady violation"""
    violation = brady_service.document_violation(
        case_id=request.case_id,
        violation_type=ViolationType(request.violation_type),
        evidence_id=request.evidence_id,
        responsible_party=request.responsible_party,
        when_discovered=request.when_discovered,
        how_discovered=request.how_discovered,
        description=request.description,
        responsible_office=request.responsible_office,
        when_should_have_been_disclosed=request.when_should_have_been_disclosed
    )
    
    return asdict(violation)


@router.post("/brady/violation/{violation_id}/motion")
async def generate_brady_motion(
    violation_id: str,
    case_caption: str,
    court: str
):
    """Generate Brady motion based on documented violation"""
    motion_text = await brady_service.generate_brady_motion(
        violation_id=violation_id,
        case_caption=case_caption,
        court=court
    )
    
    return {"violation_id": violation_id, "motion": motion_text}


@router.get("/brady/report/{case_id}")
async def get_brady_report(case_id: str):
    """Generate comprehensive Brady/Giglio report for case"""
    report = brady_service.generate_brady_report(case_id)
    return report


# ============================================================
# SERVICE STATUS
# ============================================================

@router.get("/status")
async def get_premium_services_status():
    """Get status of all premium legal analysis services"""
    return {
        "available": True,
        "services": {
            "discovery": {
                "status": "active",
                "features": [
                    "AI Privilege Analysis",
                    "Bates Numbering",
                    "Privilege Log Generation",
                    "AI Document Review",
                    "Production Set Management",
                    "Load File Generation (Concordance, Relativity)"
                ]
            },
            "deposition": {
                "status": "active",
                "features": [
                    "Witness Management",
                    "AI Credibility Assessment",
                    "Deposition Digest Generation",
                    "Impeachment Tracking",
                    "Cross-Examination Outlines",
                    "Expert Witness Daubert Analysis"
                ]
            },
            "case_strategy": {
                "status": "active",
                "features": [
                    "AI Liability Assessment",
                    "Damages Calculation",
                    "Settlement Valuation",
                    "Case Theory Development",
                    "Jury Selection Guidance",
                    "Verdict Prediction"
                ]
            },
            "deadlines": {
                "status": "active",
                "features": [
                    "FRCP Deadline Calculation",
                    "NJ State Rules",
                    "Business Day Handling",
                    "Service Extensions",
                    "Holiday Calendar",
                    "Deadline Chains"
                ]
            },
            "brady_giglio": {
                "status": "active",
                "features": [
                    "AI Evidence Classification",
                    "Officer Credibility Tracking",
                    "Giglio Material Database",
                    "Violation Documentation",
                    "Brady Motion Generation",
                    "Comprehensive Reporting"
                ]
            }
        }
    }
