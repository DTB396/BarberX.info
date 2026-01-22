"""
BarberX Legal Case Management Pro Suite
Firm Management API - Conflicts, Billing, Research

Premium API endpoints for firm management features:
- Conflict checking
- Legal billing & time tracking
- Legal research integration
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import asdict
from decimal import Decimal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.conflict_service import (
    conflict_service, 
    EntityType, 
    ConflictSeverity
)
from app.services.billing_service import (
    billing_service,
    BillingStatus,
    ExpenseType,
    TaskCode
)
from app.services.research_service import (
    research_service,
    JurisdictionType,
    CaseStatus,
    ResearchType
)

router = APIRouter()


# ============================================================
# SCHEMAS
# ============================================================

# Conflict Schemas
class EntityCreateRequest(BaseModel):
    name: str
    entity_type: str
    aliases: List[str] = []
    corporate_family: List[str] = []
    identifiers: Dict[str, str] = {}


class MatterCreateRequest(BaseModel):
    matter_number: str
    matter_name: str
    client_id: str
    matter_type: str
    status: str = "active"
    adverse_parties: List[str] = []


class ConflictCheckRequest(BaseModel):
    new_matter_name: str
    prospective_client_name: str
    adverse_party_names: List[str]
    related_party_names: List[str] = []
    matter_type: str = ""
    requested_by: str = ""
    use_ai_analysis: bool = True


class WaiverRecordRequest(BaseModel):
    conflict_check_id: str
    hit_id: str
    waiving_client_id: str
    waiver_type: str
    scope: str
    obtained_by: str
    waiver_letter_path: str = ""
    conditions: List[str] = []


class WaiverLetterRequest(BaseModel):
    conflict_check_id: str
    hit_id: str
    client_name: str
    attorney_name: str
    firm_name: str


# Billing Schemas
class TimekeeperRequest(BaseModel):
    name: str
    initials: str
    role: str
    standard_rate: float
    bar_number: str = ""
    email: str = ""


class TimeEntryRequest(BaseModel):
    timekeeper_id: str
    matter_id: str
    date: str
    hours: float
    narrative: str
    task_code: str = ""
    activity_code: str = ""
    client_id: str = ""


class ExpenseRequest(BaseModel):
    matter_id: str
    date: str
    expense_type: str
    description: str
    amount: float
    vendor: str = ""
    receipt_path: str = ""
    markup_rate: float = 0


class BillingGuidelineRequest(BaseModel):
    client_id: str
    client_name: str
    partner_rate_cap: Optional[float] = None
    associate_rate_cap: Optional[float] = None
    paralegal_rate_cap: Optional[float] = None
    block_billing_allowed: bool = False
    minimum_increment: float = 0.1
    prohibited_phrases: List[str] = []


class InvoiceGenerateRequest(BaseModel):
    matter_id: str
    client_id: str
    billing_period_start: str
    billing_period_end: str
    payment_terms_days: int = 30


class BudgetCreateRequest(BaseModel):
    matter_id: str
    total_budget: float
    phases: Dict[str, float] = {}


class TimeAdjustRequest(BaseModel):
    entry_id: str
    new_hours: float
    reason: str


# Research Schemas
class CitationCreateRequest(BaseModel):
    case_name: str
    full_citation: str
    reporter: str
    volume: str
    page: str
    year: int
    court: str
    jurisdiction: str
    holding: str = ""
    key_quotes: List[str] = []


class ResearchMemoRequest(BaseModel):
    matter_id: str
    author: str
    question: str
    facts: str
    jurisdiction: str = "federal"


class IssueCreateRequest(BaseModel):
    matter_id: str
    issue_description: str
    elements: List[str] = []


class IssueAuthorityRequest(BaseModel):
    issue_id: str
    citation_id: str
    favorable: bool
    distinguishing_factors: List[str] = []


class MotionTemplateRequest(BaseModel):
    motion_type: str
    title: str
    jurisdiction: str
    court: str
    template_text: str
    sample_arguments: List[str] = []
    key_cases: List[str] = []


class MotionOutcomeRequest(BaseModel):
    motion_id: str
    outcome: str
    court: str
    judge: str
    notes: str = ""


# ============================================================
# CONFLICT CHECKING ENDPOINTS
# ============================================================

@router.post("/conflicts/entity")
async def create_entity(request: EntityCreateRequest):
    """Add entity to conflict database"""
    entity = conflict_service.add_entity(
        name=request.name,
        entity_type=EntityType(request.entity_type),
        aliases=request.aliases,
        corporate_family=request.corporate_family,
        identifiers=request.identifiers
    )
    
    return asdict(entity)


@router.post("/conflicts/matter")
async def create_matter(request: MatterCreateRequest):
    """Add matter to conflict database"""
    matter = conflict_service.add_matter(
        matter_number=request.matter_number,
        matter_name=request.matter_name,
        client_id=request.client_id,
        matter_type=request.matter_type,
        status=request.status,
        adverse_parties=request.adverse_parties
    )
    
    return asdict(matter)


@router.post("/conflicts/check")
async def run_conflict_check(request: ConflictCheckRequest):
    """
    Run comprehensive conflict of interest check.
    
    Checks for:
    - Direct adverse conflicts with current clients
    - Former client conflicts (Rule 1.9)
    - Corporate family conflicts
    - Related party conflicts
    - AI-powered relationship detection
    """
    check = await conflict_service.run_conflict_check(
        new_matter_name=request.new_matter_name,
        prospective_client_name=request.prospective_client_name,
        adverse_party_names=request.adverse_party_names,
        related_party_names=request.related_party_names,
        matter_type=request.matter_type,
        requested_by=request.requested_by,
        use_ai_analysis=request.use_ai_analysis
    )
    
    return conflict_service.generate_conflict_report(check.check_id)


@router.get("/conflicts/check/{check_id}")
async def get_conflict_check(check_id: str):
    """Get conflict check results"""
    return conflict_service.generate_conflict_report(check_id)


@router.post("/conflicts/waiver")
async def record_waiver(request: WaiverRecordRequest):
    """Record conflict waiver"""
    waiver = conflict_service.record_waiver(
        conflict_check_id=request.conflict_check_id,
        hit_id=request.hit_id,
        waiving_client_id=request.waiving_client_id,
        waiver_type=request.waiver_type,
        scope=request.scope,
        obtained_by=request.obtained_by,
        waiver_letter_path=request.waiver_letter_path,
        conditions=request.conditions
    )
    
    return asdict(waiver)


@router.post("/conflicts/waiver-letter")
async def generate_waiver_letter(request: WaiverLetterRequest):
    """Generate conflict waiver letter"""
    letter = conflict_service.generate_waiver_letter(
        conflict_check_id=request.conflict_check_id,
        hit_id=request.hit_id,
        client_name=request.client_name,
        attorney_name=request.attorney_name,
        firm_name=request.firm_name
    )
    
    return {"letter": letter}


@router.get("/conflicts/search")
async def search_entities(
    name: str,
    include_aliases: bool = Query(True),
    include_corporate: bool = Query(True)
):
    """Search for matching entities"""
    matches = conflict_service.find_matching_entities(
        search_name=name,
        include_aliases=include_aliases,
        include_corporate_family=include_corporate
    )
    
    return {
        "query": name,
        "matches": [
            {
                "entity_id": entity.entity_id,
                "name": entity.name,
                "type": entity.entity_type.value,
                "similarity": similarity,
                "match_type": match_type
            }
            for entity, similarity, match_type in matches
        ]
    }


# ============================================================
# BILLING ENDPOINTS
# ============================================================

@router.post("/billing/timekeeper")
async def create_timekeeper(request: TimekeeperRequest):
    """Add timekeeper to billing system"""
    timekeeper = billing_service.add_timekeeper(
        name=request.name,
        initials=request.initials,
        role=request.role,
        standard_rate=request.standard_rate,
        bar_number=request.bar_number,
        email=request.email
    )
    
    return asdict(timekeeper)


@router.post("/billing/time")
async def record_time_entry(request: TimeEntryRequest):
    """Record time entry with automatic compliance checking"""
    entry = billing_service.record_time(
        timekeeper_id=request.timekeeper_id,
        matter_id=request.matter_id,
        date=request.date,
        hours=request.hours,
        narrative=request.narrative,
        task_code=request.task_code,
        activity_code=request.activity_code,
        client_id=request.client_id
    )
    
    result = asdict(entry)
    # Convert Decimal to float for JSON serialization
    result["hours"] = float(entry.hours)
    result["rate"] = float(entry.rate) if entry.rate else None
    result["amount"] = float(entry.amount) if entry.amount else None
    
    return result


@router.post("/billing/time/{entry_id}/review")
async def review_time_narrative(entry_id: str, client_name: str = ""):
    """AI-powered narrative review for billing compliance"""
    result = await billing_service.review_narrative_ai(entry_id, client_name)
    return result


@router.post("/billing/time/adjust")
async def adjust_time_entry(request: TimeAdjustRequest):
    """Adjust hours on time entry"""
    billing_service.adjust_time(
        entry_id=request.entry_id,
        new_hours=request.new_hours,
        reason=request.reason
    )
    
    entry = billing_service.time_entries.get(request.entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    
    result = asdict(entry)
    result["hours"] = float(entry.hours)
    result["original_hours"] = float(entry.original_hours) if entry.original_hours else None
    result["rate"] = float(entry.rate) if entry.rate else None
    result["amount"] = float(entry.amount) if entry.amount else None
    
    return result


@router.post("/billing/expense")
async def record_expense(request: ExpenseRequest):
    """Record billable expense"""
    expense = billing_service.record_expense(
        matter_id=request.matter_id,
        date=request.date,
        expense_type=request.expense_type,
        description=request.description,
        amount=request.amount,
        vendor=request.vendor,
        receipt_path=request.receipt_path,
        markup_rate=request.markup_rate
    )
    
    result = asdict(expense)
    result["amount"] = float(expense.amount)
    result["markup_rate"] = float(expense.markup_rate)
    result["billed_amount"] = float(expense.billed_amount) if expense.billed_amount else None
    
    return result


@router.post("/billing/guideline")
async def create_billing_guideline(request: BillingGuidelineRequest):
    """Create client billing guideline"""
    guideline = billing_service.create_billing_guideline(
        client_id=request.client_id,
        client_name=request.client_name,
        partner_rate_cap=request.partner_rate_cap,
        associate_rate_cap=request.associate_rate_cap,
        paralegal_rate_cap=request.paralegal_rate_cap,
        block_billing_allowed=request.block_billing_allowed,
        minimum_increment=request.minimum_increment,
        prohibited_phrases=request.prohibited_phrases
    )
    
    return {
        "guideline_id": guideline.guideline_id,
        "client_id": guideline.client_id,
        "client_name": guideline.client_name
    }


@router.post("/billing/invoice/generate")
async def generate_invoice(request: InvoiceGenerateRequest):
    """Generate invoice from unbilled time and expenses"""
    invoice = billing_service.generate_invoice(
        matter_id=request.matter_id,
        client_id=request.client_id,
        billing_period_start=request.billing_period_start,
        billing_period_end=request.billing_period_end,
        payment_terms_days=request.payment_terms_days
    )
    
    return {
        "invoice_id": invoice.invoice_id,
        "invoice_number": invoice.invoice_number,
        "total_fees": float(invoice.total_fees),
        "total_expenses": float(invoice.total_expenses),
        "total_due": float(invoice.total_due),
        "due_date": invoice.due_date,
        "entries_count": len(invoice.time_entries),
        "expenses_count": len(invoice.expenses)
    }


@router.get("/billing/invoice/{invoice_id}/ledes")
async def export_ledes(invoice_id: str):
    """Export invoice in LEDES 1998B format"""
    content = billing_service.export_ledes(invoice_id)
    if not content:
        raise HTTPException(status_code=404, detail="Invoice not found")
    
    return {"invoice_id": invoice_id, "format": "LEDES1998B", "content": content}


@router.post("/billing/budget")
async def create_budget(request: BudgetCreateRequest):
    """Create matter budget"""
    budget = billing_service.create_budget(
        matter_id=request.matter_id,
        total_budget=request.total_budget,
        phases=request.phases
    )
    
    return {
        "budget_id": budget.budget_id,
        "matter_id": budget.matter_id,
        "total_budget": float(budget.total_budget)
    }


@router.get("/billing/budget/{matter_id}")
async def get_budget_status(matter_id: str):
    """Get budget status with alerts"""
    result = billing_service.update_budget_tracking(matter_id)
    if not result:
        raise HTTPException(status_code=404, detail="Budget not found for matter")
    return result


@router.get("/billing/matter/{matter_id}/summary")
async def get_matter_billing_summary(matter_id: str):
    """Get comprehensive billing summary for matter"""
    return billing_service.get_matter_summary(matter_id)


@router.get("/billing/task-codes")
async def get_task_codes():
    """Get available UTBMS task codes"""
    return {
        "codes": [
            {"code": code.name, "description": code.value}
            for code in TaskCode
        ]
    }


# ============================================================
# RESEARCH ENDPOINTS
# ============================================================

@router.post("/research/citation")
async def add_citation(request: CitationCreateRequest):
    """Add citation to research database"""
    citation = research_service.add_citation(
        case_name=request.case_name,
        full_citation=request.full_citation,
        reporter=request.reporter,
        volume=request.volume,
        page=request.page,
        year=request.year,
        court=request.court,
        jurisdiction=request.jurisdiction,
        holding=request.holding,
        key_quotes=request.key_quotes
    )
    
    return asdict(citation)


@router.post("/research/citation/parse")
async def parse_citation(citation_string: str):
    """Parse citation string into components"""
    result = research_service.parse_citation(citation_string)
    if result:
        return result
    raise HTTPException(status_code=400, detail="Unable to parse citation")


@router.get("/research/citation/{citation_id}/status")
async def check_citation_status(citation_id: str):
    """Check current status of citation (Shepardize)"""
    result = await research_service.check_citation_status(citation_id)
    return result


@router.post("/research/memo")
async def create_research_memo(request: ResearchMemoRequest):
    """Create research memo with AI assistance"""
    memo = await research_service.generate_research_memo(
        matter_id=request.matter_id,
        author=request.author,
        question=request.question,
        facts=request.facts,
        jurisdiction=request.jurisdiction
    )
    
    return asdict(memo)


@router.get("/research/memo/{memo_id}")
async def get_research_memo(memo_id: str):
    """Get research memo"""
    memo = research_service.memos.get(memo_id)
    if not memo:
        raise HTTPException(status_code=404, detail="Memo not found")
    return asdict(memo)


@router.post("/research/issue")
async def create_issue(request: IssueCreateRequest):
    """Create key legal issue to track"""
    issue = research_service.create_issue(
        matter_id=request.matter_id,
        issue_description=request.issue_description,
        elements=request.elements
    )
    
    return asdict(issue)


@router.post("/research/issue/authority")
async def add_authority_to_issue(request: IssueAuthorityRequest):
    """Add authority to issue tracking"""
    research_service.add_authority_to_issue(
        issue_id=request.issue_id,
        citation_id=request.citation_id,
        favorable=request.favorable,
        distinguishing_factors=request.distinguishing_factors
    )
    
    issue = research_service.issues.get(request.issue_id)
    if not issue:
        raise HTTPException(status_code=404, detail="Issue not found")
    
    return asdict(issue)


@router.get("/research/issue/{issue_id}/analyze")
async def analyze_issue_strength(issue_id: str):
    """Analyze strength of legal position on issue"""
    result = await research_service.analyze_issue_strength(issue_id)
    return result


@router.get("/research/issue/{issue_id}/authorities")
async def get_authority_table(issue_id: str):
    """Generate table of authorities for issue"""
    table = research_service.generate_authority_table(issue_id)
    return {"table": table}


@router.post("/research/motion")
async def add_motion_template(request: MotionTemplateRequest):
    """Add motion template to bank"""
    motion = research_service.add_motion_template(
        motion_type=request.motion_type,
        title=request.title,
        jurisdiction=request.jurisdiction,
        court=request.court,
        template_text=request.template_text,
        sample_arguments=request.sample_arguments,
        key_cases=request.key_cases
    )
    
    return asdict(motion)


@router.get("/research/motions")
async def get_motion_templates(
    motion_type: Optional[str] = None,
    jurisdiction: Optional[str] = None
):
    """Get motion templates"""
    motions = research_service.get_motion_templates(motion_type, jurisdiction)
    return {
        "count": len(motions),
        "motions": [asdict(m) for m in motions]
    }


@router.post("/research/motion/outcome")
async def record_motion_outcome(request: MotionOutcomeRequest):
    """Record motion outcome for analytics"""
    research_service.record_motion_outcome(
        motion_id=request.motion_id,
        outcome=request.outcome,
        court=request.court,
        judge=request.judge,
        notes=request.notes
    )
    
    motion = research_service.motions.get(request.motion_id)
    if motion:
        return {
            "motion_id": motion.motion_id,
            "times_used": motion.times_used,
            "success_rate": motion.success_rate
        }
    raise HTTPException(status_code=404, detail="Motion not found")


@router.get("/research/search")
async def search_citations(
    query: str,
    jurisdiction: Optional[str] = None,
    min_year: Optional[int] = None
):
    """Search citation database"""
    results = research_service.search_citations(
        query=query,
        jurisdiction=jurisdiction,
        min_year=min_year
    )
    
    return {
        "query": query,
        "count": len(results),
        "citations": [
            {
                "citation_id": c.citation_id,
                "case_name": c.case_name,
                "citation": c.full_citation,
                "year": c.year,
                "holding": c.holding[:200] + "..." if len(c.holding) > 200 else c.holding,
                "status": c.status.value
            }
            for c in results
        ]
    }


@router.get("/research/key-authorities")
async def get_key_authorities():
    """Get pre-loaded key civil rights authorities"""
    return {
        "authorities": [
            {
                "citation_id": c.citation_id,
                "case_name": c.case_name,
                "citation": c.full_citation,
                "year": c.year,
                "holding": c.holding,
                "key_quotes": c.key_quotes,
                "status": c.status.value
            }
            for c in research_service.citations.values()
        ]
    }


# ============================================================
# SERVICE STATUS
# ============================================================

@router.get("/status")
async def get_firm_services_status():
    """Get status of firm management services"""
    return {
        "available": True,
        "services": {
            "conflict_checking": {
                "status": "active",
                "features": [
                    "Entity name matching (fuzzy)",
                    "Corporate family tracking",
                    "Direct adverse conflict detection",
                    "Former client conflict detection",
                    "AI relationship analysis",
                    "Waiver management",
                    "ABA Rule compliance"
                ],
                "entities_count": len(conflict_service.entities),
                "matters_count": len(conflict_service.matters),
                "checks_performed": len(conflict_service.conflict_checks)
            },
            "billing": {
                "status": "active",
                "features": [
                    "Time entry with UTBMS codes",
                    "Billing guideline compliance",
                    "AI narrative review",
                    "Block billing detection",
                    "Expense tracking",
                    "Invoice generation",
                    "LEDES 1998B export",
                    "Budget tracking with alerts"
                ],
                "timekeepers_count": len(billing_service.timekeepers),
                "entries_count": len(billing_service.time_entries),
                "invoices_count": len(billing_service.invoices)
            },
            "research": {
                "status": "active",
                "features": [
                    "Citation database",
                    "Citation status checking",
                    "Research memo generation",
                    "Issue tracking",
                    "Authority analysis",
                    "Motion bank",
                    "AI research assistance",
                    "Key civil rights authorities pre-loaded"
                ],
                "citations_count": len(research_service.citations),
                "memos_count": len(research_service.memos),
                "issues_count": len(research_service.issues)
            }
        }
    }
