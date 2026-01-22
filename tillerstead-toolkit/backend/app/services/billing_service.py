"""
BarberX Legal Case Management Pro Suite
Legal Billing & Time Tracking Service

Premium billing features including time entry, expense tracking,
billing guidelines compliance, and invoice generation.
"""
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from decimal import Decimal, ROUND_HALF_UP
import json

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class BillingStatus(Enum):
    """Time entry billing status"""
    DRAFT = "draft"
    READY = "ready"
    BILLED = "billed"
    COLLECTED = "collected"
    WRITTEN_OFF = "written_off"
    NON_BILLABLE = "non_billable"


class ExpenseType(Enum):
    """Types of billable expenses"""
    FILING_FEE = "filing_fee"
    SERVICE_OF_PROCESS = "service_of_process"
    COURT_REPORTER = "court_reporter"
    EXPERT_WITNESS = "expert_witness"
    TRAVEL = "travel"
    COPYING = "copying"
    POSTAGE = "postage"
    RESEARCH = "research"
    OVERNIGHT_DELIVERY = "overnight_delivery"
    DEPOSITION = "deposition"
    MEDIATION = "mediation"
    TRANSCRIPT = "transcript"
    TECHNOLOGY = "technology"
    OTHER = "other"


class TaskCode(Enum):
    """UTBMS/LEDES task codes for billing"""
    # Litigation
    L100 = "Case Assessment/Development"
    L110 = "Fact Investigation/Development"
    L120 = "Analysis/Strategy"
    L130 = "Experts/Consultants"
    L140 = "Document/File Management"
    L150 = "Pleadings"
    L160 = "Motions Practice"
    L170 = "Discovery"
    L180 = "Depositions"
    L190 = "Trials/Arbitrations/Hearings"
    L200 = "Appeal"
    L210 = "Mediation/Settlement"
    
    # Activity codes
    A101 = "Plan/outline"
    A102 = "Research"
    A103 = "Draft/revise"
    A104 = "Review/analyze"
    A105 = "Communicate"
    A106 = "Conference/meeting"
    A107 = "Travel"


@dataclass
class Timekeeper:
    """Attorney or staff member who records time"""
    timekeeper_id: str
    name: str
    initials: str
    role: str  # Partner, Associate, Paralegal, etc.
    standard_rate: Decimal
    bar_number: str = ""
    email: str = ""
    department: str = ""
    
    # Different rates for different clients
    client_rates: Dict[str, Decimal] = field(default_factory=dict)


@dataclass
class TimeEntry:
    """Individual time entry"""
    entry_id: str
    timekeeper_id: str
    matter_id: str
    date: str
    hours: Decimal
    narrative: str
    task_code: str = ""
    activity_code: str = ""
    status: BillingStatus = BillingStatus.DRAFT
    rate: Optional[Decimal] = None
    amount: Optional[Decimal] = None
    
    # Billing guideline compliance
    block_billed: bool = False
    block_bill_warning: str = ""
    narrative_compliant: bool = True
    narrative_issues: List[str] = field(default_factory=list)
    
    # Adjustments
    original_hours: Optional[Decimal] = None
    adjustment_reason: str = ""
    
    # Metadata
    created_date: str = ""
    modified_date: str = ""
    billed_date: str = ""
    invoice_id: str = ""


@dataclass
class Expense:
    """Billable expense entry"""
    expense_id: str
    matter_id: str
    date: str
    expense_type: ExpenseType
    description: str
    amount: Decimal
    vendor: str = ""
    receipt_path: str = ""
    status: BillingStatus = BillingStatus.DRAFT
    reimbursable: bool = True
    markup_rate: Decimal = Decimal("0")
    billed_amount: Optional[Decimal] = None
    invoice_id: str = ""


@dataclass
class BillingGuideline:
    """Client billing guidelines"""
    guideline_id: str
    client_id: str
    client_name: str
    
    # Rate caps
    partner_rate_cap: Optional[Decimal] = None
    associate_rate_cap: Optional[Decimal] = None
    paralegal_rate_cap: Optional[Decimal] = None
    
    # Time entry rules
    minimum_increment: Decimal = Decimal("0.1")  # 6-minute increments
    maximum_hours_per_day: Decimal = Decimal("10")
    block_billing_allowed: bool = False
    require_task_codes: bool = True
    require_activity_codes: bool = True
    
    # Narrative rules
    minimum_narrative_length: int = 10
    prohibited_phrases: List[str] = field(default_factory=list)
    vague_term_warnings: List[str] = field(default_factory=list)
    
    # Expense rules
    require_receipts_over: Optional[Decimal] = None
    prohibited_expenses: List[str] = field(default_factory=list)
    travel_requires_approval: bool = True
    markup_allowed: bool = False
    
    # Staffing rules
    approved_timekeepers: List[str] = field(default_factory=list)
    staffing_restrictions: str = ""


@dataclass
class Invoice:
    """Generated invoice"""
    invoice_id: str
    invoice_number: str
    matter_id: str
    client_id: str
    
    # Dates
    invoice_date: str
    billing_period_start: str
    billing_period_end: str
    due_date: str
    
    # Amounts
    total_fees: Decimal = Decimal("0")
    total_expenses: Decimal = Decimal("0")
    subtotal: Decimal = Decimal("0")
    adjustments: Decimal = Decimal("0")
    taxes: Decimal = Decimal("0")
    total_due: Decimal = Decimal("0")
    
    # Collections
    amount_paid: Decimal = Decimal("0")
    balance_due: Decimal = Decimal("0")
    status: str = "pending"  # pending, sent, paid, overdue, disputed
    
    # Content
    time_entries: List[str] = field(default_factory=list)  # Entry IDs
    expenses: List[str] = field(default_factory=list)  # Expense IDs
    
    # LEDES export
    ledes_file_path: str = ""


@dataclass
class BudgetTracker:
    """Matter budget tracking"""
    budget_id: str
    matter_id: str
    total_budget: Decimal
    phases: Dict[str, Decimal] = field(default_factory=dict)  # Phase: Amount
    
    # Tracking
    total_billed: Decimal = Decimal("0")
    total_wip: Decimal = Decimal("0")
    remaining: Decimal = Decimal("0")
    percent_used: Decimal = Decimal("0")
    
    # Alerts
    alert_thresholds: List[int] = field(default_factory=lambda: [50, 75, 90, 100])
    alerts_sent: List[str] = field(default_factory=list)


class BillingService:
    """
    Comprehensive legal billing service.
    
    Features:
    - Time entry with UTBMS/LEDES codes
    - Billing guideline compliance checking
    - AI-powered narrative review
    - Expense tracking
    - Invoice generation
    - Budget monitoring
    - LEDES export
    """
    
    def __init__(self):
        self.timekeepers: Dict[str, Timekeeper] = {}
        self.time_entries: Dict[str, TimeEntry] = {}
        self.expenses: Dict[str, Expense] = {}
        self.guidelines: Dict[str, BillingGuideline] = {}
        self.invoices: Dict[str, Invoice] = {}
        self.budgets: Dict[str, BudgetTracker] = {}
        self.client = AsyncOpenAI() if OPENAI_AVAILABLE else None
        
        # Default vague phrases to flag
        self.default_vague_phrases = [
            "reviewed file", "attention to file", "work on case",
            "research", "various", "miscellaneous", "administrative",
            "correspondence", "telephone call", "conference"
        ]
        
        # Initialize invoice counter
        self._invoice_counter = 1000
    
    # ========================================
    # TIMEKEEPER MANAGEMENT
    # ========================================
    
    def add_timekeeper(
        self,
        name: str,
        initials: str,
        role: str,
        standard_rate: float,
        **kwargs
    ) -> Timekeeper:
        """Add timekeeper to billing system"""
        timekeeper_id = str(uuid.uuid4())
        
        timekeeper = Timekeeper(
            timekeeper_id=timekeeper_id,
            name=name,
            initials=initials,
            role=role,
            standard_rate=Decimal(str(standard_rate)),
            **kwargs
        )
        
        self.timekeepers[timekeeper_id] = timekeeper
        return timekeeper
    
    def set_client_rate(
        self,
        timekeeper_id: str,
        client_id: str,
        rate: float
    ):
        """Set special rate for specific client"""
        if timekeeper_id in self.timekeepers:
            self.timekeepers[timekeeper_id].client_rates[client_id] = Decimal(str(rate))
    
    # ========================================
    # TIME ENTRY
    # ========================================
    
    def record_time(
        self,
        timekeeper_id: str,
        matter_id: str,
        date: str,
        hours: float,
        narrative: str,
        task_code: str = "",
        activity_code: str = "",
        client_id: str = ""
    ) -> TimeEntry:
        """Record time entry"""
        entry_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        # Get rate for timekeeper
        timekeeper = self.timekeepers.get(timekeeper_id)
        if timekeeper:
            rate = timekeeper.client_rates.get(client_id, timekeeper.standard_rate)
        else:
            rate = Decimal("0")
        
        hours_decimal = Decimal(str(hours))
        amount = (hours_decimal * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        
        entry = TimeEntry(
            entry_id=entry_id,
            timekeeper_id=timekeeper_id,
            matter_id=matter_id,
            date=date,
            hours=hours_decimal,
            narrative=narrative,
            task_code=task_code,
            activity_code=activity_code,
            rate=rate,
            amount=amount,
            created_date=now,
            modified_date=now
        )
        
        self.time_entries[entry_id] = entry
        
        # Check compliance if guidelines exist
        guideline = self._get_guideline_for_matter(matter_id, client_id)
        if guideline:
            self._check_entry_compliance(entry, guideline)
        
        return entry
    
    def _get_guideline_for_matter(
        self,
        matter_id: str,
        client_id: str
    ) -> Optional[BillingGuideline]:
        """Get billing guideline for matter/client"""
        # First try to find by client_id
        for guideline in self.guidelines.values():
            if guideline.client_id == client_id:
                return guideline
        return None
    
    def _check_entry_compliance(
        self,
        entry: TimeEntry,
        guideline: BillingGuideline
    ):
        """Check time entry compliance with billing guidelines"""
        issues = []
        
        # Check block billing
        if not guideline.block_billing_allowed:
            if self._is_block_billed(entry.narrative):
                entry.block_billed = True
                entry.block_bill_warning = "Entry appears to be block-billed. Separate tasks required."
                issues.append("Block billing detected")
        
        # Check task codes
        if guideline.require_task_codes and not entry.task_code:
            issues.append("Task code required")
        
        if guideline.require_activity_codes and not entry.activity_code:
            issues.append("Activity code required")
        
        # Check narrative length
        if len(entry.narrative) < guideline.minimum_narrative_length:
            issues.append(f"Narrative too short (minimum {guideline.minimum_narrative_length} characters)")
        
        # Check prohibited phrases
        narrative_lower = entry.narrative.lower()
        for phrase in guideline.prohibited_phrases:
            if phrase.lower() in narrative_lower:
                issues.append(f"Prohibited phrase: '{phrase}'")
        
        # Check vague terms
        vague_terms = guideline.vague_term_warnings or self.default_vague_phrases
        for term in vague_terms:
            if term.lower() in narrative_lower:
                issues.append(f"Vague term warning: '{term}' - consider being more specific")
        
        # Check hours
        if entry.hours > guideline.maximum_hours_per_day:
            issues.append(f"Hours exceed daily maximum ({guideline.maximum_hours_per_day})")
        
        # Check increment
        remainder = entry.hours % guideline.minimum_increment
        if remainder != Decimal("0"):
            issues.append(f"Hours not in {guideline.minimum_increment}-hour increments")
        
        # Update entry
        entry.narrative_compliant = len(issues) == 0
        entry.narrative_issues = issues
    
    def _is_block_billed(self, narrative: str) -> bool:
        """Detect if narrative is block billed"""
        # Look for multiple tasks in one entry
        indicators = [
            "; ", " and ", " & ",
            "reviewed", "drafted", "attended", "conference",
            "research", "prepare", "analyze"
        ]
        
        count = sum(1 for ind in indicators if ind.lower() in narrative.lower())
        
        # If 3+ indicators, likely block billed
        return count >= 3
    
    async def review_narrative_ai(
        self,
        entry_id: str,
        client_name: str = ""
    ) -> Dict[str, Any]:
        """AI-powered narrative review for compliance and clarity"""
        entry = self.time_entries.get(entry_id)
        if not entry:
            return {"error": "Entry not found"}
        
        if not self.client:
            return {"error": "AI service not available"}
        
        prompt = f"""Review this legal billing narrative for compliance and clarity:

Narrative: "{entry.narrative}"
Hours: {entry.hours}
Task Code: {entry.task_code or "Not specified"}
Client: {client_name or "Corporate client"}

Analyze for:
1. Block billing (multiple tasks lumped together)
2. Vague language that should be more specific
3. Prohibited or problematic phrases
4. Whether hours are reasonable for the described work
5. Missing details that add value
6. Redundant or unnecessary language

Provide:
- Compliance issues found
- Suggested improved narrative
- Recommendations for splitting if block-billed
- Risk level (high/medium/low) for client rejection

Respond in JSON format."""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert legal billing compliance analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            result["entry_id"] = entry_id
            return result
            
        except Exception as e:
            return {"error": str(e)}
    
    def adjust_time(
        self,
        entry_id: str,
        new_hours: float,
        reason: str
    ):
        """Adjust hours on time entry"""
        entry = self.time_entries.get(entry_id)
        if entry:
            entry.original_hours = entry.hours
            entry.hours = Decimal(str(new_hours))
            entry.adjustment_reason = reason
            entry.amount = (entry.hours * entry.rate).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            entry.modified_date = datetime.now().isoformat()
    
    # ========================================
    # EXPENSE TRACKING
    # ========================================
    
    def record_expense(
        self,
        matter_id: str,
        date: str,
        expense_type: str,
        description: str,
        amount: float,
        vendor: str = "",
        receipt_path: str = "",
        markup_rate: float = 0
    ) -> Expense:
        """Record billable expense"""
        expense_id = str(uuid.uuid4())
        
        amount_decimal = Decimal(str(amount))
        markup = Decimal(str(markup_rate))
        
        billed_amount = amount_decimal * (1 + markup)
        billed_amount = billed_amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        
        expense = Expense(
            expense_id=expense_id,
            matter_id=matter_id,
            date=date,
            expense_type=ExpenseType(expense_type) if expense_type in ExpenseType._value2member_map_ else ExpenseType.OTHER,
            description=description,
            amount=amount_decimal,
            vendor=vendor,
            receipt_path=receipt_path,
            markup_rate=markup,
            billed_amount=billed_amount
        )
        
        self.expenses[expense_id] = expense
        return expense
    
    # ========================================
    # BILLING GUIDELINES
    # ========================================
    
    def create_billing_guideline(
        self,
        client_id: str,
        client_name: str,
        partner_rate_cap: float = None,
        associate_rate_cap: float = None,
        paralegal_rate_cap: float = None,
        block_billing_allowed: bool = False,
        minimum_increment: float = 0.1,
        prohibited_phrases: List[str] = None,
        **kwargs
    ) -> BillingGuideline:
        """Create billing guideline for client"""
        guideline_id = str(uuid.uuid4())
        
        guideline = BillingGuideline(
            guideline_id=guideline_id,
            client_id=client_id,
            client_name=client_name,
            partner_rate_cap=Decimal(str(partner_rate_cap)) if partner_rate_cap else None,
            associate_rate_cap=Decimal(str(associate_rate_cap)) if associate_rate_cap else None,
            paralegal_rate_cap=Decimal(str(paralegal_rate_cap)) if paralegal_rate_cap else None,
            block_billing_allowed=block_billing_allowed,
            minimum_increment=Decimal(str(minimum_increment)),
            prohibited_phrases=prohibited_phrases or [],
            **kwargs
        )
        
        self.guidelines[guideline_id] = guideline
        return guideline
    
    # ========================================
    # INVOICE GENERATION
    # ========================================
    
    def generate_invoice(
        self,
        matter_id: str,
        client_id: str,
        billing_period_start: str,
        billing_period_end: str,
        payment_terms_days: int = 30
    ) -> Invoice:
        """Generate invoice from unbilled time and expenses"""
        invoice_id = str(uuid.uuid4())
        self._invoice_counter += 1
        invoice_number = f"INV-{self._invoice_counter:06d}"
        
        now = datetime.now()
        due_date = now + timedelta(days=payment_terms_days)
        
        # Collect unbilled time entries for this matter
        time_entry_ids = []
        total_fees = Decimal("0")
        
        for entry in self.time_entries.values():
            if (entry.matter_id == matter_id and 
                entry.status in [BillingStatus.DRAFT, BillingStatus.READY] and
                billing_period_start <= entry.date <= billing_period_end):
                
                time_entry_ids.append(entry.entry_id)
                total_fees += entry.amount or Decimal("0")
                entry.status = BillingStatus.BILLED
                entry.invoice_id = invoice_id
                entry.billed_date = now.isoformat()
        
        # Collect unbilled expenses
        expense_ids = []
        total_expenses = Decimal("0")
        
        for expense in self.expenses.values():
            if (expense.matter_id == matter_id and
                expense.status in [BillingStatus.DRAFT, BillingStatus.READY] and
                billing_period_start <= expense.date <= billing_period_end):
                
                expense_ids.append(expense.expense_id)
                total_expenses += expense.billed_amount or expense.amount
                expense.status = BillingStatus.BILLED
                expense.invoice_id = invoice_id
        
        subtotal = total_fees + total_expenses
        
        invoice = Invoice(
            invoice_id=invoice_id,
            invoice_number=invoice_number,
            matter_id=matter_id,
            client_id=client_id,
            invoice_date=now.strftime("%Y-%m-%d"),
            billing_period_start=billing_period_start,
            billing_period_end=billing_period_end,
            due_date=due_date.strftime("%Y-%m-%d"),
            total_fees=total_fees,
            total_expenses=total_expenses,
            subtotal=subtotal,
            total_due=subtotal,
            balance_due=subtotal,
            time_entries=time_entry_ids,
            expenses=expense_ids
        )
        
        self.invoices[invoice_id] = invoice
        return invoice
    
    def export_ledes(
        self,
        invoice_id: str
    ) -> str:
        """Export invoice in LEDES 1998B format"""
        invoice = self.invoices.get(invoice_id)
        if not invoice:
            return ""
        
        lines = []
        
        # Header
        lines.append("LEDES1998B[]")
        lines.append("INVOICE_DATE|INVOICE_NUMBER|CLIENT_ID|LAW_FIRM_MATTER_ID|INVOICE_TOTAL|"
                    "BILLING_START_DATE|BILLING_END_DATE|INVOICE_DESCRIPTION[]")
        lines.append(f"{invoice.invoice_date}|{invoice.invoice_number}|{invoice.client_id}|"
                    f"{invoice.matter_id}|{invoice.total_due}|{invoice.billing_period_start}|"
                    f"{invoice.billing_period_end}|Professional Services[]")
        
        # Line items header
        lines.append("LINE_ITEM_NUMBER|EXP/FEE/INV_ADJ_TYPE|LINE_ITEM_NUMBER_OF_UNITS|"
                    "LINE_ITEM_ADJUSTMENT_AMOUNT|LINE_ITEM_TOTAL|LINE_ITEM_DATE|"
                    "LINE_ITEM_TASK_CODE|LINE_ITEM_EXPENSE_CODE|LINE_ITEM_ACTIVITY_CODE|"
                    "TIMEKEEPER_ID|LINE_ITEM_DESCRIPTION|LAW_FIRM_ID|LINE_ITEM_UNIT_COST[]")
        
        line_num = 1
        
        # Time entries
        for entry_id in invoice.time_entries:
            entry = self.time_entries.get(entry_id)
            if entry:
                timekeeper = self.timekeepers.get(entry.timekeeper_id)
                tk_id = timekeeper.initials if timekeeper else ""
                
                lines.append(
                    f"{line_num}|F|{entry.hours}|0|{entry.amount}|{entry.date}|"
                    f"{entry.task_code}||{entry.activity_code}|{tk_id}|"
                    f"{entry.narrative.replace('|', ' ')}|FIRM|{entry.rate}[]"
                )
                line_num += 1
        
        # Expenses
        for expense_id in invoice.expenses:
            expense = self.expenses.get(expense_id)
            if expense:
                lines.append(
                    f"{line_num}|E|1|0|{expense.billed_amount}|{expense.date}|"
                    f"|{expense.expense_type.value}||{expense.vendor}|"
                    f"{expense.description.replace('|', ' ')}|FIRM|{expense.amount}[]"
                )
                line_num += 1
        
        return "\n".join(lines)
    
    # ========================================
    # BUDGET TRACKING
    # ========================================
    
    def create_budget(
        self,
        matter_id: str,
        total_budget: float,
        phases: Dict[str, float] = None
    ) -> BudgetTracker:
        """Create budget tracker for matter"""
        budget_id = str(uuid.uuid4())
        
        budget = BudgetTracker(
            budget_id=budget_id,
            matter_id=matter_id,
            total_budget=Decimal(str(total_budget)),
            phases={k: Decimal(str(v)) for k, v in (phases or {}).items()},
            remaining=Decimal(str(total_budget))
        )
        
        self.budgets[budget_id] = budget
        return budget
    
    def update_budget_tracking(
        self,
        matter_id: str
    ) -> Optional[Dict[str, Any]]:
        """Update budget tracking with current WIP and billed"""
        # Find budget for matter
        budget = next(
            (b for b in self.budgets.values() if b.matter_id == matter_id),
            None
        )
        
        if not budget:
            return None
        
        # Calculate WIP (unbilled time + expenses)
        wip = Decimal("0")
        for entry in self.time_entries.values():
            if entry.matter_id == matter_id and entry.status == BillingStatus.DRAFT:
                wip += entry.amount or Decimal("0")
        
        for expense in self.expenses.values():
            if expense.matter_id == matter_id and expense.status == BillingStatus.DRAFT:
                wip += expense.billed_amount or expense.amount
        
        # Calculate billed
        billed = Decimal("0")
        for invoice in self.invoices.values():
            if invoice.matter_id == matter_id:
                billed += invoice.total_due
        
        # Update budget
        budget.total_wip = wip
        budget.total_billed = billed
        budget.remaining = budget.total_budget - billed - wip
        
        if budget.total_budget > 0:
            budget.percent_used = ((billed + wip) / budget.total_budget * 100).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
        
        # Check thresholds
        alerts = []
        for threshold in budget.alert_thresholds:
            if budget.percent_used >= threshold and threshold not in budget.alerts_sent:
                alerts.append({
                    "threshold": threshold,
                    "percent_used": float(budget.percent_used),
                    "remaining": float(budget.remaining),
                    "message": f"Budget {threshold}% utilized - ${budget.remaining:.2f} remaining"
                })
                budget.alerts_sent.append(threshold)
        
        return {
            "budget_id": budget.budget_id,
            "total_budget": float(budget.total_budget),
            "total_billed": float(budget.total_billed),
            "total_wip": float(budget.total_wip),
            "remaining": float(budget.remaining),
            "percent_used": float(budget.percent_used),
            "new_alerts": alerts
        }
    
    # ========================================
    # REPORTS
    # ========================================
    
    def get_matter_summary(
        self,
        matter_id: str
    ) -> Dict[str, Any]:
        """Get billing summary for matter"""
        # Time entries
        time_entries = [e for e in self.time_entries.values() if e.matter_id == matter_id]
        
        total_hours = sum(e.hours for e in time_entries)
        total_fees = sum(e.amount or Decimal("0") for e in time_entries)
        unbilled_fees = sum(
            e.amount or Decimal("0") for e in time_entries
            if e.status == BillingStatus.DRAFT
        )
        
        # Expenses
        expenses = [e for e in self.expenses.values() if e.matter_id == matter_id]
        total_expenses = sum(e.billed_amount or e.amount for e in expenses)
        unbilled_expenses = sum(
            e.billed_amount or e.amount for e in expenses
            if e.status == BillingStatus.DRAFT
        )
        
        # By timekeeper
        by_timekeeper = {}
        for entry in time_entries:
            tk = self.timekeepers.get(entry.timekeeper_id)
            tk_name = tk.name if tk else "Unknown"
            if tk_name not in by_timekeeper:
                by_timekeeper[tk_name] = {"hours": Decimal("0"), "fees": Decimal("0")}
            by_timekeeper[tk_name]["hours"] += entry.hours
            by_timekeeper[tk_name]["fees"] += entry.amount or Decimal("0")
        
        return {
            "matter_id": matter_id,
            "total_hours": float(total_hours),
            "total_fees": float(total_fees),
            "unbilled_fees": float(unbilled_fees),
            "total_expenses": float(total_expenses),
            "unbilled_expenses": float(unbilled_expenses),
            "total_value": float(total_fees + total_expenses),
            "unbilled_total": float(unbilled_fees + unbilled_expenses),
            "by_timekeeper": {
                k: {"hours": float(v["hours"]), "fees": float(v["fees"])}
                for k, v in by_timekeeper.items()
            },
            "entry_count": len(time_entries),
            "expense_count": len(expenses)
        }


# Singleton instance
billing_service = BillingService()
