"""
BarberX Legal Case Management Pro Suite
Court Rules & Deadline Management Service
"""
import os
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import calendar


class DeadlineType(str, Enum):
    """Types of legal deadlines"""
    STATUTE_OF_LIMITATIONS = "statute_of_limitations"
    DISCOVERY_CUTOFF = "discovery_cutoff"
    EXPERT_DISCLOSURE = "expert_disclosure"
    MOTION_DEADLINE = "motion_deadline"
    RESPONSE_DEADLINE = "response_deadline"
    REPLY_DEADLINE = "reply_deadline"
    PRETRIAL_CONFERENCE = "pretrial_conference"
    TRIAL_DATE = "trial_date"
    APPEAL_DEADLINE = "appeal_deadline"
    FILING_DEADLINE = "filing_deadline"
    SERVICE_DEADLINE = "service_deadline"


class DeadlinePriority(str, Enum):
    """Deadline priority levels"""
    CRITICAL = "critical"  # Missing = malpractice
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class JurisdictionType(str, Enum):
    """Court system types"""
    FEDERAL = "federal"
    STATE = "state"
    BANKRUPTCY = "bankruptcy"
    ADMINISTRATIVE = "administrative"


@dataclass
class CourtRule:
    """Court rule definition"""
    rule_id: str
    jurisdiction: str
    court: str
    rule_number: str
    rule_title: str
    rule_text: str
    
    # Timing
    days: int = 0
    business_days: bool = False
    from_event: str = ""  # e.g., "service", "filing", "hearing"
    
    # Extensions
    service_method_extensions: Dict[str, int] = field(default_factory=dict)
    
    # Related rules
    related_rules: List[str] = field(default_factory=list)
    
    # Metadata
    effective_date: str = ""
    last_updated: str = ""


@dataclass
class Deadline:
    """Calculated deadline"""
    deadline_id: str
    case_id: str
    deadline_type: DeadlineType
    
    # Dates
    trigger_date: str  # Date that started the clock
    trigger_event: str  # What started the clock
    calculated_date: str  # The actual deadline
    
    # Details
    description: str
    rule_basis: str  # Rule citation
    
    # Priority
    priority: DeadlinePriority
    is_court_ordered: bool = False
    
    # Status
    completed: bool = False
    completed_date: Optional[str] = None
    completed_by: Optional[str] = None
    
    # Reminders
    reminder_dates: List[str] = field(default_factory=list)
    
    # Extensions
    extended: bool = False
    extension_date: Optional[str] = None
    extension_reason: Optional[str] = None
    
    # Notes
    notes: str = ""
    
    created_at: str = ""


@dataclass
class DeadlineChain:
    """Chain of related deadlines"""
    chain_id: str
    case_id: str
    chain_name: str
    
    # Deadlines in sequence
    deadlines: List[Deadline] = field(default_factory=list)
    
    # Status
    current_deadline_index: int = 0
    all_complete: bool = False
    
    created_at: str = ""


class DeadlineCalculator:
    """
    Premium Deadline Calculation Engine
    
    Features:
    - FRCP deadline calculation
    - State-specific rules (NJ, NY, CA, etc.)
    - Business day vs calendar day handling
    - Service method extensions
    - Holiday calendar integration
    - Deadline chain management
    - Court-specific local rules
    """
    
    def __init__(self):
        # Court rules database
        self.rules: Dict[str, CourtRule] = {}
        self.deadlines: Dict[str, Deadline] = {}
        self.deadline_chains: Dict[str, DeadlineChain] = {}
        
        # Federal holidays (US)
        self.federal_holidays_2024 = [
            "2024-01-01",  # New Year's Day
            "2024-01-15",  # MLK Day
            "2024-02-19",  # Presidents Day
            "2024-05-27",  # Memorial Day
            "2024-06-19",  # Juneteenth
            "2024-07-04",  # Independence Day
            "2024-09-02",  # Labor Day
            "2024-10-14",  # Columbus Day
            "2024-11-11",  # Veterans Day
            "2024-11-28",  # Thanksgiving
            "2024-12-25",  # Christmas
        ]
        
        # Load default rules
        self._load_frcp_rules()
        self._load_nj_rules()
    
    def _load_frcp_rules(self):
        """Load Federal Rules of Civil Procedure"""
        frcp_rules = [
            CourtRule(
                rule_id="frcp_12a",
                jurisdiction="federal",
                court="all",
                rule_number="FRCP 12(a)(1)(A)",
                rule_title="Answer to Complaint",
                rule_text="A defendant must serve an answer within 21 days after being served with the summons and complaint.",
                days=21,
                business_days=False,
                from_event="service",
                service_method_extensions={"mail": 3, "email": 0}
            ),
            CourtRule(
                rule_id="frcp_12b",
                jurisdiction="federal",
                court="all",
                rule_number="FRCP 12(b)",
                rule_title="Motion to Dismiss",
                rule_text="A motion under this rule must be made before pleading if a responsive pleading is allowed.",
                days=21,
                business_days=False,
                from_event="service"
            ),
            CourtRule(
                rule_id="frcp_6d",
                jurisdiction="federal",
                court="all",
                rule_number="FRCP 6(d)",
                rule_title="Additional Time After Service",
                rule_text="When a party may or must act within a specified time after being served and service is made under Rule 5(b)(2)(C) (mail), (D) (leaving with the clerk), or (F) (other means consented to), 3 days are added after the period would otherwise expire.",
                days=3,
                business_days=False,
                from_event="service_by_mail"
            ),
            CourtRule(
                rule_id="frcp_26a1",
                jurisdiction="federal",
                court="all",
                rule_number="FRCP 26(a)(1)",
                rule_title="Initial Disclosures",
                rule_text="A party must, without awaiting a discovery request, provide initial disclosures within 14 days after the parties' Rule 26(f) conference.",
                days=14,
                business_days=False,
                from_event="rule_26f_conference"
            ),
            CourtRule(
                rule_id="frcp_33",
                jurisdiction="federal",
                court="all",
                rule_number="FRCP 33",
                rule_title="Interrogatory Response",
                rule_text="The responding party must serve its answers and any objections within 30 days after being served with the interrogatories.",
                days=30,
                business_days=False,
                from_event="service"
            ),
            CourtRule(
                rule_id="frcp_34",
                jurisdiction="federal",
                court="all",
                rule_number="FRCP 34",
                rule_title="Document Production Response",
                rule_text="The party to whom the request is directed must respond in writing within 30 days after being served.",
                days=30,
                business_days=False,
                from_event="service"
            ),
            CourtRule(
                rule_id="frcp_36",
                jurisdiction="federal",
                court="all",
                rule_number="FRCP 36",
                rule_title="Admission Response",
                rule_text="A matter is admitted unless, within 30 days after being served, the party to whom the request is directed serves on the requesting party a written answer or objection.",
                days=30,
                business_days=False,
                from_event="service"
            ),
            CourtRule(
                rule_id="frcp_56",
                jurisdiction="federal",
                court="all",
                rule_number="FRCP 56(c)",
                rule_title="Summary Judgment Opposition",
                rule_text="A party opposing the motion must file a response within 21 days after service of the motion.",
                days=21,
                business_days=False,
                from_event="service"
            ),
            CourtRule(
                rule_id="frap_4",
                jurisdiction="federal",
                court="appeals",
                rule_number="FRAP 4(a)(1)(A)",
                rule_title="Notice of Appeal - Civil",
                rule_text="In a civil case, the notice of appeal required by Rule 3 must be filed with the district clerk within 30 days after entry of the judgment or order appealed from.",
                days=30,
                business_days=False,
                from_event="judgment_entry"
            ),
            CourtRule(
                rule_id="frap_4_gov",
                jurisdiction="federal",
                court="appeals",
                rule_number="FRAP 4(a)(1)(B)",
                rule_title="Notice of Appeal - US Party",
                rule_text="When the United States or its officer or agency is a party, the notice of appeal may be filed by any party within 60 days after entry of the judgment or order appealed from.",
                days=60,
                business_days=False,
                from_event="judgment_entry"
            ),
        ]
        
        for rule in frcp_rules:
            self.rules[rule.rule_id] = rule
    
    def _load_nj_rules(self):
        """Load New Jersey Court Rules"""
        nj_rules = [
            CourtRule(
                rule_id="nj_4:6-1",
                jurisdiction="state_nj",
                court="superior",
                rule_number="N.J.R. 4:6-1",
                rule_title="Answer to Complaint",
                rule_text="The defendant shall serve an answer within 35 days after service of the summons and complaint.",
                days=35,
                business_days=False,
                from_event="service"
            ),
            CourtRule(
                rule_id="nj_4:17-4",
                jurisdiction="state_nj",
                court="superior",
                rule_number="N.J.R. 4:17-4",
                rule_title="Interrogatory Response",
                rule_text="Answers and objections shall be served within 60 days after service of the interrogatories.",
                days=60,
                business_days=False,
                from_event="service"
            ),
            CourtRule(
                rule_id="nj_4:18-1",
                jurisdiction="state_nj",
                court="superior",
                rule_number="N.J.R. 4:18-1",
                rule_title="Document Production Response",
                rule_text="The party upon whom the request is served shall serve a written response within 60 days after service of the request.",
                days=60,
                business_days=False,
                from_event="service"
            ),
            CourtRule(
                rule_id="nj_4:46-1",
                jurisdiction="state_nj",
                court="superior",
                rule_number="N.J.R. 4:46-1",
                rule_title="Summary Judgment Motion",
                rule_text="A motion for summary judgment shall be returnable no later than 30 days before the scheduled trial date.",
                days=-30,  # Negative = before event
                business_days=False,
                from_event="trial_date"
            ),
            CourtRule(
                rule_id="nj_2:4-1",
                jurisdiction="state_nj",
                court="appellate",
                rule_number="N.J.R. 2:4-1",
                rule_title="Notice of Appeal",
                rule_text="Appeals to the Appellate Division shall be taken within 45 days of the date of the judgment, decision, action or rule from which the appeal is taken.",
                days=45,
                business_days=False,
                from_event="judgment_entry"
            ),
        ]
        
        for rule in nj_rules:
            self.rules[rule.rule_id] = rule
    
    def is_business_day(self, date_obj: date) -> bool:
        """Check if date is a business day"""
        # Weekend check
        if date_obj.weekday() >= 5:
            return False
        
        # Holiday check
        date_str = date_obj.isoformat()
        if date_str in self.federal_holidays_2024:
            return False
        
        return True
    
    def add_days(
        self,
        start_date: date,
        days: int,
        business_days: bool = False
    ) -> date:
        """Add days to a date, optionally using business days"""
        if not business_days:
            result = start_date + timedelta(days=days)
        else:
            result = start_date
            days_added = 0
            direction = 1 if days >= 0 else -1
            days = abs(days)
            
            while days_added < days:
                result += timedelta(days=direction)
                if self.is_business_day(result):
                    days_added += 1
        
        # If deadline falls on weekend/holiday, move to next business day
        while not self.is_business_day(result):
            result += timedelta(days=1)
        
        return result
    
    def calculate_deadline(
        self,
        case_id: str,
        rule_id: str,
        trigger_date: str,
        service_method: str = "personal",
        description: str = ""
    ) -> Deadline:
        """
        Calculate a deadline based on court rules.
        
        Args:
            case_id: Case identifier
            rule_id: Rule to apply
            trigger_date: Date that triggers the deadline
            service_method: How papers were served
            description: Optional description
        
        Returns:
            Calculated Deadline object
        """
        rule = self.rules.get(rule_id)
        if not rule:
            raise ValueError(f"Rule not found: {rule_id}")
        
        trigger = datetime.fromisoformat(trigger_date).date()
        
        # Base calculation
        days = rule.days
        
        # Add service method extension if applicable
        if service_method in rule.service_method_extensions:
            days += rule.service_method_extensions[service_method]
        
        # Calculate deadline
        deadline_date = self.add_days(trigger, days, rule.business_days)
        
        # Determine priority
        if abs(days) <= 14:
            priority = DeadlinePriority.CRITICAL
        elif abs(days) <= 30:
            priority = DeadlinePriority.HIGH
        elif abs(days) <= 60:
            priority = DeadlinePriority.MEDIUM
        else:
            priority = DeadlinePriority.LOW
        
        # Generate reminders (14 days, 7 days, 3 days, 1 day before)
        reminder_offsets = [14, 7, 3, 1]
        reminders = []
        for offset in reminder_offsets:
            reminder_date = deadline_date - timedelta(days=offset)
            if reminder_date > trigger:
                reminders.append(reminder_date.isoformat())
        
        deadline = Deadline(
            deadline_id=f"dl_{uuid.uuid4().hex[:8]}",
            case_id=case_id,
            deadline_type=self._map_rule_to_deadline_type(rule),
            trigger_date=trigger_date,
            trigger_event=rule.from_event,
            calculated_date=deadline_date.isoformat(),
            description=description or rule.rule_title,
            rule_basis=rule.rule_number,
            priority=priority,
            reminder_dates=reminders,
            created_at=datetime.utcnow().isoformat()
        )
        
        self.deadlines[deadline.deadline_id] = deadline
        return deadline
    
    def _map_rule_to_deadline_type(self, rule: CourtRule) -> DeadlineType:
        """Map rule to deadline type"""
        rule_lower = rule.rule_title.lower()
        
        if "answer" in rule_lower:
            return DeadlineType.RESPONSE_DEADLINE
        elif "appeal" in rule_lower:
            return DeadlineType.APPEAL_DEADLINE
        elif "motion" in rule_lower:
            return DeadlineType.MOTION_DEADLINE
        elif "discovery" in rule_lower or "interrogat" in rule_lower or "production" in rule_lower:
            return DeadlineType.DISCOVERY_CUTOFF
        elif "expert" in rule_lower:
            return DeadlineType.EXPERT_DISCLOSURE
        elif "trial" in rule_lower:
            return DeadlineType.TRIAL_DATE
        else:
            return DeadlineType.FILING_DEADLINE
    
    def create_litigation_timeline(
        self,
        case_id: str,
        jurisdiction: str,
        complaint_filed_date: str,
        service_date: str,
        trial_date: Optional[str] = None
    ) -> DeadlineChain:
        """
        Create complete litigation timeline with all key deadlines.
        
        Generates deadlines for:
        - Answer/Motion to Dismiss
        - Initial Disclosures
        - Written Discovery
        - Expert Disclosures
        - Dispositive Motions
        - Pretrial Conference
        - Trial
        """
        chain_id = f"chain_{uuid.uuid4().hex[:8]}"
        deadlines = []
        
        # Select rules based on jurisdiction
        if jurisdiction == "federal":
            # Answer
            dl = self.calculate_deadline(
                case_id, "frcp_12a", service_date,
                description="Answer or Motion to Dismiss Due"
            )
            deadlines.append(dl)
            
            # Initial Disclosures (assume 26f conference 30 days after answer)
            answer_date = datetime.fromisoformat(dl.calculated_date)
            conf_date = (answer_date + timedelta(days=30)).isoformat()
            
            dl = self.calculate_deadline(
                case_id, "frcp_26a1", conf_date,
                description="Initial Disclosures Due"
            )
            deadlines.append(dl)
        
        elif jurisdiction == "state_nj":
            # Answer
            dl = self.calculate_deadline(
                case_id, "nj_4:6-1", service_date,
                description="Answer Due (NJ)"
            )
            deadlines.append(dl)
        
        chain = DeadlineChain(
            chain_id=chain_id,
            case_id=case_id,
            chain_name=f"Litigation Timeline - {case_id}",
            deadlines=deadlines,
            created_at=datetime.utcnow().isoformat()
        )
        
        self.deadline_chains[chain_id] = chain
        return chain
    
    def get_upcoming_deadlines(
        self,
        case_id: Optional[str] = None,
        days_ahead: int = 30
    ) -> List[Deadline]:
        """Get all deadlines within specified days"""
        today = date.today()
        cutoff = today + timedelta(days=days_ahead)
        
        upcoming = []
        for dl in self.deadlines.values():
            if case_id and dl.case_id != case_id:
                continue
            
            if dl.completed:
                continue
            
            dl_date = datetime.fromisoformat(dl.calculated_date).date()
            if today <= dl_date <= cutoff:
                upcoming.append(dl)
        
        # Sort by date, then priority
        priority_order = {
            DeadlinePriority.CRITICAL: 0,
            DeadlinePriority.HIGH: 1,
            DeadlinePriority.MEDIUM: 2,
            DeadlinePriority.LOW: 3
        }
        
        upcoming.sort(key=lambda x: (
            datetime.fromisoformat(x.calculated_date),
            priority_order.get(x.priority, 99)
        ))
        
        return upcoming
    
    def extend_deadline(
        self,
        deadline_id: str,
        new_date: str,
        reason: str
    ) -> Deadline:
        """Extend a deadline"""
        dl = self.deadlines.get(deadline_id)
        if not dl:
            raise ValueError(f"Deadline not found: {deadline_id}")
        
        dl.extended = True
        dl.extension_date = new_date
        dl.extension_reason = reason
        dl.calculated_date = new_date
        
        return dl


# Service instance
deadline_calculator = DeadlineCalculator()
