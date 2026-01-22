"""
BarberX Legal Case Management Pro Suite
Brady/Giglio Tracking & Exculpatory Evidence Service
"""
import os
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import uuid

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class EvidenceClassification(str, Enum):
    """Brady/Giglio evidence classifications"""
    BRADY_EXCULPATORY = "brady_exculpatory"  # Directly exculpatory
    BRADY_IMPEACHMENT = "brady_impeachment"  # Impeachment material
    GIGLIO_CREDIBILITY = "giglio_credibility"  # Witness credibility
    GIGLIO_BENEFIT = "giglio_benefit"  # Benefits to witness
    POTENTIALLY_EXCULPATORY = "potentially_exculpatory"
    NOT_APPLICABLE = "not_applicable"


class DisclosureStatus(str, Enum):
    """Disclosure status"""
    NOT_DISCLOSED = "not_disclosed"
    REQUESTED = "requested"
    DISCLOSED = "disclosed"
    DISPUTED = "disputed"
    WITHHELD_IMPROPERLY = "withheld_improperly"


class ViolationType(str, Enum):
    """Types of Brady violations"""
    SUPPRESSION = "suppression"  # Evidence withheld
    LATE_DISCLOSURE = "late_disclosure"
    INADEQUATE_DISCLOSURE = "inadequate_disclosure"
    DESTRUCTION = "destruction"  # Evidence destroyed
    FABRICATION = "fabrication"


@dataclass
class BradyEvidence:
    """Brady/Giglio evidence item"""
    evidence_id: str
    case_id: str
    
    # Classification
    classification: EvidenceClassification
    
    # Description
    description: str
    significance: str
    
    # Source
    source_type: str  # document, witness, physical, digital
    source_description: str
    source_document_id: Optional[str] = None
    
    # Disclosure
    disclosure_status: DisclosureStatus = DisclosureStatus.NOT_DISCLOSED
    disclosure_date: Optional[str] = None
    disclosure_method: str = ""
    
    # Analysis
    materiality_analysis: str = ""
    materiality_score: float = 0.0  # 0-100
    
    # Prejudice assessment
    prejudice_if_withheld: str = ""
    
    # Related evidence
    related_evidence_ids: List[str] = field(default_factory=list)
    
    # Officers/Witnesses affected
    officers_involved: List[str] = field(default_factory=list)
    witnesses_affected: List[str] = field(default_factory=list)
    
    # Notes
    notes: str = ""
    
    created_at: str = ""
    updated_at: str = ""


@dataclass
class OfficerCredibilityRecord:
    """Officer credibility information for Giglio"""
    record_id: str
    officer_name: str
    badge_number: str
    department: str
    
    # Disciplinary history
    sustained_complaints: List[Dict[str, Any]] = field(default_factory=list)
    pending_complaints: List[Dict[str, Any]] = field(default_factory=list)
    
    # Litigation history
    civil_suits: List[Dict[str, Any]] = field(default_factory=list)
    named_in_excessive_force: bool = False
    
    # Truthfulness issues
    dishonesty_findings: List[Dict[str, Any]] = field(default_factory=list)
    brady_list_status: bool = False  # On prosecutor's Brady list
    
    # Training issues
    remedial_training: List[Dict[str, Any]] = field(default_factory=list)
    
    # Overall assessment
    credibility_score: float = 0.0
    giglio_material_exists: bool = False
    
    created_at: str = ""


@dataclass
class BradyViolation:
    """Documented Brady violation"""
    violation_id: str
    case_id: str
    
    # Violation details
    violation_type: ViolationType
    description: str
    
    # Evidence involved
    evidence_id: str
    evidence_description: str
    
    # Responsible party
    responsible_party: str
    responsible_office: str  # e.g., "County Prosecutor's Office"
    
    # Discovery
    when_discovered: str
    how_discovered: str
    
    # Timing
    when_should_have_been_disclosed: str
    actual_disclosure_date: Optional[str] = None
    delay_in_days: int = 0
    
    # Impact
    materiality: str
    prejudice_caused: str
    
    # Remedies sought
    remedies_requested: List[str] = field(default_factory=list)
    remedies_granted: List[str] = field(default_factory=list)
    
    # Documentation
    supporting_documents: List[str] = field(default_factory=list)
    motion_filed: bool = False
    motion_document_id: Optional[str] = None
    
    created_at: str = ""


class BradyTrackingService:
    """
    Premium Brady/Giglio Tracking Service
    
    Features:
    - Exculpatory evidence identification
    - Giglio material tracking
    - Officer credibility database
    - Disclosure tracking
    - Violation documentation
    - Remedy tracking
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.client = AsyncOpenAI(api_key=self.api_key) if OPENAI_AVAILABLE and self.api_key else None
        
        # Storage
        self.brady_evidence: Dict[str, BradyEvidence] = {}
        self.officer_records: Dict[str, OfficerCredibilityRecord] = {}
        self.violations: Dict[str, BradyViolation] = {}
    
    # ============================================================
    # EVIDENCE CLASSIFICATION
    # ============================================================
    
    async def classify_evidence(
        self,
        evidence_description: str,
        case_facts: str,
        defendant_theory: str
    ) -> Dict[str, Any]:
        """
        AI-powered Brady/Giglio classification.
        
        Analyzes evidence for:
        - Direct exculpatory value
        - Impeachment potential
        - Witness credibility impact
        """
        if not self.client:
            return self._rule_based_classification(evidence_description)
        
        prompt = f"""Analyze this evidence under Brady v. Maryland and Giglio v. United States.

EVIDENCE DESCRIPTION:
{evidence_description}

CASE FACTS:
{case_facts}

DEFENDANT'S THEORY:
{defendant_theory}

Classify the evidence and provide analysis in JSON:
{{
    "classification": "brady_exculpatory|brady_impeachment|giglio_credibility|giglio_benefit|potentially_exculpatory|not_applicable",
    "is_material": true/false,
    "materiality_analysis": "detailed analysis of materiality under Brady",
    "materiality_score": 0-100,
    "exculpatory_value": {{
        "directly_exculpatory": true/false,
        "explanation": "how this evidence could exculpate defendant"
    }},
    "impeachment_value": {{
        "impeaches_witness": true/false,
        "witnesses_affected": ["list of witnesses"],
        "impeachment_basis": "how this impeaches testimony"
    }},
    "giglio_analysis": {{
        "affects_witness_credibility": true/false,
        "shows_witness_benefit": true/false,
        "explanation": "Giglio analysis"
    }},
    "disclosure_obligation": {{
        "must_disclose": true/false,
        "timing": "when disclosure required",
        "basis": "legal basis for disclosure"
    }},
    "prejudice_if_withheld": "analysis of prejudice",
    "case_citations": ["relevant Brady/Giglio cases"],
    "recommended_action": "what defense should do"
}}"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a criminal defense attorney expert in Brady and Giglio obligations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            import json
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            return {"error": str(e)}
    
    def _rule_based_classification(self, description: str) -> Dict[str, Any]:
        """Rule-based Brady classification"""
        desc_lower = description.lower()
        
        # Exculpatory indicators
        exculpatory_keywords = [
            "alibi", "another suspect", "exculpatory", "clears",
            "inconsistent", "contradicts", "false", "mistaken",
            "coerced", "recanted", "different description"
        ]
        
        # Impeachment indicators
        impeachment_keywords = [
            "prior conviction", "lied", "benefit", "deal",
            "immunity", "reduced charge", "payment", "incentive",
            "bias", "motive", "prior inconsistent"
        ]
        
        # Giglio indicators
        giglio_keywords = [
            "plea agreement", "cooperation", "reduction",
            "immunity", "favorable treatment", "witness benefit"
        ]
        
        exculp_score = sum(1 for kw in exculpatory_keywords if kw in desc_lower)
        impeach_score = sum(1 for kw in impeachment_keywords if kw in desc_lower)
        giglio_score = sum(1 for kw in giglio_keywords if kw in desc_lower)
        
        if exculp_score >= 2:
            classification = EvidenceClassification.BRADY_EXCULPATORY
        elif giglio_score >= 2:
            classification = EvidenceClassification.GIGLIO_BENEFIT
        elif impeach_score >= 2:
            classification = EvidenceClassification.BRADY_IMPEACHMENT
        elif exculp_score >= 1 or impeach_score >= 1:
            classification = EvidenceClassification.POTENTIALLY_EXCULPATORY
        else:
            classification = EvidenceClassification.NOT_APPLICABLE
        
        return {
            "classification": classification.value,
            "is_material": exculp_score + impeach_score + giglio_score >= 2,
            "materiality_score": (exculp_score + impeach_score + giglio_score) * 20
        }
    
    def create_brady_evidence(
        self,
        case_id: str,
        classification: EvidenceClassification,
        description: str,
        source_type: str,
        source_description: str,
        **kwargs
    ) -> BradyEvidence:
        """Create Brady evidence record"""
        evidence = BradyEvidence(
            evidence_id=f"brady_{uuid.uuid4().hex[:8]}",
            case_id=case_id,
            classification=classification,
            description=description,
            source_type=source_type,
            source_description=source_description,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            significance=kwargs.get('significance', ''),
            **{k: v for k, v in kwargs.items() if k != 'significance'}
        )
        
        self.brady_evidence[evidence.evidence_id] = evidence
        return evidence
    
    # ============================================================
    # OFFICER CREDIBILITY TRACKING
    # ============================================================
    
    def create_officer_record(
        self,
        officer_name: str,
        badge_number: str,
        department: str,
        **kwargs
    ) -> OfficerCredibilityRecord:
        """Create officer credibility record"""
        record = OfficerCredibilityRecord(
            record_id=f"officer_{uuid.uuid4().hex[:8]}",
            officer_name=officer_name,
            badge_number=badge_number,
            department=department,
            created_at=datetime.utcnow().isoformat(),
            **kwargs
        )
        
        self.officer_records[record.record_id] = record
        return record
    
    def add_disciplinary_record(
        self,
        record_id: str,
        complaint_date: str,
        complaint_type: str,
        allegation: str,
        finding: str,
        discipline_imposed: str = ""
    ):
        """Add disciplinary record to officer"""
        record = self.officer_records.get(record_id)
        if not record:
            raise ValueError(f"Officer record not found: {record_id}")
        
        complaint = {
            "date": complaint_date,
            "type": complaint_type,
            "allegation": allegation,
            "finding": finding,
            "discipline": discipline_imposed,
            "added_at": datetime.utcnow().isoformat()
        }
        
        if finding.lower() in ["sustained", "founded", "upheld"]:
            record.sustained_complaints.append(complaint)
            
            # Check for dishonesty
            if any(kw in allegation.lower() for kw in ["dishonesty", "false", "lied", "untruthful"]):
                record.dishonesty_findings.append(complaint)
                record.giglio_material_exists = True
        else:
            record.pending_complaints.append(complaint)
        
        # Recalculate credibility score
        self._calculate_officer_credibility(record)
    
    def _calculate_officer_credibility(self, record: OfficerCredibilityRecord):
        """Calculate officer credibility score"""
        score = 100
        
        # Deductions
        score -= len(record.sustained_complaints) * 10
        score -= len(record.dishonesty_findings) * 25
        score -= len(record.civil_suits) * 5
        
        if record.named_in_excessive_force:
            score -= 15
        
        if record.brady_list_status:
            score = 0  # On Brady list = zero credibility
        
        record.credibility_score = max(0, score)
    
    async def analyze_officer_giglio(
        self,
        record_id: str
    ) -> Dict[str, Any]:
        """Generate comprehensive Giglio analysis for officer"""
        record = self.officer_records.get(record_id)
        if not record:
            raise ValueError(f"Officer record not found: {record_id}")
        
        if not self.client:
            return {
                "officer": record.officer_name,
                "giglio_material_exists": record.giglio_material_exists,
                "credibility_score": record.credibility_score,
                "sustained_complaints": len(record.sustained_complaints),
                "dishonesty_findings": len(record.dishonesty_findings)
            }
        
        # Format officer history
        complaints_text = "\n".join(
            f"- {c.get('date', '')}: {c.get('allegation', '')} - {c.get('finding', '')}"
            for c in record.sustained_complaints
        )
        
        dishonesty_text = "\n".join(
            f"- {d.get('date', '')}: {d.get('allegation', '')}"
            for d in record.dishonesty_findings
        )
        
        suits_text = "\n".join(
            f"- {s.get('case_name', '')}: {s.get('allegation', '')} - {s.get('outcome', '')}"
            for s in record.civil_suits
        )
        
        prompt = f"""Prepare Giglio analysis for this law enforcement officer.

OFFICER: {record.officer_name}
BADGE: {record.badge_number}
DEPARTMENT: {record.department}

SUSTAINED COMPLAINTS:
{complaints_text or "None on record"}

DISHONESTY FINDINGS:
{dishonesty_text or "None on record"}

CIVIL LITIGATION:
{suits_text or "None on record"}

BRADY LIST STATUS: {"Yes" if record.brady_list_status else "No"}

Provide comprehensive Giglio analysis in JSON:
{{
    "giglio_material_exists": true/false,
    "disclosure_required": true/false,
    "disclosure_basis": "legal basis for disclosure requirement",
    "credibility_assessment": {{
        "overall_credibility": "credible|questionable|not_credible",
        "credibility_score": 0-100,
        "primary_concerns": ["main credibility issues"],
        "mitigating_factors": ["factors favoring credibility"]
    }},
    "impeachment_potential": {{
        "strong_impeachment_material": ["strongest impeachment items"],
        "character_for_truthfulness": "admissible|inadmissible|borderline",
        "cross_examination_topics": ["topics for cross"]
    }},
    "motion_recommendations": {{
        "file_giglio_motion": true/false,
        "motion_arguments": ["arguments for motion"],
        "likely_success": "high|medium|low"
    }},
    "voir_dire_implications": ["impact on jury selection"],
    "trial_strategy": "recommended approach at trial"
}}"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a criminal defense attorney expert in Giglio material and police officer credibility issues."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            import json
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            return {"error": str(e)}
    
    # ============================================================
    # VIOLATION TRACKING
    # ============================================================
    
    def document_violation(
        self,
        case_id: str,
        violation_type: ViolationType,
        evidence_id: str,
        responsible_party: str,
        when_discovered: str,
        how_discovered: str,
        **kwargs
    ) -> BradyViolation:
        """Document a Brady violation"""
        evidence = self.brady_evidence.get(evidence_id)
        
        violation = BradyViolation(
            violation_id=f"viol_{uuid.uuid4().hex[:8]}",
            case_id=case_id,
            violation_type=violation_type,
            description=kwargs.get('description', ''),
            evidence_id=evidence_id,
            evidence_description=evidence.description if evidence else '',
            responsible_party=responsible_party,
            responsible_office=kwargs.get('responsible_office', ''),
            when_discovered=when_discovered,
            how_discovered=how_discovered,
            when_should_have_been_disclosed=kwargs.get('when_should_have_been_disclosed', ''),
            created_at=datetime.utcnow().isoformat()
        )
        
        self.violations[violation.violation_id] = violation
        return violation
    
    async def generate_brady_motion(
        self,
        violation_id: str,
        case_caption: str,
        court: str
    ) -> str:
        """Generate Brady motion based on documented violation"""
        violation = self.violations.get(violation_id)
        if not violation:
            raise ValueError(f"Violation not found: {violation_id}")
        
        if not self.client:
            return f"Brady Motion re: {violation.evidence_description}"
        
        prompt = f"""Draft a motion addressing this Brady violation.

CASE: {case_caption}
COURT: {court}

VIOLATION:
Type: {violation.violation_type.value}
Description: {violation.description}
Evidence: {violation.evidence_description}
Responsible Party: {violation.responsible_party}
When Discovered: {violation.when_discovered}
How Discovered: {violation.how_discovered}
When Should Have Been Disclosed: {violation.when_should_have_been_disclosed}

Draft a formal motion with:
1. Caption (leave blank for case number)
2. Introduction
3. Statement of Facts
4. Argument (citing Brady v. Maryland, 373 U.S. 83)
5. Prejudice Analysis
6. Requested Relief
7. Conclusion

Be thorough and cite relevant case law."""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a criminal defense attorney drafting a Brady motion."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating motion: {str(e)}"
    
    # ============================================================
    # REPORTING
    # ============================================================
    
    def generate_brady_report(self, case_id: str) -> Dict[str, Any]:
        """Generate comprehensive Brady/Giglio report for case"""
        case_evidence = [e for e in self.brady_evidence.values() if e.case_id == case_id]
        case_violations = [v for v in self.violations.values() if v.case_id == case_id]
        
        # Collect all officers involved
        officer_ids = set()
        for e in case_evidence:
            officer_ids.update(e.officers_involved)
        
        return {
            "case_id": case_id,
            "generated_at": datetime.utcnow().isoformat(),
            "summary": {
                "total_brady_items": len(case_evidence),
                "exculpatory_count": len([e for e in case_evidence if e.classification == EvidenceClassification.BRADY_EXCULPATORY]),
                "impeachment_count": len([e for e in case_evidence if e.classification == EvidenceClassification.BRADY_IMPEACHMENT]),
                "giglio_count": len([e for e in case_evidence if e.classification in [EvidenceClassification.GIGLIO_CREDIBILITY, EvidenceClassification.GIGLIO_BENEFIT]]),
                "undisclosed_count": len([e for e in case_evidence if e.disclosure_status == DisclosureStatus.NOT_DISCLOSED]),
                "violations_documented": len(case_violations)
            },
            "evidence": [
                {
                    "id": e.evidence_id,
                    "classification": e.classification.value,
                    "description": e.description,
                    "materiality_score": e.materiality_score,
                    "disclosure_status": e.disclosure_status.value
                }
                for e in case_evidence
            ],
            "violations": [
                {
                    "id": v.violation_id,
                    "type": v.violation_type.value,
                    "description": v.description,
                    "responsible_party": v.responsible_party
                }
                for v in case_violations
            ],
            "officers_with_giglio_material": [
                self.officer_records[oid].officer_name
                for oid in officer_ids
                if oid in self.officer_records and self.officer_records[oid].giglio_material_exists
            ]
        }


# Service instance
brady_service = BradyTrackingService()
