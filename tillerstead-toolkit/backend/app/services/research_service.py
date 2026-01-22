"""
BarberX Legal Case Management Pro Suite
Legal Research Integration Service

Premium service for integrating legal research databases,
managing research memos, and tracking case citations.
"""
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import json

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class JurisdictionType(Enum):
    """Types of jurisdictions"""
    FEDERAL = "federal"
    STATE = "state"
    THIRD_CIRCUIT = "third_circuit"
    SCOTUS = "scotus"
    NJ_SUPREME = "nj_supreme"
    NJ_APPELLATE = "nj_appellate"
    NJ_SUPERIOR = "nj_superior"


class CaseStatus(Enum):
    """Status of legal authority"""
    GOOD_LAW = "good_law"
    CAUTION = "caution"
    QUESTIONED = "questioned"
    OVERRULED = "overruled"
    SUPERSEDED = "superseded"
    DISTINGUISHED = "distinguished"


class ResearchType(Enum):
    """Types of legal research"""
    CASE_LAW = "case_law"
    STATUTE = "statute"
    REGULATION = "regulation"
    SECONDARY_SOURCE = "secondary_source"
    COURT_RULES = "court_rules"
    LEGISLATIVE_HISTORY = "legislative_history"


@dataclass
class LegalCitation:
    """Standardized legal citation"""
    citation_id: str
    full_citation: str
    short_citation: str
    case_name: str
    reporter: str
    volume: str
    page: str
    year: int
    court: str
    jurisdiction: JurisdictionType
    pinpoint: str = ""
    parallel_citations: List[str] = field(default_factory=list)
    
    # Status tracking
    status: CaseStatus = CaseStatus.GOOD_LAW
    status_date: str = ""
    status_source: str = ""
    
    # Key information
    holding: str = ""
    key_quotes: List[str] = field(default_factory=list)
    headnotes: List[str] = field(default_factory=list)
    

@dataclass
class CitationAnalysis:
    """Analysis of citation relationships"""
    analysis_id: str
    primary_citation_id: str
    citing_cases: List[str] = field(default_factory=list)  # Citation IDs
    cited_cases: List[str] = field(default_factory=list)  # Citation IDs
    treatment_summary: Dict[str, int] = field(default_factory=dict)  # Treatment: Count
    negative_treatment: List[Dict[str, Any]] = field(default_factory=list)
    distinguishing_cases: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ResearchMemo:
    """Legal research memorandum"""
    memo_id: str
    matter_id: str
    author: str
    date: str
    
    # Content
    question_presented: str
    brief_answer: str
    facts: str
    discussion: str
    conclusion: str
    
    # Citations
    authorities: List[str] = field(default_factory=list)  # Citation IDs
    
    # Metadata
    research_type: ResearchType = ResearchType.CASE_LAW
    jurisdiction: JurisdictionType = JurisdictionType.FEDERAL
    time_spent_hours: float = 0
    status: str = "draft"  # draft, review, final
    reviewed_by: str = ""
    review_date: str = ""


@dataclass
class KeyIssue:
    """Key legal issue being researched"""
    issue_id: str
    matter_id: str
    issue_description: str
    elements: List[str] = field(default_factory=list)
    favorable_cases: List[str] = field(default_factory=list)  # Citation IDs
    unfavorable_cases: List[str] = field(default_factory=list)  # Citation IDs
    distinguishing_factors: List[str] = field(default_factory=list)
    research_notes: str = ""
    assessment: str = ""  # favorable, unfavorable, uncertain


@dataclass
class MotionBank:
    """Bank of motion templates and precedents"""
    motion_id: str
    motion_type: str
    title: str
    jurisdiction: JurisdictionType
    court: str
    
    # Content
    template_text: str
    sample_arguments: List[str] = field(default_factory=list)
    key_cases: List[str] = field(default_factory=list)  # Citation IDs
    success_rate: Optional[float] = None
    
    # Usage
    times_used: int = 0
    last_used: str = ""
    outcomes: List[Dict[str, Any]] = field(default_factory=list)


class LegalResearchService:
    """
    Comprehensive legal research service.
    
    Features:
    - Citation parsing and validation
    - Case status checking (Shepardizing)
    - Research memo management
    - Issue tracking
    - AI-powered research assistance
    - Motion bank
    """
    
    def __init__(self):
        self.citations: Dict[str, LegalCitation] = {}
        self.analyses: Dict[str, CitationAnalysis] = {}
        self.memos: Dict[str, ResearchMemo] = {}
        self.issues: Dict[str, KeyIssue] = {}
        self.motions: Dict[str, MotionBank] = {}
        self.client = AsyncOpenAI() if OPENAI_AVAILABLE else None
        
        # Initialize with key civil rights cases
        self._initialize_key_authorities()
    
    def _initialize_key_authorities(self):
        """Initialize database with key civil rights authorities"""
        key_cases = [
            {
                "case_name": "Graham v. Connor",
                "full_citation": "490 U.S. 386 (1989)",
                "short_citation": "Graham, 490 U.S. at 386",
                "reporter": "U.S.",
                "volume": "490",
                "page": "386",
                "year": 1989,
                "court": "Supreme Court of the United States",
                "jurisdiction": JurisdictionType.SCOTUS,
                "holding": "Claims of excessive force by law enforcement must be analyzed under the Fourth Amendment's 'objective reasonableness' standard.",
                "key_quotes": [
                    "The 'reasonableness' of a particular use of force must be judged from the perspective of a reasonable officer on the scene, rather than with the 20/20 vision of hindsight.",
                    "The calculus of reasonableness must embody allowance for the fact that police officers are often forced to make split-second judgments."
                ]
            },
            {
                "case_name": "Monell v. Department of Social Services",
                "full_citation": "436 U.S. 658 (1978)",
                "short_citation": "Monell, 436 U.S. at 658",
                "reporter": "U.S.",
                "volume": "436",
                "page": "658",
                "year": 1978,
                "court": "Supreme Court of the United States",
                "jurisdiction": JurisdictionType.SCOTUS,
                "holding": "Local governments can be sued under § 1983 for policies that cause constitutional violations, but not on respondeat superior theory.",
                "key_quotes": [
                    "Local governing bodies... can be sued directly under § 1983 for monetary, declaratory, or injunctive relief where... the action that is alleged to be unconstitutional implements or executes a policy statement, ordinance, regulation, or decision officially adopted."
                ]
            },
            {
                "case_name": "Tennessee v. Garner",
                "full_citation": "471 U.S. 1 (1985)",
                "short_citation": "Garner, 471 U.S. at 1",
                "reporter": "U.S.",
                "volume": "471",
                "page": "1",
                "year": 1985,
                "court": "Supreme Court of the United States",
                "jurisdiction": JurisdictionType.SCOTUS,
                "holding": "Use of deadly force to prevent escape of an unarmed fleeing felon violates the Fourth Amendment unless the officer has probable cause to believe the suspect poses a significant threat of death or serious physical injury.",
                "key_quotes": [
                    "Where the suspect poses no immediate threat to the officer and no threat to others, the harm resulting from failing to apprehend him does not justify the use of deadly force to do so."
                ]
            },
            {
                "case_name": "Saucier v. Katz",
                "full_citation": "533 U.S. 194 (2001)",
                "short_citation": "Saucier, 533 U.S. at 194",
                "reporter": "U.S.",
                "volume": "533",
                "page": "194",
                "year": 2001,
                "court": "Supreme Court of the United States",
                "jurisdiction": JurisdictionType.SCOTUS,
                "holding": "Qualified immunity analysis requires determining (1) whether a constitutional right was violated, and (2) whether that right was clearly established.",
                "key_quotes": [
                    "The relevant, dispositive inquiry in determining whether a right is clearly established is whether it would be clear to a reasonable officer that his conduct was unlawful in the situation he confronted."
                ]
            },
            {
                "case_name": "Pearson v. Callahan",
                "full_citation": "555 U.S. 223 (2009)",
                "short_citation": "Pearson, 555 U.S. at 223",
                "reporter": "U.S.",
                "volume": "555",
                "page": "223",
                "year": 2009,
                "court": "Supreme Court of the United States",
                "jurisdiction": JurisdictionType.SCOTUS,
                "holding": "Courts have discretion to decide qualified immunity cases in whatever order is appropriate, rather than being required to follow the Saucier sequence.",
                "key_quotes": [
                    "The judges of the district courts and the courts of appeals should be permitted to exercise their sound discretion in deciding which of the two prongs of the qualified immunity analysis should be addressed first."
                ]
            },
            {
                "case_name": "Brady v. Maryland",
                "full_citation": "373 U.S. 83 (1963)",
                "short_citation": "Brady, 373 U.S. at 83",
                "reporter": "U.S.",
                "volume": "373",
                "page": "83",
                "year": 1963,
                "court": "Supreme Court of the United States",
                "jurisdiction": JurisdictionType.SCOTUS,
                "holding": "Suppression by the prosecution of evidence favorable to an accused violates due process where the evidence is material to guilt or punishment.",
                "key_quotes": [
                    "Society wins not only when the guilty are convicted but when criminal trials are fair; our system of the administration of justice suffers when any accused is treated unfairly."
                ]
            },
            {
                "case_name": "Giglio v. United States",
                "full_citation": "405 U.S. 150 (1972)",
                "short_citation": "Giglio, 405 U.S. at 150",
                "reporter": "U.S.",
                "volume": "405",
                "page": "150",
                "year": 1972,
                "court": "Supreme Court of the United States",
                "jurisdiction": JurisdictionType.SCOTUS,
                "holding": "Brady disclosure obligation extends to impeachment evidence affecting witness credibility.",
                "key_quotes": [
                    "When the reliability of a given witness may well be determinative of guilt or innocence, nondisclosure of evidence affecting credibility falls within the general rule of Brady."
                ]
            }
        ]
        
        for case in key_cases:
            citation_id = str(uuid.uuid4())
            self.citations[citation_id] = LegalCitation(
                citation_id=citation_id,
                full_citation=case["full_citation"],
                short_citation=case["short_citation"],
                case_name=case["case_name"],
                reporter=case["reporter"],
                volume=case["volume"],
                page=case["page"],
                year=case["year"],
                court=case["court"],
                jurisdiction=case["jurisdiction"],
                status=CaseStatus.GOOD_LAW,
                holding=case["holding"],
                key_quotes=case.get("key_quotes", [])
            )
    
    # ========================================
    # CITATION MANAGEMENT
    # ========================================
    
    def add_citation(
        self,
        case_name: str,
        full_citation: str,
        reporter: str,
        volume: str,
        page: str,
        year: int,
        court: str,
        jurisdiction: str,
        holding: str = "",
        key_quotes: List[str] = None,
        **kwargs
    ) -> LegalCitation:
        """Add citation to database"""
        citation_id = str(uuid.uuid4())
        short_citation = f"{case_name.split(' v. ')[0] if ' v. ' in case_name else case_name.split()[0]}, {volume} {reporter} at {page}"
        
        citation = LegalCitation(
            citation_id=citation_id,
            full_citation=full_citation,
            short_citation=short_citation,
            case_name=case_name,
            reporter=reporter,
            volume=volume,
            page=page,
            year=year,
            court=court,
            jurisdiction=JurisdictionType(jurisdiction) if jurisdiction in JurisdictionType._value2member_map_ else JurisdictionType.FEDERAL,
            holding=holding,
            key_quotes=key_quotes or [],
            **kwargs
        )
        
        self.citations[citation_id] = citation
        return citation
    
    def parse_citation(self, citation_string: str) -> Optional[Dict[str, Any]]:
        """Parse citation string into components"""
        import re
        
        # Common citation patterns
        patterns = [
            # Federal Reporter: "123 F.3d 456 (3d Cir. 2020)"
            r"(\d+)\s+(F\.\d?d|F\.\s?Supp\.\s?\d?d?)\s+(\d+)\s*\((.+?)\s+(\d{4})\)",
            # U.S. Reports: "123 U.S. 456 (1900)"
            r"(\d+)\s+U\.S\.\s+(\d+)\s*\((\d{4})\)",
            # S.Ct.: "123 S.Ct. 456 (2020)"
            r"(\d+)\s+S\.\s?Ct\.\s+(\d+)\s*\((\d{4})\)",
            # State reporters: "123 N.J. 456 (2020)"
            r"(\d+)\s+(N\.J\.|N\.J\.\s?Super\.)\s+(\d+)\s*\((.+?)\s*(\d{4})\)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, citation_string)
            if match:
                groups = match.groups()
                
                # Parse based on pattern matched
                if "U.S." in citation_string and len(groups) == 3:
                    return {
                        "volume": groups[0],
                        "reporter": "U.S.",
                        "page": groups[1],
                        "year": int(groups[2]),
                        "court": "Supreme Court of the United States"
                    }
                elif "F." in citation_string:
                    return {
                        "volume": groups[0],
                        "reporter": groups[1],
                        "page": groups[2],
                        "court": groups[3],
                        "year": int(groups[4])
                    }
                elif "N.J." in citation_string:
                    return {
                        "volume": groups[0],
                        "reporter": groups[1],
                        "page": groups[2],
                        "court": groups[3] if len(groups) > 3 else "New Jersey",
                        "year": int(groups[-1])
                    }
        
        return None
    
    async def check_citation_status(
        self,
        citation_id: str
    ) -> Dict[str, Any]:
        """
        Check current status of citation (simulated Shepardizing).
        
        In production, this would integrate with Westlaw/LexisNexis APIs.
        """
        citation = self.citations.get(citation_id)
        if not citation:
            return {"error": "Citation not found"}
        
        # For now, use AI to analyze likely status based on case details
        if self.client:
            prompt = f"""Analyze the current legal status of this case:

Case: {citation.case_name}
Citation: {citation.full_citation}
Year: {citation.year}
Court: {citation.court}
Holding: {citation.holding}

Based on your legal knowledge, assess:
1. Is this case still good law?
2. Has it been overruled, limited, or questioned?
3. What are the key cases that cite or modify this holding?
4. Any circuit splits or areas of uncertainty?

Provide a JSON response with:
- status: good_law/caution/questioned/overruled
- treatment_summary: dict of treatment types and counts
- key_citing_cases: list of important subsequent cases
- notes: analysis summary"""

            try:
                response = await self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a legal research expert analyzing case authority."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                
                # Update citation status
                status_map = {
                    "good_law": CaseStatus.GOOD_LAW,
                    "caution": CaseStatus.CAUTION,
                    "questioned": CaseStatus.QUESTIONED,
                    "overruled": CaseStatus.OVERRULED
                }
                
                new_status = status_map.get(result.get("status", "").lower(), CaseStatus.GOOD_LAW)
                citation.status = new_status
                citation.status_date = datetime.now().isoformat()
                citation.status_source = "AI Analysis"
                
                return {
                    "citation_id": citation_id,
                    "case_name": citation.case_name,
                    "status": new_status.value,
                    **result
                }
                
            except Exception as e:
                return {
                    "citation_id": citation_id,
                    "status": citation.status.value,
                    "error": f"Analysis failed: {str(e)}"
                }
        
        return {
            "citation_id": citation_id,
            "status": citation.status.value,
            "note": "Live status checking requires research database integration"
        }
    
    # ========================================
    # RESEARCH MEMOS
    # ========================================
    
    def create_research_memo(
        self,
        matter_id: str,
        author: str,
        question_presented: str,
        brief_answer: str = "",
        facts: str = "",
        discussion: str = "",
        conclusion: str = "",
        jurisdiction: str = "federal"
    ) -> ResearchMemo:
        """Create new research memo"""
        memo_id = str(uuid.uuid4())
        
        memo = ResearchMemo(
            memo_id=memo_id,
            matter_id=matter_id,
            author=author,
            date=datetime.now().strftime("%Y-%m-%d"),
            question_presented=question_presented,
            brief_answer=brief_answer,
            facts=facts,
            discussion=discussion,
            conclusion=conclusion,
            jurisdiction=JurisdictionType(jurisdiction) if jurisdiction in JurisdictionType._value2member_map_ else JurisdictionType.FEDERAL
        )
        
        self.memos[memo_id] = memo
        return memo
    
    async def generate_research_memo(
        self,
        matter_id: str,
        author: str,
        question: str,
        facts: str,
        jurisdiction: str = "federal"
    ) -> ResearchMemo:
        """
        AI-assisted research memo generation.
        
        Generates a draft memo with relevant case analysis.
        """
        memo_id = str(uuid.uuid4())
        
        # Get relevant citations from database
        relevant_citations = self._find_relevant_citations(question, jurisdiction)
        
        if self.client:
            # Build citations context
            citations_context = "\n".join([
                f"- {c.case_name}, {c.full_citation}: {c.holding}"
                for c in relevant_citations[:10]
            ])
            
            prompt = f"""Draft a legal research memorandum addressing:

QUESTION PRESENTED:
{question}

RELEVANT FACTS:
{facts}

JURISDICTION: {jurisdiction}

POTENTIALLY RELEVANT AUTHORITIES:
{citations_context}

Provide a complete research memo with:
1. Brief Answer (2-3 sentences)
2. Discussion (analyze the law and apply to facts, cite specific cases)
3. Conclusion (actionable recommendation)

Format as JSON with fields: brief_answer, discussion, conclusion"""

            try:
                response = await self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert legal researcher drafting a formal research memorandum. Be thorough, cite cases accurately, and provide clear analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.4,
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                
                memo = ResearchMemo(
                    memo_id=memo_id,
                    matter_id=matter_id,
                    author=author,
                    date=datetime.now().strftime("%Y-%m-%d"),
                    question_presented=question,
                    brief_answer=result.get("brief_answer", ""),
                    facts=facts,
                    discussion=result.get("discussion", ""),
                    conclusion=result.get("conclusion", ""),
                    jurisdiction=JurisdictionType(jurisdiction) if jurisdiction in JurisdictionType._value2member_map_ else JurisdictionType.FEDERAL,
                    authorities=[c.citation_id for c in relevant_citations[:10]]
                )
                
                self.memos[memo_id] = memo
                return memo
                
            except Exception as e:
                # Return empty memo on error
                pass
        
        # Fallback: return empty memo structure
        memo = ResearchMemo(
            memo_id=memo_id,
            matter_id=matter_id,
            author=author,
            date=datetime.now().strftime("%Y-%m-%d"),
            question_presented=question,
            brief_answer="[AI generation unavailable]",
            facts=facts,
            discussion="",
            conclusion=""
        )
        
        self.memos[memo_id] = memo
        return memo
    
    def _find_relevant_citations(
        self,
        question: str,
        jurisdiction: str
    ) -> List[LegalCitation]:
        """Find relevant citations based on question keywords"""
        question_lower = question.lower()
        relevant = []
        
        # Keywords to match
        civil_rights_keywords = [
            "excessive force", "fourth amendment", "qualified immunity",
            "municipal liability", "monell", "§ 1983", "section 1983",
            "brady", "giglio", "due process", "seizure", "arrest"
        ]
        
        for citation in self.citations.values():
            score = 0
            
            # Check holding for keyword matches
            holding_lower = citation.holding.lower()
            for keyword in civil_rights_keywords:
                if keyword in question_lower and keyword in holding_lower:
                    score += 2
                elif keyword in holding_lower:
                    score += 1
            
            # Jurisdiction match
            if jurisdiction == "federal" and citation.jurisdiction in [
                JurisdictionType.SCOTUS, JurisdictionType.FEDERAL, JurisdictionType.THIRD_CIRCUIT
            ]:
                score += 1
            
            if score > 0:
                relevant.append((citation, score))
        
        # Sort by relevance score
        relevant.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in relevant]
    
    # ========================================
    # ISSUE TRACKING
    # ========================================
    
    def create_issue(
        self,
        matter_id: str,
        issue_description: str,
        elements: List[str] = None
    ) -> KeyIssue:
        """Create key legal issue to track"""
        issue_id = str(uuid.uuid4())
        
        issue = KeyIssue(
            issue_id=issue_id,
            matter_id=matter_id,
            issue_description=issue_description,
            elements=elements or []
        )
        
        self.issues[issue_id] = issue
        return issue
    
    def add_authority_to_issue(
        self,
        issue_id: str,
        citation_id: str,
        favorable: bool,
        distinguishing_factors: List[str] = None
    ):
        """Add authority to issue tracking"""
        issue = self.issues.get(issue_id)
        if not issue:
            return
        
        if favorable:
            if citation_id not in issue.favorable_cases:
                issue.favorable_cases.append(citation_id)
        else:
            if citation_id not in issue.unfavorable_cases:
                issue.unfavorable_cases.append(citation_id)
        
        if distinguishing_factors:
            issue.distinguishing_factors.extend(distinguishing_factors)
    
    async def analyze_issue_strength(
        self,
        issue_id: str
    ) -> Dict[str, Any]:
        """Analyze strength of position on issue"""
        issue = self.issues.get(issue_id)
        if not issue:
            return {"error": "Issue not found"}
        
        # Gather case information
        favorable_cases = [
            self.citations.get(cid) for cid in issue.favorable_cases
            if self.citations.get(cid)
        ]
        unfavorable_cases = [
            self.citations.get(cid) for cid in issue.unfavorable_cases
            if self.citations.get(cid)
        ]
        
        if self.client:
            prompt = f"""Analyze the strength of this legal position:

ISSUE:
{issue.issue_description}

ELEMENTS TO PROVE:
{chr(10).join(f"- {e}" for e in issue.elements)}

FAVORABLE AUTHORITIES:
{chr(10).join(f"- {c.case_name}: {c.holding}" for c in favorable_cases)}

UNFAVORABLE AUTHORITIES:
{chr(10).join(f"- {c.case_name}: {c.holding}" for c in unfavorable_cases)}

DISTINGUISHING FACTORS:
{chr(10).join(f"- {f}" for f in issue.distinguishing_factors)}

Analyze:
1. Overall strength (strong/moderate/weak)
2. Key strengths of the position
3. Key weaknesses/vulnerabilities
4. Recommended arguments
5. Anticipated counterarguments
6. Cases that need distinguishing

Respond in JSON format."""

            try:
                response = await self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert litigation strategist analyzing legal positions."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                
                # Update issue assessment
                issue.assessment = result.get("overall_strength", "uncertain")
                
                return {
                    "issue_id": issue_id,
                    "issue": issue.issue_description,
                    **result
                }
                
            except Exception as e:
                return {"error": str(e)}
        
        # Basic analysis without AI
        favorable_count = len(favorable_cases)
        unfavorable_count = len(unfavorable_cases)
        
        if favorable_count > unfavorable_count * 2:
            assessment = "strong"
        elif favorable_count > unfavorable_count:
            assessment = "moderate"
        else:
            assessment = "weak"
        
        issue.assessment = assessment
        
        return {
            "issue_id": issue_id,
            "overall_strength": assessment,
            "favorable_case_count": favorable_count,
            "unfavorable_case_count": unfavorable_count,
            "note": "AI analysis unavailable - basic assessment only"
        }
    
    # ========================================
    # MOTION BANK
    # ========================================
    
    def add_motion_template(
        self,
        motion_type: str,
        title: str,
        jurisdiction: str,
        court: str,
        template_text: str,
        sample_arguments: List[str] = None,
        key_cases: List[str] = None
    ) -> MotionBank:
        """Add motion template to bank"""
        motion_id = str(uuid.uuid4())
        
        motion = MotionBank(
            motion_id=motion_id,
            motion_type=motion_type,
            title=title,
            jurisdiction=JurisdictionType(jurisdiction) if jurisdiction in JurisdictionType._value2member_map_ else JurisdictionType.FEDERAL,
            court=court,
            template_text=template_text,
            sample_arguments=sample_arguments or [],
            key_cases=key_cases or []
        )
        
        self.motions[motion_id] = motion
        return motion
    
    def get_motion_templates(
        self,
        motion_type: str = None,
        jurisdiction: str = None
    ) -> List[MotionBank]:
        """Get motion templates matching criteria"""
        results = []
        
        for motion in self.motions.values():
            if motion_type and motion.motion_type != motion_type:
                continue
            if jurisdiction:
                try:
                    if motion.jurisdiction != JurisdictionType(jurisdiction):
                        continue
                except ValueError:
                    pass
            results.append(motion)
        
        return sorted(results, key=lambda m: m.times_used, reverse=True)
    
    def record_motion_outcome(
        self,
        motion_id: str,
        outcome: str,
        court: str,
        judge: str,
        notes: str = ""
    ):
        """Record outcome of filed motion"""
        motion = self.motions.get(motion_id)
        if motion:
            motion.times_used += 1
            motion.last_used = datetime.now().isoformat()
            motion.outcomes.append({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "outcome": outcome,
                "court": court,
                "judge": judge,
                "notes": notes
            })
            
            # Update success rate
            successful = sum(1 for o in motion.outcomes if o["outcome"] in ["granted", "granted_in_part"])
            motion.success_rate = successful / len(motion.outcomes) if motion.outcomes else None
    
    # ========================================
    # SEARCH & REPORTING
    # ========================================
    
    def search_citations(
        self,
        query: str,
        jurisdiction: str = None,
        min_year: int = None
    ) -> List[LegalCitation]:
        """Search citation database"""
        query_lower = query.lower()
        results = []
        
        for citation in self.citations.values():
            # Check jurisdiction
            if jurisdiction:
                try:
                    if citation.jurisdiction != JurisdictionType(jurisdiction):
                        continue
                except ValueError:
                    pass
            
            # Check year
            if min_year and citation.year < min_year:
                continue
            
            # Search in case name, holding, quotes
            searchable = f"{citation.case_name} {citation.holding} {' '.join(citation.key_quotes)}".lower()
            
            if query_lower in searchable:
                results.append(citation)
        
        return results
    
    def generate_authority_table(
        self,
        issue_id: str
    ) -> str:
        """Generate table of authorities for issue"""
        issue = self.issues.get(issue_id)
        if not issue:
            return "Issue not found"
        
        lines = [
            f"TABLE OF AUTHORITIES",
            f"Issue: {issue.issue_description}",
            "",
            "FAVORABLE AUTHORITIES:",
            "-" * 80
        ]
        
        for cid in issue.favorable_cases:
            citation = self.citations.get(cid)
            if citation:
                lines.append(f"  {citation.full_citation}")
                lines.append(f"    Holding: {citation.holding[:100]}...")
                lines.append("")
        
        lines.extend([
            "UNFAVORABLE AUTHORITIES:",
            "-" * 80
        ])
        
        for cid in issue.unfavorable_cases:
            citation = self.citations.get(cid)
            if citation:
                lines.append(f"  {citation.full_citation}")
                lines.append(f"    Holding: {citation.holding[:100]}...")
                lines.append("")
        
        return "\n".join(lines)


# Singleton instance
research_service = LegalResearchService()
