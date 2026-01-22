"""
BarberX Legal Case Management Pro Suite
Deposition Analysis & Witness Management Service
"""
import os
import re
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import uuid

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class WitnessType(str, Enum):
    """Types of witnesses"""
    FACT = "fact"
    EXPERT = "expert"
    CHARACTER = "character"
    EYEWITNESS = "eyewitness"
    DEFENDANT = "defendant"
    PLAINTIFF = "plaintiff"
    POLICE_OFFICER = "police_officer"
    MEDICAL = "medical"
    CORPORATE_REP = "corporate_representative"


class CredibilityFactor(str, Enum):
    """Factors affecting witness credibility"""
    PRIOR_INCONSISTENT = "prior_inconsistent_statement"
    BIAS = "bias"
    MOTIVE = "motive"
    PERCEPTION = "perception_issues"
    MEMORY = "memory_issues"
    CHARACTER_TRUTHFULNESS = "character_for_untruthfulness"
    CRIMINAL_CONVICTION = "criminal_conviction"
    CONTRADICTION = "internal_contradiction"
    CORROBORATION = "corroboration"


@dataclass
class WitnessProfile:
    """Comprehensive witness profile"""
    witness_id: str
    name: str
    witness_type: WitnessType
    
    # Contact
    address: str = ""
    phone: str = ""
    email: str = ""
    employer: str = ""
    occupation: str = ""
    
    # Relationship to case
    relationship: str = ""  # How they're connected
    favorable_to: str = ""  # plaintiff, defendant, neutral
    
    # Interview/Deposition status
    interviewed: bool = False
    interview_date: Optional[str] = None
    interview_notes: str = ""
    deposed: bool = False
    deposition_date: Optional[str] = None
    deposition_transcript_id: Optional[str] = None
    
    # Credibility assessment
    credibility_score: float = 0.0  # 0-100
    credibility_factors: List[Dict[str, Any]] = field(default_factory=list)
    
    # Prior statements
    prior_statements: List[Dict[str, Any]] = field(default_factory=list)
    
    # Availability
    available_for_trial: bool = True
    availability_notes: str = ""
    subpoena_required: bool = False
    
    # Notes
    attorney_notes: str = ""
    created_at: str = ""
    updated_at: str = ""


@dataclass
class DepositionDigest:
    """Deposition digest/summary"""
    digest_id: str
    deposition_id: str
    witness_name: str
    deposition_date: str
    
    # Summary
    executive_summary: str = ""
    key_admissions: List[Dict[str, Any]] = field(default_factory=list)
    key_denials: List[Dict[str, Any]] = field(default_factory=list)
    
    # Topics covered
    topics: List[Dict[str, Any]] = field(default_factory=list)
    
    # Impeachment material
    inconsistencies: List[Dict[str, Any]] = field(default_factory=list)
    prior_inconsistent_statements: List[Dict[str, Any]] = field(default_factory=list)
    
    # Exhibits
    exhibits_discussed: List[str] = field(default_factory=list)
    
    # Page/line references
    key_testimony: List[Dict[str, Any]] = field(default_factory=list)
    
    # Credibility issues
    credibility_concerns: List[str] = field(default_factory=list)
    
    # Follow-up needed
    follow_up_questions: List[str] = field(default_factory=list)
    additional_discovery_needed: List[str] = field(default_factory=list)
    
    created_at: str = ""


@dataclass
class ImpeachmentMaterial:
    """Material for impeaching witness"""
    material_id: str
    witness_id: str
    impeachment_type: CredibilityFactor
    
    # Prior statement
    prior_statement: str = ""
    prior_statement_source: str = ""  # deposition, interview, document
    prior_statement_date: str = ""
    prior_statement_page_line: str = ""  # "45:12-46:3"
    
    # Current/contradicting statement
    current_statement: str = ""
    current_statement_source: str = ""
    current_statement_page_line: str = ""
    
    # Analysis
    contradiction_analysis: str = ""
    severity: str = "minor"  # minor, moderate, severe
    
    # Trial use
    recommended_approach: str = ""
    exhibit_needed: bool = False
    exhibit_id: Optional[str] = None


@dataclass
class ExpertWitness:
    """Expert witness profile with Daubert analysis"""
    expert_id: str
    name: str
    specialty: str
    
    # Qualifications
    education: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    publications: List[str] = field(default_factory=list)
    prior_testimony_count: int = 0
    cv_document_id: Optional[str] = None
    
    # Engagement
    retained_by: str = ""  # plaintiff, defendant
    fee_schedule: Dict[str, float] = field(default_factory=dict)
    report_due_date: Optional[str] = None
    report_document_id: Optional[str] = None
    
    # Daubert factors
    daubert_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Prior testimony
    prior_cases: List[Dict[str, Any]] = field(default_factory=list)
    excluded_cases: List[Dict[str, Any]] = field(default_factory=list)  # Cases where excluded
    
    # Opinions
    expected_opinions: List[str] = field(default_factory=list)
    opinion_basis: List[str] = field(default_factory=list)
    
    # Rebuttal points
    vulnerabilities: List[str] = field(default_factory=list)
    rebuttal_points: List[str] = field(default_factory=list)


class DepositionService:
    """
    Premium Deposition Analysis & Witness Management Service
    
    Features:
    - Automated deposition digest generation
    - Impeachment material tracking
    - Prior inconsistent statement detection
    - Witness credibility scoring
    - Expert witness Daubert analysis
    - Cross-examination preparation
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.client = AsyncOpenAI(api_key=self.api_key) if OPENAI_AVAILABLE and self.api_key else None
        
        # Data storage
        self.witnesses: Dict[str, WitnessProfile] = {}
        self.depositions: Dict[str, DepositionDigest] = {}
        self.experts: Dict[str, ExpertWitness] = {}
        self.impeachment_materials: Dict[str, List[ImpeachmentMaterial]] = {}
    
    # ============================================================
    # WITNESS MANAGEMENT
    # ============================================================
    
    def create_witness(
        self,
        name: str,
        witness_type: WitnessType,
        **kwargs
    ) -> WitnessProfile:
        """Create new witness profile"""
        witness = WitnessProfile(
            witness_id=f"wit_{uuid.uuid4().hex[:8]}",
            name=name,
            witness_type=witness_type,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            **kwargs
        )
        self.witnesses[witness.witness_id] = witness
        return witness
    
    def add_prior_statement(
        self,
        witness_id: str,
        statement: str,
        source: str,
        date: str,
        page_line: str = ""
    ):
        """Add prior statement to witness record"""
        if witness_id not in self.witnesses:
            raise ValueError(f"Witness not found: {witness_id}")
        
        self.witnesses[witness_id].prior_statements.append({
            "statement": statement,
            "source": source,
            "date": date,
            "page_line": page_line,
            "added_at": datetime.utcnow().isoformat()
        })
    
    async def assess_credibility(
        self,
        witness_id: str,
        all_statements: List[Dict[str, Any]],
        case_context: str
    ) -> Dict[str, Any]:
        """
        AI-powered witness credibility assessment.
        
        Analyzes:
        - Internal consistency
        - Consistency with other evidence
        - Bias indicators
        - Memory/perception issues
        - Motive to fabricate
        """
        witness = self.witnesses.get(witness_id)
        if not witness:
            raise ValueError(f"Witness not found: {witness_id}")
        
        if not self.client:
            return self._basic_credibility_check(witness, all_statements)
        
        statements_text = "\n".join(
            f"- [{s.get('source', 'Unknown')} {s.get('date', '')}]: {s.get('statement', '')}"
            for s in all_statements
        )
        
        prompt = f"""Assess the credibility of this witness for litigation purposes.

WITNESS: {witness.name}
TYPE: {witness.witness_type.value}
RELATIONSHIP TO CASE: {witness.relationship}

CASE CONTEXT:
{case_context}

STATEMENTS FROM THIS WITNESS:
{statements_text}

Analyze for credibility factors under Federal Rules of Evidence:
1. Prior inconsistent statements (FRE 613)
2. Bias, prejudice, interest (FRE 611)
3. Character for untruthfulness (FRE 608)
4. Perception/memory issues
5. Motive to fabricate
6. Internal contradictions

Provide analysis in JSON:
{{
    "credibility_score": 0-100,
    "overall_assessment": "highly credible|credible|questionable|not credible",
    "strengths": ["credibility strengths"],
    "weaknesses": ["credibility weaknesses"],
    "inconsistencies": [
        {{
            "statement_1": "first statement",
            "statement_2": "contradicting statement",
            "severity": "minor|moderate|severe",
            "explanation": "why this is problematic"
        }}
    ],
    "bias_indicators": ["potential biases"],
    "perception_issues": ["issues affecting what they could perceive"],
    "memory_concerns": ["factors affecting memory reliability"],
    "motive_analysis": "analysis of potential motive to fabricate",
    "impeachment_opportunities": ["ways to impeach this witness"],
    "rehabilitation_opportunities": ["ways to rehabilitate if our witness"],
    "recommended_approach": "how to handle this witness at trial",
    "key_admissions_to_lock_in": ["important concessions to get"],
    "areas_to_avoid": ["topics that may hurt our case"]
}}"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert litigation consultant specializing in witness credibility assessment and trial preparation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            # Update witness profile
            witness.credibility_score = result.get('credibility_score', 0)
            witness.credibility_factors = result.get('weaknesses', []) + result.get('inconsistencies', [])
            witness.updated_at = datetime.utcnow().isoformat()
            
            return result
            
        except Exception as e:
            return {"error": str(e)}
    
    def _basic_credibility_check(
        self,
        witness: WitnessProfile,
        statements: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Basic credibility check without AI"""
        score = 70  # Base score
        issues = []
        
        # Check for multiple conflicting statements
        if len(statements) > 1:
            # Simple word overlap check for inconsistency
            for i, s1 in enumerate(statements):
                for s2 in statements[i+1:]:
                    # Very basic contradiction detection
                    if "not" in s1.get('statement', '').lower() and "not" not in s2.get('statement', '').lower():
                        score -= 10
                        issues.append("Potential inconsistency detected")
        
        # Bias based on relationship
        if witness.favorable_to in ["plaintiff", "defendant"]:
            score -= 5
            issues.append(f"Potential bias toward {witness.favorable_to}")
        
        return {
            "credibility_score": max(0, score),
            "issues": issues,
            "statement_count": len(statements)
        }
    
    # ============================================================
    # DEPOSITION ANALYSIS
    # ============================================================
    
    async def generate_deposition_digest(
        self,
        transcript_text: str,
        witness_name: str,
        deposition_date: str,
        case_issues: List[str]
    ) -> DepositionDigest:
        """
        Generate comprehensive deposition digest.
        
        Creates:
        - Executive summary
        - Topic-by-topic breakdown
        - Key admissions/denials
        - Impeachment material
        - Follow-up questions needed
        """
        digest_id = f"digest_{uuid.uuid4().hex[:8]}"
        deposition_id = f"depo_{uuid.uuid4().hex[:8]}"
        
        if not self.client:
            return DepositionDigest(
                digest_id=digest_id,
                deposition_id=deposition_id,
                witness_name=witness_name,
                deposition_date=deposition_date,
                executive_summary="AI analysis not available",
                created_at=datetime.utcnow().isoformat()
            )
        
        issues_text = "\n".join(f"- {issue}" for issue in case_issues)
        
        prompt = f"""Analyze this deposition transcript and create a comprehensive digest.

WITNESS: {witness_name}
DATE: {deposition_date}

CASE ISSUES:
{issues_text}

TRANSCRIPT:
{transcript_text[:15000]}

Create a detailed deposition digest in JSON format:
{{
    "executive_summary": "2-3 paragraph summary of key testimony",
    "key_admissions": [
        {{
            "topic": "subject matter",
            "admission": "what was admitted",
            "page_line": "estimated page:line",
            "significance": "why this matters",
            "verbatim_quote": "exact quote if identifiable"
        }}
    ],
    "key_denials": [
        {{
            "topic": "subject matter",
            "denial": "what was denied",
            "page_line": "estimated page:line",
            "can_be_contradicted": true/false,
            "contradicting_evidence": "evidence that contradicts this"
        }}
    ],
    "topics_covered": [
        {{
            "topic": "topic name",
            "summary": "summary of testimony on this topic",
            "page_range": "approximate pages",
            "favorable": true/false,
            "key_quotes": ["important quotes"]
        }}
    ],
    "inconsistencies": [
        {{
            "description": "what is inconsistent",
            "statement_1": "first statement",
            "statement_1_location": "page:line",
            "statement_2": "contradicting statement", 
            "statement_2_location": "page:line",
            "severity": "minor|moderate|severe"
        }}
    ],
    "exhibits_discussed": ["list of exhibits referenced"],
    "credibility_concerns": ["issues affecting credibility"],
    "follow_up_questions": ["questions that should have been asked or need follow-up"],
    "additional_discovery_needed": ["documents or information to pursue"],
    "trial_testimony_predictions": ["how this witness will likely testify at trial"],
    "cross_examination_points": ["points to hit on cross-examination"]
}}"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a senior litigation associate preparing deposition digests for trial preparation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            digest = DepositionDigest(
                digest_id=digest_id,
                deposition_id=deposition_id,
                witness_name=witness_name,
                deposition_date=deposition_date,
                executive_summary=result.get('executive_summary', ''),
                key_admissions=result.get('key_admissions', []),
                key_denials=result.get('key_denials', []),
                topics=result.get('topics_covered', []),
                inconsistencies=result.get('inconsistencies', []),
                exhibits_discussed=result.get('exhibits_discussed', []),
                credibility_concerns=result.get('credibility_concerns', []),
                follow_up_questions=result.get('follow_up_questions', []),
                additional_discovery_needed=result.get('additional_discovery_needed', []),
                created_at=datetime.utcnow().isoformat()
            )
            
            self.depositions[digest_id] = digest
            return digest
            
        except Exception as e:
            return DepositionDigest(
                digest_id=digest_id,
                deposition_id=deposition_id,
                witness_name=witness_name,
                deposition_date=deposition_date,
                executive_summary=f"Error generating digest: {str(e)}",
                created_at=datetime.utcnow().isoformat()
            )
    
    # ============================================================
    # IMPEACHMENT TRACKING
    # ============================================================
    
    def create_impeachment_material(
        self,
        witness_id: str,
        impeachment_type: CredibilityFactor,
        prior_statement: str,
        prior_source: str,
        current_statement: str,
        current_source: str,
        **kwargs
    ) -> ImpeachmentMaterial:
        """Create impeachment material record"""
        material = ImpeachmentMaterial(
            material_id=f"imp_{uuid.uuid4().hex[:8]}",
            witness_id=witness_id,
            impeachment_type=impeachment_type,
            prior_statement=prior_statement,
            prior_statement_source=prior_source,
            current_statement=current_statement,
            current_statement_source=current_source,
            **kwargs
        )
        
        if witness_id not in self.impeachment_materials:
            self.impeachment_materials[witness_id] = []
        self.impeachment_materials[witness_id].append(material)
        
        return material
    
    async def find_impeachment_opportunities(
        self,
        witness_id: str,
        deposition_text: str,
        prior_statements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """AI-powered impeachment opportunity detection"""
        if not self.client:
            return []
        
        prior_text = "\n".join(
            f"[{s.get('source', '')} - {s.get('date', '')}]: {s.get('statement', '')}"
            for s in prior_statements
        )
        
        prompt = f"""Compare this deposition testimony against prior statements to identify impeachment opportunities.

PRIOR STATEMENTS:
{prior_text}

DEPOSITION TESTIMONY:
{deposition_text[:10000]}

Find all inconsistencies that could be used for impeachment under FRE 613.

Return JSON:
{{
    "impeachment_opportunities": [
        {{
            "prior_statement": "exact prior statement",
            "prior_source": "source of prior statement",
            "deposition_statement": "contradicting deposition testimony",
            "deposition_location": "estimated page:line",
            "type_of_inconsistency": "direct_contradiction|omission|embellishment|timeline_discrepancy",
            "severity": "minor|moderate|severe",
            "recommended_approach": "how to use this at trial",
            "setup_questions": ["questions to ask before confronting"],
            "confrontation_script": "suggested language for confrontation"
        }}
    ],
    "summary": "overall assessment of impeachment potential"
}}"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a trial lawyer expert in witness impeachment techniques."},
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
    # EXPERT WITNESS MANAGEMENT
    # ============================================================
    
    def create_expert_witness(
        self,
        name: str,
        specialty: str,
        retained_by: str,
        **kwargs
    ) -> ExpertWitness:
        """Create expert witness profile"""
        expert = ExpertWitness(
            expert_id=f"exp_{uuid.uuid4().hex[:8]}",
            name=name,
            specialty=specialty,
            retained_by=retained_by,
            **kwargs
        )
        self.experts[expert.expert_id] = expert
        return expert
    
    async def analyze_daubert_factors(
        self,
        expert_id: str,
        expert_report: str,
        methodology_description: str
    ) -> Dict[str, Any]:
        """
        Analyze expert testimony under Daubert factors.
        
        Factors:
        1. Testing - Has the theory/technique been tested?
        2. Peer Review - Has it been subjected to peer review?
        3. Error Rate - What is the known/potential error rate?
        4. Standards - Are there standards controlling the technique?
        5. General Acceptance - Is it generally accepted in the field?
        """
        if not self.client:
            return {"error": "AI analysis not available"}
        
        prompt = f"""Analyze this expert's methodology under Daubert v. Merrell Dow factors.

EXPERT REPORT SUMMARY:
{expert_report[:5000]}

METHODOLOGY DESCRIPTION:
{methodology_description}

Analyze under Daubert/Kumho Tire factors and provide JSON:
{{
    "overall_admissibility": "likely_admissible|questionable|likely_excludable",
    "daubert_factors": {{
        "testability": {{
            "score": 1-5,
            "analysis": "can the theory/technique be tested?",
            "evidence": "supporting evidence from report"
        }},
        "peer_review": {{
            "score": 1-5,
            "analysis": "has methodology been peer reviewed/published?",
            "evidence": "citations or lack thereof"
        }},
        "error_rate": {{
            "score": 1-5,
            "analysis": "known or potential error rate",
            "rate_if_stated": "error rate if mentioned"
        }},
        "standards": {{
            "score": 1-5,
            "analysis": "existence of controlling standards",
            "standards_cited": ["standards mentioned"]
        }},
        "general_acceptance": {{
            "score": 1-5,
            "analysis": "acceptance in relevant scientific community",
            "evidence": "supporting evidence"
        }}
    }},
    "methodology_concerns": ["specific concerns about methodology"],
    "foundation_gaps": ["areas where foundation is lacking"],
    "cross_examination_points": ["points to attack on cross"],
    "motion_in_limine_arguments": ["arguments for exclusion"],
    "voir_dire_questions": ["questions to ask in voir dire of expert"],
    "recommended_response": "how to respond if opposing expert"
}}"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert in Daubert analysis and expert witness qualification."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            # Update expert profile
            expert = self.experts.get(expert_id)
            if expert:
                expert.daubert_analysis = result
            
            return result
            
        except Exception as e:
            return {"error": str(e)}
    
    # ============================================================
    # CROSS-EXAMINATION PREPARATION
    # ============================================================
    
    async def generate_cross_examination_outline(
        self,
        witness_id: str,
        deposition_digest: DepositionDigest,
        case_theory: str,
        goals: List[str]
    ) -> Dict[str, Any]:
        """Generate cross-examination outline for trial"""
        witness = self.witnesses.get(witness_id)
        if not witness:
            raise ValueError(f"Witness not found: {witness_id}")
        
        if not self.client:
            return {"error": "AI analysis not available"}
        
        goals_text = "\n".join(f"- {g}" for g in goals)
        admissions_text = "\n".join(
            f"- {a.get('admission', '')} (p. {a.get('page_line', 'N/A')})"
            for a in deposition_digest.key_admissions
        )
        
        prompt = f"""Create a cross-examination outline for trial.

WITNESS: {witness.name}
TYPE: {witness.witness_type.value}
FAVORABLE TO: {witness.favorable_to}

CASE THEORY:
{case_theory}

CROSS-EXAMINATION GOALS:
{goals_text}

KEY ADMISSIONS FROM DEPOSITION:
{admissions_text}

CREDIBILITY CONCERNS:
{', '.join(deposition_digest.credibility_concerns)}

Create a detailed cross-examination outline in JSON:
{{
    "opening_strategy": "how to begin cross",
    "primacy_points": ["most important points to establish first"],
    "recency_points": ["most important points to end with"],
    "topic_sequences": [
        {{
            "topic": "topic name",
            "goal": "what to accomplish",
            "questions": [
                {{
                    "question": "the question",
                    "expected_answer": "likely answer",
                    "if_denies": "follow-up if witness denies",
                    "impeachment_ready": true/false,
                    "exhibit_needed": "exhibit number if needed"
                }}
            ],
            "transition": "how to move to next topic"
        }}
    ],
    "impeachment_sequences": [
        {{
            "target_statement": "statement to impeach",
            "setup_questions": ["questions to lock witness in"],
            "confrontation": "how to confront with prior statement",
            "prior_statement_source": "deposition page or document"
        }}
    ],
    "damage_control_topics": ["topics to avoid or minimize"],
    "redirect_predictions": ["likely redirect examination points"],
    "re-cross_preparation": ["points to hit on re-cross"]
}}"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are Irving Younger teaching cross-examination. Create precise, controlled cross-examination questions that only allow yes/no answers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            import json
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            return {"error": str(e)}


# Service instance
deposition_service = DepositionService()
