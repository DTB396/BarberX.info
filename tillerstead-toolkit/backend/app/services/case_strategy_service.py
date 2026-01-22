"""
BarberX Legal Case Management Pro Suite
Case Strategy & Predictive Analytics Service
"""
import os
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class CaseOutcome(str, Enum):
    """Possible case outcomes"""
    PLAINTIFF_VERDICT = "plaintiff_verdict"
    DEFENSE_VERDICT = "defense_verdict"
    SETTLEMENT = "settlement"
    DISMISSAL = "dismissal"
    SUMMARY_JUDGMENT_PLAINTIFF = "summary_judgment_plaintiff"
    SUMMARY_JUDGMENT_DEFENDANT = "summary_judgment_defendant"
    DIRECTED_VERDICT = "directed_verdict"
    MISTRIAL = "mistrial"


class RiskLevel(str, Enum):
    """Risk assessment levels"""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


@dataclass
class LiabilityAssessment:
    """Liability assessment for each defendant/claim"""
    assessment_id: str
    defendant_name: str
    claim_type: str
    
    # Liability analysis
    liability_probability: float  # 0-100%
    liability_factors: List[Dict[str, Any]] = field(default_factory=list)
    
    # Defenses
    viable_defenses: List[Dict[str, Any]] = field(default_factory=list)
    defense_strength: float = 0.0
    
    # Comparative fault
    comparative_fault_applicable: bool = False
    estimated_fault_percentage: float = 0.0
    
    # Immunity
    immunity_applicable: bool = False
    immunity_type: str = ""
    immunity_analysis: str = ""
    
    # Overall assessment
    risk_level: RiskLevel = RiskLevel.MODERATE
    recommended_action: str = ""
    
    created_at: str = ""


@dataclass
class DamagesAnalysis:
    """Comprehensive damages analysis"""
    analysis_id: str
    case_id: str
    
    # Economic damages
    economic_damages: Dict[str, float] = field(default_factory=dict)
    economic_total: float = 0.0
    economic_documentation_strength: float = 0.0
    
    # Non-economic damages
    non_economic_damages: Dict[str, float] = field(default_factory=dict)
    non_economic_total: float = 0.0
    pain_suffering_multiplier: float = 1.0
    
    # Punitive damages
    punitive_applicable: bool = False
    punitive_analysis: str = ""
    punitive_estimate_low: float = 0.0
    punitive_estimate_high: float = 0.0
    
    # Caps and limitations
    damage_caps_applicable: bool = False
    applicable_caps: List[Dict[str, Any]] = field(default_factory=list)
    
    # Totals
    total_exposure_low: float = 0.0
    total_exposure_high: float = 0.0
    most_likely_verdict: float = 0.0
    
    # Mitigation
    mitigation_issues: List[str] = field(default_factory=list)
    
    created_at: str = ""


@dataclass
class SettlementAnalysis:
    """Settlement valuation and negotiation analysis"""
    analysis_id: str
    case_id: str
    
    # Valuation
    settlement_value_low: float = 0.0
    settlement_value_high: float = 0.0
    recommended_settlement_range: Tuple[float, float] = (0.0, 0.0)
    
    # Factors
    liability_discount: float = 0.0  # Reduction for liability risk
    collectability_factor: float = 1.0  # Can defendant pay?
    litigation_costs_to_trial: float = 0.0
    time_value_discount: float = 0.0
    
    # Negotiation
    opening_demand: float = 0.0
    target_settlement: float = 0.0
    walk_away_point: float = 0.0
    
    # Strategy
    negotiation_strategy: str = ""
    leverage_points: List[str] = field(default_factory=list)
    weaknesses_to_address: List[str] = field(default_factory=list)
    
    # Insurance
    policy_limits: float = 0.0
    excess_coverage: float = 0.0
    bad_faith_exposure: bool = False
    
    created_at: str = ""


@dataclass
class CaseTheory:
    """Case theory development"""
    theory_id: str
    case_id: str
    party: str  # plaintiff or defendant
    
    # Core theory
    theme: str = ""  # One-sentence case theme
    narrative: str = ""  # The story of the case
    
    # Legal framework
    causes_of_action: List[Dict[str, Any]] = field(default_factory=list)
    legal_elements: List[Dict[str, Any]] = field(default_factory=list)
    
    # Key facts
    undisputed_facts: List[str] = field(default_factory=list)
    disputed_facts: List[str] = field(default_factory=list)
    
    # Evidence mapping
    evidence_by_element: Dict[str, List[str]] = field(default_factory=dict)
    evidence_gaps: List[str] = field(default_factory=list)
    
    # Witness strategy
    witness_order: List[str] = field(default_factory=list)
    
    # Weaknesses
    vulnerabilities: List[str] = field(default_factory=list)
    responses_to_weaknesses: Dict[str, str] = field(default_factory=dict)
    
    created_at: str = ""


@dataclass
class JuryProfile:
    """Ideal juror profile for voir dire"""
    profile_id: str
    case_id: str
    
    # Demographics
    favorable_demographics: Dict[str, List[str]] = field(default_factory=dict)
    unfavorable_demographics: Dict[str, List[str]] = field(default_factory=dict)
    
    # Attitudes
    favorable_attitudes: List[str] = field(default_factory=list)
    unfavorable_attitudes: List[str] = field(default_factory=list)
    
    # Experiences
    favorable_experiences: List[str] = field(default_factory=list)
    unfavorable_experiences: List[str] = field(default_factory=list)
    
    # Voir dire questions
    must_ask_questions: List[str] = field(default_factory=list)
    follow_up_questions: Dict[str, List[str]] = field(default_factory=dict)
    
    # Challenges
    for_cause_triggers: List[str] = field(default_factory=list)
    peremptory_priorities: List[str] = field(default_factory=list)
    
    created_at: str = ""


class CaseStrategyService:
    """
    Premium Case Strategy & Predictive Analytics Service
    
    Features:
    - Liability probability assessment
    - Damages calculation with multipliers
    - Settlement valuation
    - Verdict prediction
    - Case theory development
    - Jury selection guidance
    - Motion strategy planning
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.client = AsyncOpenAI(api_key=self.api_key) if OPENAI_AVAILABLE and self.api_key else None
        
        # Storage
        self.liability_assessments: Dict[str, LiabilityAssessment] = {}
        self.damages_analyses: Dict[str, DamagesAnalysis] = {}
        self.settlement_analyses: Dict[str, SettlementAnalysis] = {}
        self.case_theories: Dict[str, CaseTheory] = {}
        self.jury_profiles: Dict[str, JuryProfile] = {}
    
    # ============================================================
    # LIABILITY ASSESSMENT
    # ============================================================
    
    async def assess_liability(
        self,
        case_facts: str,
        defendant_name: str,
        claim_type: str,
        jurisdiction: str,
        applicable_law: str
    ) -> LiabilityAssessment:
        """
        AI-powered liability assessment.
        
        Analyzes:
        - Elements of each claim
        - Evidence supporting/opposing each element
        - Available defenses
        - Immunity issues
        - Comparative fault
        """
        assessment_id = f"liab_{uuid.uuid4().hex[:8]}"
        
        if not self.client:
            return LiabilityAssessment(
                assessment_id=assessment_id,
                defendant_name=defendant_name,
                claim_type=claim_type,
                liability_probability=50.0,
                risk_level=RiskLevel.MODERATE,
                created_at=datetime.utcnow().isoformat()
            )
        
        prompt = f"""Perform a detailed liability assessment for this civil rights case.

DEFENDANT: {defendant_name}
CLAIM TYPE: {claim_type}
JURISDICTION: {jurisdiction}

APPLICABLE LAW:
{applicable_law}

CASE FACTS:
{case_facts}

Analyze liability considering:
1. Elements of the claim under {claim_type}
2. Evidence supporting each element
3. Available defenses (qualified immunity, good faith, etc.)
4. Comparative fault issues
5. Any applicable immunities

Provide analysis in JSON:
{{
    "liability_probability": 0-100,
    "elements_analysis": [
        {{
            "element": "element name",
            "element_description": "what must be proven",
            "evidence_supporting": ["supporting evidence"],
            "evidence_opposing": ["opposing evidence"],
            "element_likely_met": true/false,
            "confidence": 0-100
        }}
    ],
    "viable_defenses": [
        {{
            "defense": "defense name",
            "basis": "legal basis",
            "likelihood_of_success": 0-100,
            "key_facts_needed": ["facts that support this defense"]
        }}
    ],
    "qualified_immunity_analysis": {{
        "applicable": true/false,
        "clearly_established_analysis": "was the right clearly established?",
        "objectively_reasonable_analysis": "was conduct objectively reasonable?",
        "likely_to_defeat_claim": true/false
    }},
    "comparative_fault": {{
        "applicable": true/false,
        "plaintiff_conduct": "relevant plaintiff conduct",
        "estimated_fault_percentage": 0-100
    }},
    "risk_level": "minimal|low|moderate|high|severe",
    "recommended_action": "settlement|litigate|early_motion|trial",
    "key_discovery_needed": ["critical discovery to pursue"],
    "dispositive_motion_potential": "assessment of summary judgment chances"
}}"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a senior civil rights litigator assessing case liability."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            assessment = LiabilityAssessment(
                assessment_id=assessment_id,
                defendant_name=defendant_name,
                claim_type=claim_type,
                liability_probability=result.get('liability_probability', 50),
                liability_factors=result.get('elements_analysis', []),
                viable_defenses=result.get('viable_defenses', []),
                defense_strength=100 - result.get('liability_probability', 50),
                immunity_applicable=result.get('qualified_immunity_analysis', {}).get('applicable', False),
                immunity_analysis=str(result.get('qualified_immunity_analysis', {})),
                comparative_fault_applicable=result.get('comparative_fault', {}).get('applicable', False),
                estimated_fault_percentage=result.get('comparative_fault', {}).get('estimated_fault_percentage', 0),
                risk_level=RiskLevel(result.get('risk_level', 'moderate')),
                recommended_action=result.get('recommended_action', ''),
                created_at=datetime.utcnow().isoformat()
            )
            
            self.liability_assessments[assessment_id] = assessment
            return assessment
            
        except Exception as e:
            return LiabilityAssessment(
                assessment_id=assessment_id,
                defendant_name=defendant_name,
                claim_type=claim_type,
                liability_probability=50.0,
                risk_level=RiskLevel.MODERATE,
                recommended_action=f"Error: {str(e)}",
                created_at=datetime.utcnow().isoformat()
            )
    
    # ============================================================
    # DAMAGES ANALYSIS
    # ============================================================
    
    async def calculate_damages(
        self,
        case_id: str,
        economic_items: Dict[str, float],
        injury_description: str,
        plaintiff_age: int,
        jurisdiction: str,
        comparable_verdicts: List[Dict[str, Any]] = None
    ) -> DamagesAnalysis:
        """
        Comprehensive damages calculation.
        
        Calculates:
        - Economic damages (medical, lost wages, future losses)
        - Non-economic damages (pain & suffering, emotional distress)
        - Punitive damages potential
        - Applicable caps
        """
        analysis_id = f"dmg_{uuid.uuid4().hex[:8]}"
        
        economic_total = sum(economic_items.values())
        
        if not self.client:
            # Basic multiplier-based calculation
            non_economic_total = economic_total * 2.5  # Standard multiplier
            return DamagesAnalysis(
                analysis_id=analysis_id,
                case_id=case_id,
                economic_damages=economic_items,
                economic_total=economic_total,
                non_economic_total=non_economic_total,
                pain_suffering_multiplier=2.5,
                total_exposure_low=economic_total + non_economic_total * 0.5,
                total_exposure_high=economic_total + non_economic_total * 2,
                most_likely_verdict=economic_total + non_economic_total,
                created_at=datetime.utcnow().isoformat()
            )
        
        economic_text = "\n".join(f"- {k}: ${v:,.2f}" for k, v in economic_items.items())
        verdicts_text = ""
        if comparable_verdicts:
            verdicts_text = "\nCOMPARABLE VERDICTS:\n" + "\n".join(
                f"- {v.get('case', '')}: ${v.get('amount', 0):,.2f} ({v.get('injuries', '')})"
                for v in comparable_verdicts
            )
        
        prompt = f"""Calculate damages for this civil rights case.

JURISDICTION: {jurisdiction}
PLAINTIFF AGE: {plaintiff_age}

INJURY DESCRIPTION:
{injury_description}

ECONOMIC DAMAGES:
{economic_text}
Total Economic: ${economic_total:,.2f}

{verdicts_text}

Analyze and provide JSON:
{{
    "economic_damages_analysis": {{
        "medical_expenses_past": 0.0,
        "medical_expenses_future": 0.0,
        "lost_wages_past": 0.0,
        "lost_earning_capacity": 0.0,
        "other_economic": 0.0,
        "documentation_strength": "strong|moderate|weak",
        "economic_total": 0.0
    }},
    "non_economic_damages": {{
        "pain_and_suffering": 0.0,
        "emotional_distress": 0.0,
        "loss_of_enjoyment": 0.0,
        "disfigurement": 0.0,
        "loss_of_consortium": 0.0,
        "non_economic_total": 0.0,
        "multiplier_used": 0.0,
        "multiplier_justification": "why this multiplier"
    }},
    "punitive_damages": {{
        "applicable": true/false,
        "conduct_supporting": "description of egregious conduct",
        "estimated_range_low": 0.0,
        "estimated_range_high": 0.0,
        "ratio_to_compensatory": "typical ratio"
    }},
    "damage_caps": {{
        "applicable": true/false,
        "caps": [
            {{
                "type": "cap type",
                "amount": 0.0,
                "statute": "applicable statute"
            }}
        ]
    }},
    "verdict_prediction": {{
        "low_estimate": 0.0,
        "high_estimate": 0.0,
        "most_likely": 0.0,
        "confidence": "low|medium|high"
    }},
    "mitigation_issues": ["any failure to mitigate"],
    "jury_appeal_factors": ["factors that may increase/decrease award"]
}}"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert in damages calculation for civil rights litigation. Provide realistic damage estimates based on comparable verdicts and legal standards."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            analysis = DamagesAnalysis(
                analysis_id=analysis_id,
                case_id=case_id,
                economic_damages=result.get('economic_damages_analysis', {}),
                economic_total=result.get('economic_damages_analysis', {}).get('economic_total', economic_total),
                non_economic_damages=result.get('non_economic_damages', {}),
                non_economic_total=result.get('non_economic_damages', {}).get('non_economic_total', 0),
                pain_suffering_multiplier=result.get('non_economic_damages', {}).get('multiplier_used', 1.0),
                punitive_applicable=result.get('punitive_damages', {}).get('applicable', False),
                punitive_analysis=result.get('punitive_damages', {}).get('conduct_supporting', ''),
                punitive_estimate_low=result.get('punitive_damages', {}).get('estimated_range_low', 0),
                punitive_estimate_high=result.get('punitive_damages', {}).get('estimated_range_high', 0),
                damage_caps_applicable=result.get('damage_caps', {}).get('applicable', False),
                applicable_caps=result.get('damage_caps', {}).get('caps', []),
                total_exposure_low=result.get('verdict_prediction', {}).get('low_estimate', 0),
                total_exposure_high=result.get('verdict_prediction', {}).get('high_estimate', 0),
                most_likely_verdict=result.get('verdict_prediction', {}).get('most_likely', 0),
                mitigation_issues=result.get('mitigation_issues', []),
                created_at=datetime.utcnow().isoformat()
            )
            
            self.damages_analyses[analysis_id] = analysis
            return analysis
            
        except Exception as e:
            return DamagesAnalysis(
                analysis_id=analysis_id,
                case_id=case_id,
                economic_total=economic_total,
                created_at=datetime.utcnow().isoformat()
            )
    
    # ============================================================
    # SETTLEMENT ANALYSIS
    # ============================================================
    
    async def analyze_settlement(
        self,
        case_id: str,
        liability_assessment: LiabilityAssessment,
        damages_analysis: DamagesAnalysis,
        litigation_costs: float,
        policy_limits: float,
        time_to_trial_months: int
    ) -> SettlementAnalysis:
        """Calculate settlement value and negotiation strategy"""
        analysis_id = f"sett_{uuid.uuid4().hex[:8]}"
        
        # Risk-adjusted value
        liability_factor = liability_assessment.liability_probability / 100
        expected_verdict = damages_analysis.most_likely_verdict * liability_factor
        
        # Time value discount (3% annual)
        time_discount = (1 - 0.03) ** (time_to_trial_months / 12)
        
        # Present value
        present_value = expected_verdict * time_discount
        
        # Settlement range
        settlement_low = present_value * 0.7
        settlement_high = min(present_value * 1.2, policy_limits)
        
        analysis = SettlementAnalysis(
            analysis_id=analysis_id,
            case_id=case_id,
            settlement_value_low=settlement_low,
            settlement_value_high=settlement_high,
            recommended_settlement_range=(settlement_low, settlement_high),
            liability_discount=1 - liability_factor,
            litigation_costs_to_trial=litigation_costs,
            time_value_discount=1 - time_discount,
            opening_demand=damages_analysis.total_exposure_high,
            target_settlement=(settlement_low + settlement_high) / 2,
            walk_away_point=settlement_low * 0.7,
            policy_limits=policy_limits,
            created_at=datetime.utcnow().isoformat()
        )
        
        self.settlement_analyses[analysis_id] = analysis
        return analysis
    
    # ============================================================
    # CASE THEORY DEVELOPMENT
    # ============================================================
    
    async def develop_case_theory(
        self,
        case_id: str,
        party: str,
        case_facts: str,
        claims: List[str],
        key_evidence: List[str]
    ) -> CaseTheory:
        """Develop comprehensive case theory"""
        theory_id = f"theory_{uuid.uuid4().hex[:8]}"
        
        if not self.client:
            return CaseTheory(
                theory_id=theory_id,
                case_id=case_id,
                party=party,
                theme="Case theory development requires AI",
                created_at=datetime.utcnow().isoformat()
            )
        
        claims_text = "\n".join(f"- {c}" for c in claims)
        evidence_text = "\n".join(f"- {e}" for e in key_evidence)
        
        prompt = f"""Develop a comprehensive case theory for the {party}.

CASE FACTS:
{case_facts}

CLAIMS/DEFENSES:
{claims_text}

KEY EVIDENCE:
{evidence_text}

Create a persuasive case theory in JSON:
{{
    "theme": "One powerful sentence that captures the case",
    "narrative": "The compelling story of what happened (3-4 paragraphs)",
    "legal_framework": [
        {{
            "claim": "claim name",
            "elements": ["elements to prove"],
            "evidence_map": {{"element": ["supporting evidence"]}}
        }}
    ],
    "undisputed_facts": ["facts both sides agree on"],
    "disputed_facts": ["contested facts"],
    "evidence_strengths": ["strongest evidence"],
    "evidence_gaps": ["evidence we need but don't have"],
    "witness_strategy": {{
        "witness_order": ["order to call witnesses"],
        "purpose": {{"witness_name": "why calling this witness"}}
    }},
    "vulnerabilities": ["weaknesses in our case"],
    "vulnerability_responses": {{"weakness": "how to address"}},
    "opening_statement_themes": ["themes for opening"],
    "closing_argument_themes": ["themes for closing"],
    "jury_instructions_focus": ["key jury instructions to emphasize"]
}}"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are Gerry Spence developing a compelling case theory for trial."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            theory = CaseTheory(
                theory_id=theory_id,
                case_id=case_id,
                party=party,
                theme=result.get('theme', ''),
                narrative=result.get('narrative', ''),
                causes_of_action=result.get('legal_framework', []),
                undisputed_facts=result.get('undisputed_facts', []),
                disputed_facts=result.get('disputed_facts', []),
                evidence_gaps=result.get('evidence_gaps', []),
                witness_order=result.get('witness_strategy', {}).get('witness_order', []),
                vulnerabilities=result.get('vulnerabilities', []),
                responses_to_weaknesses=result.get('vulnerability_responses', {}),
                created_at=datetime.utcnow().isoformat()
            )
            
            self.case_theories[theory_id] = theory
            return theory
            
        except Exception as e:
            return CaseTheory(
                theory_id=theory_id,
                case_id=case_id,
                party=party,
                theme=f"Error: {str(e)}",
                created_at=datetime.utcnow().isoformat()
            )
    
    # ============================================================
    # JURY SELECTION
    # ============================================================
    
    async def develop_jury_profile(
        self,
        case_id: str,
        case_type: str,
        case_theory: CaseTheory,
        venue: str
    ) -> JuryProfile:
        """Develop ideal juror profile for voir dire"""
        profile_id = f"jury_{uuid.uuid4().hex[:8]}"
        
        if not self.client:
            return JuryProfile(
                profile_id=profile_id,
                case_id=case_id,
                created_at=datetime.utcnow().isoformat()
            )
        
        prompt = f"""Develop a jury selection profile for this civil rights case.

CASE TYPE: {case_type}
VENUE: {venue}
CASE THEME: {case_theory.theme}

CASE NARRATIVE:
{case_theory.narrative}

VULNERABILITIES:
{', '.join(case_theory.vulnerabilities)}

Create a comprehensive jury selection guide in JSON:
{{
    "ideal_juror_profile": {{
        "demographics": {{
            "favorable": {{"category": ["favorable traits"]}},
            "unfavorable": {{"category": ["unfavorable traits"]}}
        }},
        "occupations": {{
            "favorable": ["occupations"],
            "unfavorable": ["occupations"]
        }},
        "attitudes": {{
            "favorable": ["attitudes/beliefs"],
            "unfavorable": ["attitudes/beliefs"]
        }},
        "experiences": {{
            "favorable": ["life experiences"],
            "unfavorable": ["life experiences"]
        }}
    }},
    "voir_dire_questions": {{
        "must_ask": ["essential questions"],
        "attitudes_toward_police": ["questions about police"],
        "attitudes_toward_lawsuits": ["questions about civil litigation"],
        "relevant_experiences": ["questions about experiences"],
        "follow_ups": {{"if_answer": ["follow-up questions"]}}
    }},
    "red_flags": ["warning signs to watch for"],
    "green_flags": ["positive signs"],
    "for_cause_challenges": ["grounds for cause challenges"],
    "peremptory_priorities": ["who to strike first with peremptories"],
    "stealth_juror_detection": ["questions to uncover hidden bias"],
    "body_language_cues": ["non-verbal signals to watch"]
}}"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a jury consultant specializing in civil rights cases."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            profile = JuryProfile(
                profile_id=profile_id,
                case_id=case_id,
                favorable_demographics=result.get('ideal_juror_profile', {}).get('demographics', {}).get('favorable', {}),
                unfavorable_demographics=result.get('ideal_juror_profile', {}).get('demographics', {}).get('unfavorable', {}),
                favorable_attitudes=result.get('ideal_juror_profile', {}).get('attitudes', {}).get('favorable', []),
                unfavorable_attitudes=result.get('ideal_juror_profile', {}).get('attitudes', {}).get('unfavorable', []),
                favorable_experiences=result.get('ideal_juror_profile', {}).get('experiences', {}).get('favorable', []),
                unfavorable_experiences=result.get('ideal_juror_profile', {}).get('experiences', {}).get('unfavorable', []),
                must_ask_questions=result.get('voir_dire_questions', {}).get('must_ask', []),
                follow_up_questions=result.get('voir_dire_questions', {}).get('follow_ups', {}),
                for_cause_triggers=result.get('for_cause_challenges', []),
                peremptory_priorities=result.get('peremptory_priorities', []),
                created_at=datetime.utcnow().isoformat()
            )
            
            self.jury_profiles[profile_id] = profile
            return profile
            
        except Exception as e:
            return JuryProfile(
                profile_id=profile_id,
                case_id=case_id,
                created_at=datetime.utcnow().isoformat()
            )


# Service instance
case_strategy_service = CaseStrategyService()
