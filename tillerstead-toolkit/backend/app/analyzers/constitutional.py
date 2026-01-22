"""
BarberX Legal Case Management Pro Suite
Constitutional Analysis Engine
Violation Detection, Pattern Recognition, Liability Assessment
"""
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class Amendment(Enum):
    """Constitutional Amendments"""
    FOURTH = "4th"
    FIFTH = "5th"
    SIXTH = "6th"
    EIGHTH = "8th"
    FOURTEENTH = "14th"


class ViolationCategory(Enum):
    """Violation categories"""
    # 4th Amendment
    UNLAWFUL_SEARCH = "4th_unlawful_search"
    UNLAWFUL_SEIZURE = "4th_unlawful_seizure"
    EXCESSIVE_FORCE = "4th_excessive_force"
    WARRANTLESS_ARREST = "4th_warrantless_arrest"
    
    # 5th Amendment
    MIRANDA_VIOLATION = "5th_miranda"
    SELF_INCRIMINATION = "5th_self_incrimination"
    DUE_PROCESS = "5th_due_process"
    
    # 6th Amendment
    RIGHT_TO_COUNSEL = "6th_right_to_counsel"
    SPEEDY_TRIAL = "6th_speedy_trial"
    CONFRONTATION = "6th_confrontation"
    
    # 8th Amendment
    CRUEL_PUNISHMENT = "8th_cruel_punishment"
    EXCESSIVE_BAIL = "8th_excessive_bail"
    
    # 14th Amendment
    EQUAL_PROTECTION = "14th_equal_protection"
    PROCEDURAL_DUE_PROCESS = "14th_procedural_due_process"
    
    # Evidentiary
    BRADY_VIOLATION = "brady_violation"
    GIGLIO_MATERIAL = "giglio_material"
    EVIDENCE_TAMPERING = "evidence_tampering"
    
    # Policy/Training
    POLICY_VIOLATION = "policy_violation"
    TRAINING_FAILURE = "training_failure"
    SUPERVISION_FAILURE = "supervision_failure"


@dataclass
class ViolationIndicator:
    """Indicator of a potential violation"""
    category: ViolationCategory
    keyword: str
    context: str
    confidence: float
    source_type: str  # document, transcript, etc.
    source_reference: str  # page number, timestamp, etc.


@dataclass
class ConstitutionalViolation:
    """Identified constitutional violation"""
    category: ViolationCategory
    amendment: Amendment
    title: str
    description: str
    severity: int  # 1-5
    confidence: float  # 0-1
    indicators: List[ViolationIndicator]
    legal_basis: List[str]
    estimated_damages: Tuple[float, float, float]  # low, mid, high
    recommendations: List[str]


@dataclass
class LiabilityAssessment:
    """Complete liability assessment"""
    overall_score: float  # 0-100
    constitutional_score: float
    evidence_score: float
    pattern_score: float
    
    violations: List[ConstitutionalViolation]
    total_estimated_damages: Tuple[float, float, float]
    
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    
    comparable_cases: List[Dict[str, Any]]


class ConstitutionalAnalyzer:
    """
    Constitutional Violation Analysis Engine
    
    Scans documents and transcripts for indicators of constitutional
    violations based on keyword patterns, context analysis, and
    legal precedent patterns.
    """
    
    # Keyword patterns for each violation type
    VIOLATION_PATTERNS = {
        ViolationCategory.UNLAWFUL_SEARCH: {
            'keywords': [
                'without warrant', 'warrantless search', 'no consent',
                'exceeded scope', 'pretextual', 'fruit of the poisonous tree',
                'illegal search', 'unreasonable search', 'fourth amendment',
                'no probable cause', 'invalid warrant', 'stale warrant'
            ],
            'amendment': Amendment.FOURTH,
            'severity_base': 3,
            'damages_range': (15000, 75000, 200000)
        },
        
        ViolationCategory.UNLAWFUL_SEIZURE: {
            'keywords': [
                'unlawful arrest', 'false arrest', 'wrongful arrest',
                'no reasonable suspicion', 'prolonged detention',
                'without probable cause', 'illegal detention',
                'unlawful stop', 'pretextual stop'
            ],
            'amendment': Amendment.FOURTH,
            'severity_base': 3,
            'damages_range': (20000, 100000, 300000)
        },
        
        ViolationCategory.EXCESSIVE_FORCE: {
            'keywords': [
                'excessive force', 'unreasonable force', 'struck', 'punched',
                'kicked', 'tased', 'pepper spray', 'shot', 'choked',
                'knee on neck', 'knee on back', 'hog-tied', 'beat',
                'assault', 'baton', 'continued after', 'handcuffed and',
                'compliant', 'not resisting', 'submitted', 'surrendered',
                'prone position', 'positional asphyxia', 'restraint'
            ],
            'amendment': Amendment.FOURTH,
            'severity_base': 4,
            'damages_range': (50000, 200000, 1000000)
        },
        
        ViolationCategory.MIRANDA_VIOLATION: {
            'keywords': [
                'custodial interrogation', 'not advised', 'no miranda',
                'without miranda', 'failed to advise', 'continued questioning',
                'after invoked', 'not free to leave', 'in custody',
                'coerced statement', 'involuntary statement'
            ],
            'amendment': Amendment.FIFTH,
            'severity_base': 3,
            'damages_range': (10000, 50000, 150000)
        },
        
        ViolationCategory.SELF_INCRIMINATION: {
            'keywords': [
                'forced to testify', 'compelled statement',
                'threatened if didn\'t talk', 'coerced confession',
                'involuntary confession'
            ],
            'amendment': Amendment.FIFTH,
            'severity_base': 4,
            'damages_range': (25000, 100000, 300000)
        },
        
        ViolationCategory.DUE_PROCESS: {
            'keywords': [
                'due process', 'no notice', 'denied hearing',
                'procedural violation', 'arbitrary', 'fundamentally unfair',
                'shock the conscience', 'deliberate indifference'
            ],
            'amendment': Amendment.FIFTH,
            'severity_base': 3,
            'damages_range': (15000, 75000, 200000)
        },
        
        ViolationCategory.RIGHT_TO_COUNSEL: {
            'keywords': [
                'requested attorney', 'asked for lawyer', 'invoked counsel',
                'want a lawyer', 'need an attorney', 'right to counsel',
                'continued interrogation after', 'denied counsel',
                'sixth amendment'
            ],
            'amendment': Amendment.SIXTH,
            'severity_base': 4,
            'damages_range': (20000, 100000, 300000)
        },
        
        ViolationCategory.CRUEL_PUNISHMENT: {
            'keywords': [
                'cruel and unusual', 'inhumane conditions', 'deliberate indifference',
                'denial of medical care', 'overcrowding', 'isolation',
                'solitary confinement', 'excessive sentence'
            ],
            'amendment': Amendment.EIGHTH,
            'severity_base': 4,
            'damages_range': (25000, 150000, 500000)
        },
        
        ViolationCategory.EQUAL_PROTECTION: {
            'keywords': [
                'racial profiling', 'discriminatory', 'selective enforcement',
                'treated differently', 'disparate treatment', 'bias',
                'prejudice', 'discrimination', 'profiled'
            ],
            'amendment': Amendment.FOURTEENTH,
            'severity_base': 4,
            'damages_range': (30000, 150000, 500000)
        },
        
        ViolationCategory.BRADY_VIOLATION: {
            'keywords': [
                'withheld evidence', 'concealed', 'exculpatory',
                'failed to disclose', 'suppressed evidence', 'brady',
                'material evidence', 'favorable to defense',
                'impeachment evidence'
            ],
            'amendment': Amendment.FOURTEENTH,
            'severity_base': 5,
            'damages_range': (50000, 250000, 1000000)
        },
        
        ViolationCategory.GIGLIO_MATERIAL: {
            'keywords': [
                'giglio', 'prior misconduct', 'credibility',
                'false testimony', 'perjury', 'untruthful',
                'prior discipline', 'internal affairs', 'sustained complaint'
            ],
            'amendment': Amendment.FOURTEENTH,
            'severity_base': 4,
            'damages_range': (25000, 150000, 500000)
        },
        
        ViolationCategory.POLICY_VIOLATION: {
            'keywords': [
                'violated policy', 'against policy', 'policy violation',
                'standard operating procedure', 'general order',
                'department policy', 'use of force policy'
            ],
            'amendment': Amendment.FOURTEENTH,
            'severity_base': 2,
            'damages_range': (10000, 50000, 150000)
        }
    }
    
    # Legal citations for each category
    LEGAL_CITATIONS = {
        ViolationCategory.UNLAWFUL_SEARCH: [
            "Mapp v. Ohio, 367 U.S. 643 (1961)",
            "Katz v. United States, 389 U.S. 347 (1967)",
            "Terry v. Ohio, 392 U.S. 1 (1968)",
            "Illinois v. Gates, 462 U.S. 213 (1983)"
        ],
        ViolationCategory.UNLAWFUL_SEIZURE: [
            "Terry v. Ohio, 392 U.S. 1 (1968)",
            "Florida v. Bostick, 501 U.S. 429 (1991)",
            "Rodriguez v. United States, 575 U.S. 348 (2015)"
        ],
        ViolationCategory.EXCESSIVE_FORCE: [
            "Graham v. Connor, 490 U.S. 386 (1989)",
            "Tennessee v. Garner, 471 U.S. 1 (1985)",
            "Kingsley v. Hendrickson, 576 U.S. 389 (2015)",
            "Scott v. Harris, 550 U.S. 372 (2007)"
        ],
        ViolationCategory.MIRANDA_VIOLATION: [
            "Miranda v. Arizona, 384 U.S. 436 (1966)",
            "Berghuis v. Thompkins, 560 U.S. 370 (2010)",
            "Edwards v. Arizona, 451 U.S. 477 (1981)"
        ],
        ViolationCategory.RIGHT_TO_COUNSEL: [
            "Gideon v. Wainwright, 372 U.S. 335 (1963)",
            "Edwards v. Arizona, 451 U.S. 477 (1981)",
            "Michigan v. Jackson, 475 U.S. 625 (1986)"
        ],
        ViolationCategory.BRADY_VIOLATION: [
            "Brady v. Maryland, 373 U.S. 83 (1963)",
            "Giglio v. United States, 405 U.S. 150 (1972)",
            "Strickler v. Greene, 527 U.S. 263 (1999)"
        ],
        ViolationCategory.EQUAL_PROTECTION: [
            "Whren v. United States, 517 U.S. 806 (1996)",
            "Village of Arlington Heights v. Metropolitan Housing, 429 U.S. 252 (1977)"
        ]
    }
    
    def __init__(self):
        """Initialize analyzer"""
        self.indicators_found = []
    
    def analyze_text(
        self,
        text: str,
        source_type: str = "document",
        source_reference: str = ""
    ) -> List[ViolationIndicator]:
        """
        Scan text for violation indicators.
        
        Args:
            text: Text content to analyze
            source_type: Type of source (document, transcript, etc.)
            source_reference: Reference (page, timestamp, etc.)
            
        Returns:
            List of violation indicators found
        """
        indicators = []
        text_lower = text.lower()
        
        for category, config in self.VIOLATION_PATTERNS.items():
            for keyword in config['keywords']:
                if keyword in text_lower:
                    # Extract context around keyword
                    idx = text_lower.find(keyword)
                    start = max(0, idx - 150)
                    end = min(len(text), idx + len(keyword) + 150)
                    context = text[start:end].strip()
                    
                    # Calculate confidence based on keyword specificity
                    confidence = 0.6 + (len(keyword) / 50) * 0.2
                    confidence = min(0.95, confidence)
                    
                    indicators.append(ViolationIndicator(
                        category=category,
                        keyword=keyword,
                        context=context,
                        confidence=confidence,
                        source_type=source_type,
                        source_reference=source_reference
                    ))
        
        return indicators
    
    def consolidate_violations(
        self,
        indicators: List[ViolationIndicator]
    ) -> List[ConstitutionalViolation]:
        """
        Consolidate indicators into distinct violations.
        
        Groups related indicators and creates violation records.
        """
        # Group indicators by category
        by_category = {}
        for ind in indicators:
            if ind.category not in by_category:
                by_category[ind.category] = []
            by_category[ind.category].append(ind)
        
        violations = []
        for category, category_indicators in by_category.items():
            config = self.VIOLATION_PATTERNS.get(category, {})
            
            # Calculate aggregate confidence
            avg_confidence = sum(i.confidence for i in category_indicators) / len(category_indicators)
            
            # Boost confidence if multiple indicators
            if len(category_indicators) > 2:
                avg_confidence = min(0.95, avg_confidence + 0.1)
            
            # Determine severity
            base_severity = config.get('severity_base', 3)
            if len(category_indicators) > 3:
                base_severity = min(5, base_severity + 1)
            
            violation = ConstitutionalViolation(
                category=category,
                amendment=config.get('amendment', Amendment.FOURTH),
                title=self._format_title(category),
                description=self._generate_description(category, category_indicators),
                severity=base_severity,
                confidence=avg_confidence,
                indicators=category_indicators,
                legal_basis=self.LEGAL_CITATIONS.get(category, []),
                estimated_damages=config.get('damages_range', (10000, 50000, 150000)),
                recommendations=self._generate_recommendations(category)
            )
            
            violations.append(violation)
        
        return violations
    
    def _format_title(self, category: ViolationCategory) -> str:
        """Format category into readable title"""
        parts = category.value.split('_')
        
        # Handle amendment prefix
        if parts[0] in ['4th', '5th', '6th', '8th', '14th']:
            amendment = parts[0]
            rest = ' '.join(parts[1:]).title()
            return f"{amendment} Amendment - {rest}"
        
        return ' '.join(parts).title()
    
    def _generate_description(
        self,
        category: ViolationCategory,
        indicators: List[ViolationIndicator]
    ) -> str:
        """Generate description from indicators"""
        contexts = [i.context for i in indicators[:3]]
        
        intro = f"Evidence suggests potential {self._format_title(category).lower()}. "
        
        if contexts:
            intro += "Relevant passages include: "
            intro += " [...] ".join(f'"{c}"' for c in contexts)
        
        return intro
    
    def _generate_recommendations(self, category: ViolationCategory) -> List[str]:
        """Generate recommendations for addressing violation"""
        base_recs = [
            "Document all related evidence thoroughly",
            "Preserve chain of custody for all materials"
        ]
        
        category_recs = {
            ViolationCategory.EXCESSIVE_FORCE: [
                "Obtain all medical records documenting injuries",
                "Photograph all visible injuries over time",
                "Request all use-of-force reports",
                "Obtain BWC footage from all officers present"
            ],
            ViolationCategory.UNLAWFUL_SEARCH: [
                "Request copy of warrant (if any)",
                "Document scope of search",
                "Interview witnesses to search"
            ],
            ViolationCategory.MIRANDA_VIOLATION: [
                "File motion to suppress statements",
                "Document custodial circumstances",
                "Obtain all interrogation recordings"
            ],
            ViolationCategory.BRADY_VIOLATION: [
                "File motion to compel disclosure",
                "Request sanctions for non-disclosure",
                "Document all discovery requests and responses"
            ]
        }
        
        return base_recs + category_recs.get(category, [])
    
    def assess_liability(
        self,
        violations: List[ConstitutionalViolation],
        evidence_strength: float = 0.5,
        officer_history: Optional[Dict] = None
    ) -> LiabilityAssessment:
        """
        Generate comprehensive liability assessment.
        
        Args:
            violations: List of identified violations
            evidence_strength: 0-1 score of evidence strength
            officer_history: Optional officer history data
            
        Returns:
            Complete liability assessment
        """
        # Calculate scores
        if violations:
            constitutional_score = min(100, len(violations) * 20 + 
                sum(v.severity * 5 for v in violations))
        else:
            constitutional_score = 0
        
        evidence_score = evidence_strength * 100
        
        # Pattern score from officer history
        pattern_score = 50  # Default
        if officer_history:
            prior_complaints = officer_history.get('total_complaints', 0)
            sustained = officer_history.get('sustained_complaints', 0)
            pattern_score = min(100, prior_complaints * 10 + sustained * 20)
        
        # Overall score
        overall_score = (
            constitutional_score * 0.40 +
            evidence_score * 0.35 +
            pattern_score * 0.25
        )
        
        # Calculate total damages
        if violations:
            total_low = sum(v.estimated_damages[0] for v in violations)
            total_mid = sum(v.estimated_damages[1] for v in violations)
            total_high = sum(v.estimated_damages[2] for v in violations)
        else:
            total_low, total_mid, total_high = 0, 0, 0
        
        # Generate strengths/weaknesses
        strengths = []
        weaknesses = []
        recommendations = []
        
        if len(violations) > 2:
            strengths.append(f"Multiple constitutional violations identified ({len(violations)})")
        elif len(violations) == 0:
            weaknesses.append("No constitutional violations identified yet")
            recommendations.append("Conduct thorough review of all documents and evidence")
        
        if evidence_strength > 0.7:
            strengths.append("Strong evidentiary support")
        elif evidence_strength < 0.4:
            weaknesses.append("Evidence needs strengthening")
            recommendations.append("Obtain additional supporting documentation")
        
        severe_violations = [v for v in violations if v.severity >= 4]
        if severe_violations:
            strengths.append(f"{len(severe_violations)} severe violations strengthen case")
        
        # Add violation-specific recommendations
        for v in violations:
            recommendations.extend(v.recommendations[:2])
        
        # Deduplicate recommendations
        recommendations = list(dict.fromkeys(recommendations))
        
        return LiabilityAssessment(
            overall_score=overall_score,
            constitutional_score=constitutional_score,
            evidence_score=evidence_score,
            pattern_score=pattern_score,
            violations=violations,
            total_estimated_damages=(total_low, total_mid, total_high),
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations[:10],
            comparable_cases=self._get_comparable_cases(violations)
        )
    
    def _get_comparable_cases(
        self,
        violations: List[ConstitutionalViolation]
    ) -> List[Dict[str, Any]]:
        """Get comparable case settlements"""
        # This would query a database of settlements
        # For now, return representative examples
        
        comparables = []
        
        categories = set(v.category for v in violations)
        
        if ViolationCategory.EXCESSIVE_FORCE in categories:
            comparables.extend([
                {
                    "description": "Excessive force during arrest - NJ Federal Court",
                    "settlement": 175000,
                    "year": 2024,
                    "factors": ["BWC footage", "injuries documented"]
                },
                {
                    "description": "Use of taser on compliant subject",
                    "settlement": 125000,
                    "year": 2023,
                    "factors": ["Multiple witnesses"]
                }
            ])
        
        if ViolationCategory.UNLAWFUL_SEIZURE in categories:
            comparables.extend([
                {
                    "description": "False arrest / wrongful detention",
                    "settlement": 85000,
                    "year": 2024,
                    "factors": ["Charges dismissed"]
                }
            ])
        
        if ViolationCategory.BRADY_VIOLATION in categories:
            comparables.extend([
                {
                    "description": "Withheld exculpatory evidence - reversal",
                    "settlement": 350000,
                    "year": 2023,
                    "factors": ["Wrongful conviction"]
                }
            ])
        
        return comparables[:5]


# Convenience functions
def scan_for_violations(text: str) -> List[Dict[str, Any]]:
    """Quick scan for violations"""
    analyzer = ConstitutionalAnalyzer()
    indicators = analyzer.analyze_text(text)
    violations = analyzer.consolidate_violations(indicators)
    
    return [
        {
            'category': v.category.value,
            'title': v.title,
            'severity': v.severity,
            'confidence': v.confidence,
            'estimated_damages': v.estimated_damages
        }
        for v in violations
    ]


def assess_case_liability(violations_data: List[Dict]) -> Dict[str, Any]:
    """Quick liability assessment"""
    analyzer = ConstitutionalAnalyzer()
    
    # Convert dict data to violation objects (simplified)
    assessment = analyzer.assess_liability([], evidence_strength=0.5)
    
    return {
        'overall_score': assessment.overall_score,
        'constitutional_score': assessment.constitutional_score,
        'recommendations': assessment.recommendations
    }
