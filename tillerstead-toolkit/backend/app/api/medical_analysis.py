"""
Medical & Expert Analysis API
Medical NER, injury assessment, ICD-10 coding, expert report analysis
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import spacy
from datetime import datetime

router = APIRouter(prefix="/api/v1/medical", tags=["Medical Analysis"])

# Global models
medical_models = {}


class MedicalDocumentRequest(BaseModel):
    text: str = Field(..., description="Medical record text")
    extract_diagnoses: bool = True
    extract_medications: bool = True
    extract_procedures: bool = True
    extract_injuries: bool = True
    extract_timeline: bool = True


class MedicalEntity(BaseModel):
    text: str
    label: str  # DIAGNOSIS, MEDICATION, PROCEDURE, ANATOMY, etc.
    start: int
    end: int
    confidence: float
    icd_10_code: Optional[str] = None


class MedicalAnalysisResult(BaseModel):
    diagnoses: List[MedicalEntity]
    medications: List[MedicalEntity]
    procedures: List[MedicalEntity]
    injuries: List[MedicalEntity]
    anatomical_sites: List[MedicalEntity]
    timeline: List[Dict[str, Any]]
    summary: str


class InjuryAssessmentRequest(BaseModel):
    injury_description: str
    medical_records: Optional[str] = None
    imaging_reports: Optional[str] = None


class InjuryAssessment(BaseModel):
    injury_type: str
    severity: str  # minor, moderate, severe, critical
    body_region: str
    icd_10_codes: List[str]
    treatment_required: List[str]
    recovery_timeline: str
    permanency_likelihood: str  # low, medium, high
    damages_assessment: Dict[str, Any]


class DrugInteractionRequest(BaseModel):
    medications: List[str] = Field(..., description="List of medication names")
    check_severity: bool = True


class DrugInteraction(BaseModel):
    drug1: str
    drug2: str
    interaction_type: str
    severity: str  # minor, moderate, major, contraindicated
    description: str
    recommendation: str


class MedicalBillingRequest(BaseModel):
    billing_codes: List[str]
    procedure_descriptions: List[str]
    dates_of_service: List[datetime]


class BillingAnalysis(BaseModel):
    total_amount: float
    code_validation: List[Dict[str, Any]]
    potential_errors: List[str]
    comparison_to_reasonable: str


@router.on_event("startup")
async def load_medical_models():
    """Load medical NLP models"""
    try:
        # Load medspaCy or scispaCy
        # Note: Install with: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz
        medical_models['scispacy'] = spacy.load("en_core_sci_sm")
        print("✅ Loaded medical NLP model (scispaCy)")
        
    except Exception as e:
        print(f"⚠️ Failed to load medical models: {e}")


@router.post("/analyze-medical-record", response_model=MedicalAnalysisResult)
async def analyze_medical_record(request: MedicalDocumentRequest):
    """
    Comprehensive medical record analysis
    
    Extracts:
    - Diagnoses with ICD-10 codes
    - Medications and dosages
    - Procedures performed
    - Injuries and anatomical sites
    - Medical timeline
    """
    try:
        nlp = medical_models.get('scispacy') or spacy.load("en_core_sci_sm")
        doc = nlp(request.text)
        
        # Extract entities
        diagnoses = []
        medications = []
        procedures = []
        injuries = []
        anatomical_sites = []
        
        for ent in doc.ents:
            entity = MedicalEntity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                confidence=0.85
            )
            
            if ent.label_ in ['DISEASE', 'DISORDER', 'DIAGNOSIS']:
                # Try to map to ICD-10
                entity.icd_10_code = map_to_icd10(ent.text)
                diagnoses.append(entity)
                
            elif ent.label_ in ['DRUG', 'MEDICATION']:
                medications.append(entity)
                
            elif ent.label_ == 'PROCEDURE':
                procedures.append(entity)
                
            elif ent.label_ in ['INJURY', 'TRAUMA']:
                injuries.append(entity)
                
            elif ent.label_ in ['ANATOMY', 'BODY_PART']:
                anatomical_sites.append(entity)
        
        # Extract timeline (dates and events)
        timeline = []
        for sent in doc.sents:
            # Look for date entities
            dates = [ent for ent in sent.ents if ent.label_ == 'DATE']
            if dates:
                timeline.append({
                    'date': dates[0].text,
                    'event': sent.text,
                    'type': 'medical_event'
                })
        
        # Generate summary
        summary = generate_medical_summary(diagnoses, medications, procedures, injuries)
        
        return MedicalAnalysisResult(
            diagnoses=diagnoses,
            medications=medications,
            procedures=procedures,
            injuries=injuries,
            anatomical_sites=anatomical_sites,
            timeline=timeline,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Medical record analysis failed: {str(e)}")


@router.post("/assess-injury", response_model=InjuryAssessment)
async def assess_injury(request: InjuryAssessmentRequest):
    """
    Comprehensive injury assessment
    
    Analyzes:
    - Type and severity of injury
    - Body region affected
    - Treatment requirements
    - Recovery timeline
    - Permanency potential
    - Damages estimate
    """
    try:
        nlp = medical_models.get('scispacy') or spacy.load("en_core_sci_sm")
        doc = nlp(request.injury_description)
        
        # Extract injury details
        injury_type = "Unknown"
        body_region = "Unknown"
        anatomical_entities = []
        
        for ent in doc.ents:
            if ent.label_ in ['INJURY', 'DISEASE', 'DISORDER']:
                injury_type = ent.text
            elif ent.label_ in ['ANATOMY', 'BODY_PART']:
                anatomical_entities.append(ent.text)
                if not body_region or body_region == "Unknown":
                    body_region = ent.text
        
        # Assess severity based on keywords
        severity_keywords = {
            'critical': ['fracture', 'hemorrhage', 'traumatic brain', 'spinal cord', 'amputation'],
            'severe': ['laceration', 'dislocation', 'torn', 'ruptured', 'compound'],
            'moderate': ['sprain', 'strain', 'contusion', 'bruise'],
            'minor': ['abrasion', 'minor cut', 'superficial']
        }
        
        severity = "moderate"  # default
        text_lower = request.injury_description.lower()
        
        for sev_level, keywords in severity_keywords.items():
            if any(kw in text_lower for kw in keywords):
                severity = sev_level
                break
        
        # Map to ICD-10 codes
        icd_10_codes = [map_to_icd10(injury_type)] if map_to_icd10(injury_type) else []
        
        # Determine treatment
        treatment_map = {
            'critical': ['Emergency surgery', 'ICU monitoring', 'Extended hospitalization'],
            'severe': ['Surgery', 'Hospitalization', 'Physical therapy', 'Pain management'],
            'moderate': ['Outpatient care', 'Physical therapy', 'Medication', 'Follow-up visits'],
            'minor': ['First aid', 'Over-the-counter medication', 'Rest']
        }
        treatment_required = treatment_map.get(severity, ['Medical evaluation required'])
        
        # Recovery timeline
        recovery_map = {
            'critical': '6-24 months with potential permanent disability',
            'severe': '3-12 months with possible permanent effects',
            'moderate': '4-12 weeks with full recovery expected',
            'minor': '1-4 weeks with full recovery expected'
        }
        recovery_timeline = recovery_map.get(severity, 'Variable, requires medical evaluation')
        
        # Permanency likelihood
        permanency_map = {
            'critical': 'high',
            'severe': 'medium',
            'moderate': 'low',
            'minor': 'low'
        }
        permanency_likelihood = permanency_map.get(severity, 'medium')
        
        # Damages assessment (rough estimates)
        damages_map = {
            'critical': {'medical': (100000, 500000), 'pain_suffering': (200000, 1000000)},
            'severe': {'medical': (25000, 100000), 'pain_suffering': (50000, 200000)},
            'moderate': {'medical': (5000, 25000), 'pain_suffering': (10000, 50000)},
            'minor': {'medical': (500, 5000), 'pain_suffering': (1000, 10000)}
        }
        
        damage_estimates = damages_map.get(severity, {'medical': (0, 0), 'pain_suffering': (0, 0)})
        
        damages_assessment = {
            'medical_costs_range': damage_estimates['medical'],
            'pain_suffering_range': damage_estimates['pain_suffering'],
            'lost_wages': 'Requires employment data',
            'future_care': 'Requires life care plan' if severity in ['critical', 'severe'] else 'Not typically required'
        }
        
        return InjuryAssessment(
            injury_type=injury_type,
            severity=severity,
            body_region=body_region,
            icd_10_codes=icd_10_codes,
            treatment_required=treatment_required,
            recovery_timeline=recovery_timeline,
            permanency_likelihood=permanency_likelihood,
            damages_assessment=damages_assessment
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Injury assessment failed: {str(e)}")


@router.post("/check-drug-interactions", response_model=List[DrugInteraction])
async def check_drug_interactions(request: DrugInteractionRequest):
    """
    Check for drug-drug interactions
    
    Uses medical knowledge base to identify potential conflicts
    """
    try:
        interactions = []
        
        # Known interaction database (simplified - would use DrugBank API in production)
        known_interactions = {
            ('warfarin', 'aspirin'): {
                'type': 'Bleeding Risk',
                'severity': 'major',
                'description': 'Both drugs increase bleeding risk when combined',
                'recommendation': 'Use with extreme caution; monitor INR closely'
            },
            ('lisinopril', 'ibuprofen'): {
                'type': 'Reduced Effectiveness',
                'severity': 'moderate',
                'description': 'NSAIDs may reduce the effectiveness of ACE inhibitors',
                'recommendation': 'Monitor blood pressure; consider alternative pain reliever'
            }
        }
        
        # Check all combinations
        for i, drug1 in enumerate(request.medications):
            for drug2 in request.medications[i+1:]:
                # Normalize names
                d1 = drug1.lower().strip()
                d2 = drug2.lower().strip()
                
                # Check both orderings
                interaction_data = known_interactions.get((d1, d2)) or known_interactions.get((d2, d1))
                
                if interaction_data:
                    interactions.append(DrugInteraction(
                        drug1=drug1,
                        drug2=drug2,
                        interaction_type=interaction_data['type'],
                        severity=interaction_data['severity'],
                        description=interaction_data['description'],
                        recommendation=interaction_data['recommendation']
                    ))
        
        return interactions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drug interaction check failed: {str(e)}")


@router.post("/analyze-medical-billing", response_model=BillingAnalysis)
async def analyze_medical_billing(request: MedicalBillingRequest):
    """
    Analyze medical billing for errors and reasonableness
    
    Features:
    - CPT code validation
    - Billing error detection
    - Comparison to Medicare rates
    - Duplicate charge detection
    """
    try:
        code_validation = []
        potential_errors = []
        total = 0.0
        
        # Validate each billing code
        for code, description, date in zip(
            request.billing_codes,
            request.procedure_descriptions,
            request.dates_of_service
        ):
            # Check code format (CPT codes are 5 digits)
            if not code.isdigit() or len(code) != 5:
                potential_errors.append(f"Invalid CPT code format: {code}")
            
            # Placeholder cost (would look up actual Medicare rates)
            estimated_cost = 500.0
            total += estimated_cost
            
            code_validation.append({
                'code': code,
                'description': description,
                'date': date,
                'estimated_cost': estimated_cost,
                'valid': code.isdigit() and len(code) == 5
            })
        
        # Check for duplicates
        code_counts = {}
        for code in request.billing_codes:
            code_counts[code] = code_counts.get(code, 0) + 1
        
        for code, count in code_counts.items():
            if count > 1:
                potential_errors.append(f"Duplicate billing code detected: {code} (billed {count} times)")
        
        # Assess reasonableness
        comparison = "Requires detailed review with Medicare fee schedule"
        if total > 50000:
            comparison = "Significantly elevated - warrant detailed audit"
        elif total > 10000:
            comparison = "Above average - review recommended"
        
        return BillingAnalysis(
            total_amount=total,
            code_validation=code_validation,
            potential_errors=potential_errors,
            comparison_to_reasonable=comparison
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Billing analysis failed: {str(e)}")


# Helper functions
def map_to_icd10(diagnosis_text: str) -> Optional[str]:
    """Map diagnosis text to ICD-10 code"""
    # Simplified mapping (would use full ICD-10 database in production)
    icd10_mapping = {
        'fracture': 'S02-S92',
        'laceration': 'S01',
        'concussion': 'S06.0',
        'contusion': 'S00',
        'sprain': 'S03-S93',
        'strain': 'S03-S93'
    }
    
    text_lower = diagnosis_text.lower()
    for term, code in icd10_mapping.items():
        if term in text_lower:
            return code
    
    return None


def generate_medical_summary(diagnoses, medications, procedures, injuries):
    """Generate natural language summary of medical findings"""
    summary_parts = []
    
    if diagnoses:
        diag_list = ', '.join([d.text for d in diagnoses[:3]])
        summary_parts.append(f"Diagnosed conditions include: {diag_list}")
    
    if injuries:
        injury_list = ', '.join([i.text for i in injuries[:3]])
        summary_parts.append(f"Injuries sustained: {injury_list}")
    
    if medications:
        med_list = ', '.join([m.text for m in medications[:3]])
        summary_parts.append(f"Current medications: {med_list}")
    
    if procedures:
        proc_list = ', '.join([p.text for p in procedures[:3]])
        summary_parts.append(f"Procedures performed: {proc_list}")
    
    return '. '.join(summary_parts) + '.' if summary_parts else "No significant medical findings extracted."
