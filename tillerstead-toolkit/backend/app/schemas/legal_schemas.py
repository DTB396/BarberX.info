"""
BarberX Legal Case Management Pro Suite
Pydantic Schemas for API Request/Response Validation
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator
from enum import Enum


# ============================================================
# ENUMS
# ============================================================

class CaseStatus(str, Enum):
    INTAKE = "intake"
    INVESTIGATION = "investigation"
    PRE_LITIGATION = "pre_litigation"
    LITIGATION = "litigation"
    DISCOVERY = "discovery"
    TRIAL = "trial"
    APPEAL = "appeal"
    SETTLED = "settled"
    CLOSED = "closed"


class CaseType(str, Enum):
    CIVIL_RIGHTS = "civil_rights"
    POLICE_MISCONDUCT = "police_misconduct"
    EXCESSIVE_FORCE = "excessive_force"
    FALSE_ARREST = "false_arrest"
    MALICIOUS_PROSECUTION = "malicious_prosecution"
    EMPLOYMENT = "employment"
    DISCRIMINATION = "discrimination"
    PCR = "post_conviction_relief"
    OPRA = "opra_request"
    ADMINISTRATIVE = "administrative"
    OTHER = "other"


class DocumentType(str, Enum):
    COMPLAINT = "complaint"
    ANSWER = "answer"
    MOTION = "motion"
    ORDER = "order"
    BRIEF = "brief"
    DISCOVERY_REQUEST = "discovery_request"
    DISCOVERY_RESPONSE = "discovery_response"
    DEPOSITION = "deposition"
    AFFIDAVIT = "affidavit"
    EXHIBIT = "exhibit"
    POLICE_REPORT = "police_report"
    INCIDENT_REPORT = "incident_report"
    MEDICAL_RECORD = "medical_record"
    CORRESPONDENCE = "correspondence"
    CONTRACT = "contract"
    INTERNAL_MEMO = "internal_memo"
    BWC_METADATA = "bwc_metadata"
    OTHER = "other"


class EvidenceType(str, Enum):
    BWC_FOOTAGE = "bwc_footage"
    DASH_CAM = "dash_cam"
    SURVEILLANCE = "surveillance"
    CELL_PHONE_VIDEO = "cell_phone_video"
    AUDIO_RECORDING = "audio_recording"
    PHOTOGRAPH = "photograph"
    PHYSICAL_EVIDENCE = "physical_evidence"
    DIGITAL_EVIDENCE = "digital_evidence"
    WITNESS_STATEMENT = "witness_statement"
    EXPERT_REPORT = "expert_report"
    OTHER = "other"


class ViolationType(str, Enum):
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
    # Other
    POLICY_VIOLATION = "policy_violation"
    TRAINING_FAILURE = "training_failure"
    SUPERVISION_FAILURE = "supervision_failure"


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_REVIEW = "requires_review"


class PartyRole(str, Enum):
    PLAINTIFF = "plaintiff"
    DEFENDANT = "defendant"
    WITNESS = "witness"
    EXPERT_WITNESS = "expert_witness"
    ATTORNEY = "attorney"
    JUDGE = "judge"
    OFFICER = "officer"
    DEPARTMENT = "department"
    MUNICIPALITY = "municipality"


# ============================================================
# CASE SCHEMAS
# ============================================================

class CaseBase(BaseModel):
    """Base schema for cases"""
    title: str = Field(..., min_length=1, max_length=500)
    short_title: Optional[str] = Field(None, max_length=100)
    case_number: Optional[str] = Field(None, max_length=100)
    docket_number: Optional[str] = Field(None, max_length=100)
    case_type: CaseType = CaseType.CIVIL_RIGHTS
    status: CaseStatus = CaseStatus.INTAKE
    
    # Court info
    court_name: Optional[str] = None
    court_division: Optional[str] = None
    venue: Optional[str] = None
    judge_name: Optional[str] = None
    
    # Dates
    incident_date: Optional[datetime] = None
    filing_date: Optional[datetime] = None
    statute_of_limitations: Optional[datetime] = None
    
    # Content
    summary: Optional[str] = None
    facts: Optional[str] = None
    claims: Optional[List[str]] = None
    
    # Financial
    estimated_damages: Optional[float] = None
    
    # Meta
    priority: int = Field(default=3, ge=1, le=5)
    notes: Optional[str] = None


class CaseCreate(CaseBase):
    """Schema for creating a new case"""
    pass


class CaseUpdate(BaseModel):
    """Schema for updating a case"""
    title: Optional[str] = None
    short_title: Optional[str] = None
    case_number: Optional[str] = None
    docket_number: Optional[str] = None
    case_type: Optional[CaseType] = None
    status: Optional[CaseStatus] = None
    court_name: Optional[str] = None
    court_division: Optional[str] = None
    venue: Optional[str] = None
    judge_name: Optional[str] = None
    incident_date: Optional[datetime] = None
    filing_date: Optional[datetime] = None
    statute_of_limitations: Optional[datetime] = None
    next_deadline: Optional[datetime] = None
    summary: Optional[str] = None
    facts: Optional[str] = None
    claims: Optional[List[str]] = None
    estimated_damages: Optional[float] = None
    settlement_amount: Optional[float] = None
    priority: Optional[int] = Field(None, ge=1, le=5)
    notes: Optional[str] = None
    is_active: Optional[bool] = None
    is_archived: Optional[bool] = None


class CaseResponse(CaseBase):
    """Schema for case responses"""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    internal_reference: Optional[str] = None
    settlement_amount: Optional[float] = None
    verdict_amount: Optional[float] = None
    attorney_fees: Optional[float] = None
    next_deadline: Optional[datetime] = None
    is_active: bool
    is_archived: bool
    created_at: datetime
    updated_at: datetime
    
    # Counts
    document_count: Optional[int] = 0
    evidence_count: Optional[int] = 0
    violation_count: Optional[int] = 0


class CaseListResponse(BaseModel):
    """Schema for paginated case list"""
    items: List[CaseResponse]
    total: int
    page: int
    per_page: int
    pages: int


# ============================================================
# PARTY SCHEMAS
# ============================================================

class PartyBase(BaseModel):
    """Base schema for parties"""
    name: str = Field(..., min_length=1, max_length=255)
    role: PartyRole
    
    # Contact
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    
    # Officer/Official fields
    badge_number: Optional[str] = None
    rank: Optional[str] = None
    department: Optional[str] = None
    unit: Optional[str] = None
    
    # Attorney fields
    bar_number: Optional[str] = None
    firm_name: Optional[str] = None
    
    notes: Optional[str] = None


class PartyCreate(PartyBase):
    """Schema for creating a party"""
    case_id: int


class PartyResponse(PartyBase):
    """Schema for party responses"""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    case_id: int
    prior_incidents: Optional[List[Dict[str, Any]]] = None
    created_at: datetime
    updated_at: datetime


# ============================================================
# DOCUMENT SCHEMAS
# ============================================================

class DocumentBase(BaseModel):
    """Base schema for documents"""
    title: Optional[str] = None
    description: Optional[str] = None
    document_type: DocumentType = DocumentType.OTHER
    document_date: Optional[datetime] = None
    
    # Legal metadata
    bates_start: Optional[str] = None
    bates_end: Optional[str] = None
    is_privileged: bool = False
    privilege_type: Optional[str] = None
    is_confidential: bool = False


class DocumentUploadResponse(BaseModel):
    """Response after uploading documents"""
    id: int
    filename: str
    original_filename: str
    file_size: int
    mime_type: str
    file_hash: str
    processing_status: ProcessingStatus
    message: str


class BatchUploadResponse(BaseModel):
    """Response for batch document upload"""
    uploaded: List[DocumentUploadResponse]
    failed: List[Dict[str, str]]
    total_uploaded: int
    total_failed: int


class DocumentResponse(DocumentBase):
    """Schema for document responses"""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    filename: str
    original_filename: str
    file_path: str
    file_size: int
    mime_type: str
    file_hash: str
    
    # Processing
    processing_status: ProcessingStatus
    processed_at: Optional[datetime] = None
    page_count: Optional[int] = None
    ocr_confidence: Optional[float] = None
    
    # Extracted data
    extracted_text: Optional[str] = None
    entities_extracted: Optional[Dict[str, Any]] = None
    key_phrases: Optional[List[str]] = None
    
    created_at: datetime
    updated_at: datetime


class DocumentClassifyRequest(BaseModel):
    """Request to classify documents"""
    document_ids: List[int]
    auto_tag: bool = True


class DocumentClassifyResponse(BaseModel):
    """Response from document classification"""
    document_id: int
    predicted_type: DocumentType
    confidence: float
    suggested_tags: List[str]
    extracted_date: Optional[datetime] = None


# ============================================================
# EVIDENCE / BWC SCHEMAS
# ============================================================

class EvidenceBase(BaseModel):
    """Base schema for evidence"""
    title: Optional[str] = None
    description: Optional[str] = None
    evidence_type: EvidenceType = EvidenceType.OTHER
    
    # Chain of custody
    collected_by: Optional[str] = None
    collected_date: Optional[datetime] = None


class BWCMetadata(BaseModel):
    """Metadata extracted from BWC filename and file"""
    officer_name: str
    timestamp: datetime
    device_id: str
    segment: int = 0
    duration_seconds: Optional[float] = None
    resolution: Optional[str] = None
    frame_rate: Optional[float] = None
    file_size: int
    file_hash: str


class BWCUploadResponse(BaseModel):
    """Response after uploading BWC footage"""
    id: int
    filename: str
    original_filename: str
    evidence_type: EvidenceType
    metadata: BWCMetadata
    processing_status: ProcessingStatus
    message: str


class BatchBWCUploadResponse(BaseModel):
    """Response for batch BWC upload"""
    uploaded: List[BWCUploadResponse]
    failed: List[Dict[str, str]]
    total_uploaded: int
    total_failed: int
    sync_group_id: Optional[str] = None


class EvidenceResponse(EvidenceBase):
    """Schema for evidence responses"""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    filename: str
    original_filename: str
    file_path: str
    file_size: int
    mime_type: str
    file_hash: str
    
    # Media metadata
    duration_seconds: Optional[float] = None
    start_timestamp: Optional[datetime] = None
    end_timestamp: Optional[datetime] = None
    frame_rate: Optional[float] = None
    resolution: Optional[str] = None
    codec: Optional[str] = None
    
    # BWC specific
    officer_name: Optional[str] = None
    officer_badge: Optional[str] = None
    bwc_device_id: Optional[str] = None
    motorola_event_id: Optional[str] = None
    
    # Processing
    processing_status: ProcessingStatus
    processed_at: Optional[datetime] = None
    transcript: Optional[str] = None
    key_frames: Optional[List[Dict[str, Any]]] = None
    
    # Sync
    sync_group_id: Optional[str] = None
    sync_offset_ms: int = 0
    is_primary_pov: bool = False
    
    created_at: datetime
    updated_at: datetime


class VideoSyncRequest(BaseModel):
    """Request to synchronize multiple videos"""
    evidence_ids: List[int]
    primary_evidence_id: int
    sync_method: str = "audio"  # audio, timestamp, manual
    manual_offsets: Optional[Dict[int, int]] = None  # evidence_id -> offset_ms


class VideoSyncResponse(BaseModel):
    """Response from video synchronization"""
    sync_group_id: str
    primary_evidence_id: int
    synced_videos: List[Dict[str, Any]]
    sync_quality: float
    timeline_start: datetime
    timeline_end: datetime


class AudioHarmonizeRequest(BaseModel):
    """Request to harmonize audio from evidence"""
    evidence_ids: List[int]
    normalize_levels: bool = True
    reduce_noise: bool = True
    enhance_speech: bool = True
    output_format: str = "wav"


class AudioHarmonizeResponse(BaseModel):
    """Response from audio harmonization"""
    output_file_id: int
    output_filename: str
    sources: List[int]
    processing_notes: List[str]


# ============================================================
# VIOLATION / ANALYSIS SCHEMAS
# ============================================================

class ViolationBase(BaseModel):
    """Base schema for violations"""
    violation_type: ViolationType
    severity: int = Field(default=3, ge=1, le=5)
    title: str
    description: Optional[str] = None
    legal_basis: Optional[str] = None
    
    # Source reference
    timestamp_start: Optional[str] = None
    timestamp_end: Optional[str] = None
    page_reference: Optional[str] = None


class ViolationCreate(ViolationBase):
    """Schema for creating a violation"""
    case_id: int
    source_document_id: Optional[int] = None
    source_evidence_id: Optional[int] = None


class ViolationResponse(ViolationBase):
    """Schema for violation responses"""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    case_id: int
    source_document_id: Optional[int] = None
    source_evidence_id: Optional[int] = None
    
    confidence_score: Optional[float] = None
    is_verified: bool
    verified_by: Optional[str] = None
    verified_at: Optional[datetime] = None
    
    liability_score: Optional[float] = None
    estimated_damages: Optional[float] = None
    supporting_citations: Optional[List[str]] = None
    
    created_at: datetime
    updated_at: datetime


class ConstitutionalAnalysisRequest(BaseModel):
    """Request for constitutional analysis"""
    case_id: int
    document_ids: Optional[List[int]] = None
    evidence_ids: Optional[List[int]] = None
    scan_types: Optional[List[ViolationType]] = None  # None = scan all
    include_suggestions: bool = True


class ConstitutionalAnalysisResponse(BaseModel):
    """Response from constitutional analysis"""
    case_id: int
    analysis_id: int
    violations_found: List[ViolationResponse]
    total_violations: int
    severity_breakdown: Dict[str, int]
    type_breakdown: Dict[str, int]
    liability_score: float
    estimated_total_damages: float
    recommendations: List[str]
    processing_time_seconds: float


class LiabilityAssessmentRequest(BaseModel):
    """Request for liability assessment"""
    case_id: int
    include_officer_history: bool = True
    include_department_patterns: bool = True
    comparable_cases: bool = True


class LiabilityAssessmentResponse(BaseModel):
    """Response from liability assessment"""
    case_id: int
    overall_liability_score: float  # 0-100
    
    # Breakdown
    constitutional_violations_score: float
    evidence_strength_score: float
    officer_history_score: float
    department_pattern_score: float
    
    # Financial
    estimated_damages_low: float
    estimated_damages_mid: float
    estimated_damages_high: float
    comparable_settlements: List[Dict[str, Any]]
    
    # Recommendations
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]


class OfficerHistoryRequest(BaseModel):
    """Request for officer history analysis"""
    badge_number: Optional[str] = None
    officer_name: Optional[str] = None
    department: Optional[str] = None


class OfficerHistoryResponse(BaseModel):
    """Response from officer history analysis"""
    officer_name: str
    badge_number: Optional[str] = None
    department: str
    
    # Statistics
    total_incidents: int
    total_complaints: int
    sustained_complaints: int
    total_lawsuits: int
    total_settlements: float
    
    # Pattern analysis
    violation_patterns: Dict[str, int]
    risk_score: float
    
    # Related cases
    related_case_ids: List[int]
    incident_timeline: List[Dict[str, Any]]


# ============================================================
# TIMELINE SCHEMAS
# ============================================================

class TimelineEventBase(BaseModel):
    """Base schema for timeline events"""
    event_date: datetime
    event_time: Optional[str] = None
    title: str
    description: Optional[str] = None
    event_category: Optional[str] = None
    is_key_event: bool = False


class TimelineEventCreate(TimelineEventBase):
    """Schema for creating timeline events"""
    case_id: int
    source_document_id: Optional[int] = None
    source_evidence_id: Optional[int] = None


class TimelineEventResponse(TimelineEventBase):
    """Schema for timeline event responses"""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    case_id: int
    source_document_id: Optional[int] = None
    source_evidence_id: Optional[int] = None
    source_description: Optional[str] = None
    created_at: datetime


class CaseTimelineResponse(BaseModel):
    """Full timeline for a case"""
    case_id: int
    case_title: str
    events: List[TimelineEventResponse]
    total_events: int
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None


# ============================================================
# OPRA SCHEMAS
# ============================================================

class OPRARequestBase(BaseModel):
    """Base schema for OPRA requests"""
    agency: str
    records_requested: str
    purpose: Optional[str] = None
    custodian_name: Optional[str] = None
    custodian_email: Optional[str] = None


class OPRARequestCreate(OPRARequestBase):
    """Schema for creating OPRA requests"""
    case_id: Optional[int] = None
    date_submitted: datetime


class OPRARequestUpdate(BaseModel):
    """Schema for updating OPRA requests"""
    status: Optional[str] = None
    date_due: Optional[datetime] = None
    date_received: Optional[datetime] = None
    denial_reason: Optional[str] = None
    documents_received: Optional[int] = None
    response_notes: Optional[str] = None
    is_appealed: Optional[bool] = None
    appeal_date: Optional[datetime] = None
    grc_case_number: Optional[str] = None


class OPRARequestResponse(OPRARequestBase):
    """Schema for OPRA request responses"""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    case_id: Optional[int] = None
    request_number: Optional[str] = None
    date_submitted: datetime
    date_due: Optional[datetime] = None
    date_received: Optional[datetime] = None
    status: str
    denial_reason: Optional[str] = None
    documents_received: Optional[int] = None
    response_notes: Optional[str] = None
    is_appealed: bool
    appeal_date: Optional[datetime] = None
    grc_case_number: Optional[str] = None
    created_at: datetime
    updated_at: datetime


# ============================================================
# EXPORT SCHEMAS
# ============================================================

class ExportTimelineRequest(BaseModel):
    """Request to export case timeline"""
    case_id: int
    format: str = "pdf"  # pdf, docx, csv
    include_sources: bool = True
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None


class ExportViolationsRequest(BaseModel):
    """Request to export violations report"""
    case_id: int
    format: str = "pdf"
    include_evidence_refs: bool = True
    include_legal_basis: bool = True
    include_damages_estimate: bool = True


class ExportEvidenceBinderRequest(BaseModel):
    """Request to compile evidence binder"""
    case_id: int
    document_ids: Optional[List[int]] = None
    evidence_ids: Optional[List[int]] = None
    include_index: bool = True
    include_timeline: bool = True
    bates_stamp: bool = True
    bates_prefix: str = "EX"


class ExportResponse(BaseModel):
    """Response for export operations"""
    export_id: int
    filename: str
    file_path: str
    file_size: int
    format: str
    created_at: datetime
    download_url: str


# ============================================================
# SEARCH SCHEMAS
# ============================================================

class SearchRequest(BaseModel):
    """Full-text search request"""
    query: str
    case_ids: Optional[List[int]] = None
    document_types: Optional[List[DocumentType]] = None
    evidence_types: Optional[List[EvidenceType]] = None
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    limit: int = Field(default=50, le=200)
    offset: int = 0


class SearchResult(BaseModel):
    """Individual search result"""
    result_type: str  # document, evidence, case, violation
    id: int
    title: str
    snippet: str
    relevance_score: float
    case_id: Optional[int] = None
    case_title: Optional[str] = None
    highlights: List[str]


class SearchResponse(BaseModel):
    """Search response"""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: int
