"""
BarberX Legal Case Management Pro Suite
SQLAlchemy Database Models
"""
from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, 
    Text, Boolean, JSON, Enum as SQLEnum, Table, LargeBinary
)
from sqlalchemy.orm import relationship
from app.db.database import Base
import enum


# ============================================================
# ENUMS
# ============================================================

class CaseStatus(str, enum.Enum):
    INTAKE = "intake"
    INVESTIGATION = "investigation"
    PRE_LITIGATION = "pre_litigation"
    LITIGATION = "litigation"
    DISCOVERY = "discovery"
    TRIAL = "trial"
    APPEAL = "appeal"
    SETTLED = "settled"
    CLOSED = "closed"


class CaseType(str, enum.Enum):
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


class DocumentType(str, enum.Enum):
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
    OTHER = "other"


class EvidenceType(str, enum.Enum):
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


class ViolationType(str, enum.Enum):
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


class ProcessingStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_REVIEW = "requires_review"


class PartyRole(str, enum.Enum):
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
# ASSOCIATION TABLES
# ============================================================

case_documents = Table(
    'case_documents',
    Base.metadata,
    Column('case_id', Integer, ForeignKey('cases.id'), primary_key=True),
    Column('document_id', Integer, ForeignKey('documents.id'), primary_key=True)
)

case_evidence = Table(
    'case_evidence',
    Base.metadata,
    Column('case_id', Integer, ForeignKey('cases.id'), primary_key=True),
    Column('evidence_id', Integer, ForeignKey('evidence.id'), primary_key=True)
)

document_tags = Table(
    'document_tags',
    Base.metadata,
    Column('document_id', Integer, ForeignKey('documents.id'), primary_key=True),
    Column('tag_id', Integer, ForeignKey('tags.id'), primary_key=True)
)


# ============================================================
# CORE MODELS
# ============================================================

class Case(Base):
    """Legal Case / Matter"""
    __tablename__ = "cases"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Case identifiers
    case_number = Column(String(100), unique=True, index=True)
    docket_number = Column(String(100), index=True)
    internal_reference = Column(String(100), index=True)
    
    # Basic info
    title = Column(String(500), nullable=False)
    short_title = Column(String(100))
    case_type = Column(SQLEnum(CaseType), default=CaseType.CIVIL_RIGHTS)
    status = Column(SQLEnum(CaseStatus), default=CaseStatus.INTAKE)
    
    # Court info
    court_name = Column(String(255))
    court_division = Column(String(100))
    venue = Column(String(100))
    judge_name = Column(String(255))
    
    # Dates
    incident_date = Column(DateTime)
    filing_date = Column(DateTime)
    statute_of_limitations = Column(DateTime)
    next_deadline = Column(DateTime)
    
    # Summary
    summary = Column(Text)
    facts = Column(Text)
    claims = Column(JSON)  # List of claims/causes of action
    
    # Financial
    estimated_damages = Column(Float)
    settlement_amount = Column(Float)
    verdict_amount = Column(Float)
    attorney_fees = Column(Float)
    
    # Metadata
    priority = Column(Integer, default=3)  # 1=Urgent, 5=Low
    is_active = Column(Boolean, default=True)
    is_archived = Column(Boolean, default=False)
    notes = Column(Text)
    custom_fields = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    parties = relationship("Party", back_populates="case", cascade="all, delete-orphan")
    documents = relationship("Document", secondary=case_documents, back_populates="cases")
    evidence = relationship("Evidence", secondary=case_evidence, back_populates="cases")
    violations = relationship("Violation", back_populates="case", cascade="all, delete-orphan")
    timeline_events = relationship("TimelineEvent", back_populates="case", cascade="all, delete-orphan")
    opra_requests = relationship("OPRARequest", back_populates="case", cascade="all, delete-orphan")


class Party(Base):
    """Party to a case (plaintiff, defendant, witness, officer, etc.)"""
    __tablename__ = "parties"
    
    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=False)
    
    # Identity
    name = Column(String(255), nullable=False, index=True)
    role = Column(SQLEnum(PartyRole), nullable=False)
    
    # Contact
    email = Column(String(255))
    phone = Column(String(50))
    address = Column(Text)
    
    # For officers/officials
    badge_number = Column(String(50), index=True)
    rank = Column(String(100))
    department = Column(String(255), index=True)
    unit = Column(String(100))
    
    # For attorneys
    bar_number = Column(String(50))
    firm_name = Column(String(255))
    
    # Notes
    notes = Column(Text)
    prior_incidents = Column(JSON)  # List of known prior incidents
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    case = relationship("Case", back_populates="parties")


class Document(Base):
    """Legal document (PDF, filing, report, etc.)"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # File info
    filename = Column(String(500), nullable=False)
    original_filename = Column(String(500))
    file_path = Column(String(1000), nullable=False)
    file_size = Column(Integer)
    mime_type = Column(String(100))
    file_hash = Column(String(64), index=True)  # SHA-256 for integrity
    
    # Classification
    document_type = Column(SQLEnum(DocumentType), default=DocumentType.OTHER)
    title = Column(String(500))
    description = Column(Text)
    
    # Extracted content
    extracted_text = Column(Text)
    ocr_confidence = Column(Float)
    page_count = Column(Integer)
    
    # Processing
    processing_status = Column(SQLEnum(ProcessingStatus), default=ProcessingStatus.PENDING)
    processed_at = Column(DateTime)
    processing_notes = Column(Text)
    
    # Legal metadata
    document_date = Column(DateTime)
    bates_start = Column(String(50))  # Bates number range
    bates_end = Column(String(50))
    is_privileged = Column(Boolean, default=False)
    privilege_type = Column(String(100))  # Attorney-client, work product, etc.
    is_confidential = Column(Boolean, default=False)
    
    # Analysis results
    entities_extracted = Column(JSON)  # Named entities (people, dates, places)
    key_phrases = Column(JSON)
    sentiment_score = Column(Float)
    relevance_score = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    cases = relationship("Case", secondary=case_documents, back_populates="documents")
    tags = relationship("Tag", secondary=document_tags, back_populates="documents")
    violations_found = relationship("Violation", back_populates="source_document")


class Evidence(Base):
    """Physical or digital evidence (BWC footage, photos, recordings)"""
    __tablename__ = "evidence"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # File info
    filename = Column(String(500), nullable=False)
    original_filename = Column(String(500))
    file_path = Column(String(1000), nullable=False)
    file_size = Column(Integer)
    mime_type = Column(String(100))
    file_hash = Column(String(64), index=True)
    
    # Classification
    evidence_type = Column(SQLEnum(EvidenceType), default=EvidenceType.OTHER)
    title = Column(String(500))
    description = Column(Text)
    
    # Chain of custody
    collected_by = Column(String(255))
    collected_date = Column(DateTime)
    chain_of_custody = Column(JSON)  # List of custody transfers
    
    # Media metadata (for video/audio)
    duration_seconds = Column(Float)
    start_timestamp = Column(DateTime)
    end_timestamp = Column(DateTime)
    frame_rate = Column(Float)
    resolution = Column(String(50))
    codec = Column(String(50))
    
    # BWC specific
    officer_name = Column(String(255), index=True)
    officer_badge = Column(String(50), index=True)
    bwc_device_id = Column(String(100))
    bwc_serial = Column(String(100))
    motorola_event_id = Column(String(100))  # Motorola Solutions reference
    
    # Processing
    processing_status = Column(SQLEnum(ProcessingStatus), default=ProcessingStatus.PENDING)
    processed_at = Column(DateTime)
    
    # Analysis results
    transcript = Column(Text)
    transcript_confidence = Column(Float)
    transcript_language = Column(String(10), default="en")
    transcript_uploaded_at = Column(DateTime)
    key_frames = Column(JSON)  # List of important frame timestamps
    detected_events = Column(JSON)  # AI-detected events
    audio_analysis = Column(JSON)  # Audio analysis results
    ai_analysis = Column(JSON)  # Full AI analysis results
    force_incidents = Column(JSON)  # Detected use of force
    
    # Synchronization (for multi-POV)
    sync_group_id = Column(String(100))
    sync_offset_ms = Column(Integer, default=0)
    is_primary_pov = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    cases = relationship("Case", secondary=case_evidence, back_populates="evidence")
    violations_found = relationship("Violation", back_populates="source_evidence")


class Violation(Base):
    """Constitutional violation or legal issue identified"""
    __tablename__ = "violations"
    
    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=False)
    
    # Classification
    violation_type = Column(SQLEnum(ViolationType), nullable=False)
    severity = Column(Integer, default=3)  # 1=Minor, 5=Severe
    
    # Details
    title = Column(String(500))
    description = Column(Text)
    legal_basis = Column(Text)  # Relevant case law, statutes
    
    # Evidence linking
    source_document_id = Column(Integer, ForeignKey("documents.id"))
    source_evidence_id = Column(Integer, ForeignKey("evidence.id"))
    timestamp_start = Column(String(50))  # For video: HH:MM:SS
    timestamp_end = Column(String(50))
    page_reference = Column(String(50))  # For documents
    
    # Analysis
    confidence_score = Column(Float)  # AI confidence
    is_verified = Column(Boolean, default=False)  # Human verified
    verified_by = Column(String(255))
    verified_at = Column(DateTime)
    
    # Impact
    liability_score = Column(Float)
    estimated_damages = Column(Float)
    
    # Notes
    notes = Column(Text)
    supporting_citations = Column(JSON)  # List of case citations
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    case = relationship("Case", back_populates="violations")
    source_document = relationship("Document", back_populates="violations_found")
    source_evidence = relationship("Evidence", back_populates="violations_found")


class TimelineEvent(Base):
    """Event in case timeline"""
    __tablename__ = "timeline_events"
    
    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=False)
    
    # Event details
    event_date = Column(DateTime, nullable=False)
    event_time = Column(String(20))  # For precision
    title = Column(String(500), nullable=False)
    description = Column(Text)
    
    # Categorization
    event_category = Column(String(100))  # Incident, Filing, Hearing, etc.
    is_key_event = Column(Boolean, default=False)
    
    # Source
    source_document_id = Column(Integer, ForeignKey("documents.id"))
    source_evidence_id = Column(Integer, ForeignKey("evidence.id"))
    source_description = Column(String(500))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    case = relationship("Case", back_populates="timeline_events")


class OPRARequest(Base):
    """NJ Open Public Records Act request tracking"""
    __tablename__ = "opra_requests"
    
    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id"))
    
    # Request details
    request_number = Column(String(100), index=True)
    agency = Column(String(255), nullable=False)
    custodian_name = Column(String(255))
    custodian_email = Column(String(255))
    
    # Content
    records_requested = Column(Text, nullable=False)
    purpose = Column(Text)
    
    # Dates
    date_submitted = Column(DateTime, nullable=False)
    date_due = Column(DateTime)
    date_received = Column(DateTime)
    
    # Status
    status = Column(String(100), default="pending")  # pending, extended, denied, received, appealed
    denial_reason = Column(Text)
    
    # Response
    documents_received = Column(Integer)
    response_notes = Column(Text)
    
    # Appeal
    is_appealed = Column(Boolean, default=False)
    appeal_date = Column(DateTime)
    grc_case_number = Column(String(100))  # Government Records Council
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    case = relationship("Case", back_populates="opra_requests")


class Tag(Base):
    """Tags for organizing documents and evidence"""
    __tablename__ = "tags"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    color = Column(String(20))  # Hex color for UI
    description = Column(String(500))
    
    # Relationships
    documents = relationship("Document", secondary=document_tags, back_populates="tags")


class Officer(Base):
    """Officer database for tracking across cases"""
    __tablename__ = "officers"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Identity
    name = Column(String(255), nullable=False, index=True)
    badge_number = Column(String(50), index=True)
    department = Column(String(255), index=True)
    rank = Column(String(100))
    unit = Column(String(100))
    
    # Employment
    hire_date = Column(DateTime)
    termination_date = Column(DateTime)
    current_status = Column(String(100))  # Active, Suspended, Terminated, etc.
    
    # History
    total_complaints = Column(Integer, default=0)
    sustained_complaints = Column(Integer, default=0)
    civil_suits = Column(Integer, default=0)
    settlements_total = Column(Float, default=0)
    
    # Related cases (JSON list of case_ids)
    related_case_ids = Column(JSON)
    
    # Notes
    notes = Column(Text)
    prior_incidents = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AnalysisJob(Base):
    """Background analysis job tracking"""
    __tablename__ = "analysis_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Job info
    job_type = Column(String(100), nullable=False)  # ocr, constitutional, sync, etc.
    status = Column(SQLEnum(ProcessingStatus), default=ProcessingStatus.PENDING)
    
    # Target
    case_id = Column(Integer, ForeignKey("cases.id"))
    document_id = Column(Integer, ForeignKey("documents.id"))
    evidence_id = Column(Integer, ForeignKey("evidence.id"))
    
    # Progress
    progress_percent = Column(Integer, default=0)
    current_step = Column(String(255))
    
    # Results
    results = Column(JSON)
    error_message = Column(Text)
    
    # Timing
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    """Audit trail for all actions"""
    __tablename__ = "audit_log"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Action details
    action = Column(String(100), nullable=False)  # CREATE, UPDATE, DELETE, VIEW, EXPORT
    entity_type = Column(String(100), nullable=False)
    entity_id = Column(Integer)
    
    # User
    user_id = Column(String(100))
    user_name = Column(String(255))
    ip_address = Column(String(50))
    
    # Details
    old_values = Column(JSON)
    new_values = Column(JSON)
    notes = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
