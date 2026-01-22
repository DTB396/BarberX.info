"""
BarberX Legal Case Management Pro Suite
Services Module
"""
from app.services.ai_service import ai_service, AIService, DocumentAnalysis, AIConversation
from app.services.bwc_analysis_service import (
    bwc_analysis_service, 
    BWCAnalysisService, 
    BWCVideoInfo,
    BWCAnalysisResult,
    RealTimeSyncManager,
    SyncTimeline
)
from app.services.transcription_service import (
    transcription_service,
    TranscriptionService,
    TranscriptionResult,
    BatchTranscriptionJob,
    AudioPreprocessor
)
from app.services.audio_analysis_service import (
    audio_analyzer,
    audio_enhancer,
    multi_track_mixer,
    AdvancedAudioAnalyzer,
    AdvancedAudioEnhancer,
    MultiTrackMixer,
    AudioAnalysisResult,
    AudioEnhancementResult
)

# Premium Legal Analysis Services
from app.services.discovery_service import (
    discovery_service,
    EDiscoveryService,
    PrivilegeType,
    ReviewStatus,
    BatesRange,
    PrivilegeLogEntry,
    DocumentReview,
    ProductionSet
)
from app.services.deposition_service import (
    deposition_service,
    DepositionService,
    WitnessType,
    CredibilityFactor,
    WitnessProfile,
    DepositionDigest,
    ImpeachmentMaterial,
    ExpertWitness
)
from app.services.case_strategy_service import (
    case_strategy_service,
    CaseStrategyService,
    RiskLevel,
    LiabilityAssessment,
    DamagesAnalysis,
    SettlementAnalysis,
    CaseTheory,
    JuryProfile
)
from app.services.deadline_service import (
    deadline_calculator,
    DeadlineCalculator,
    CourtRule,
    Deadline,
    DeadlineChain
)
from app.services.brady_service import (
    brady_service,
    BradyTrackingService,
    EvidenceClassification,
    ViolationType,
    BradyEvidence,
    OfficerCredibilityRecord,
    BradyViolation
)

# Firm Management Services
from app.services.conflict_service import (
    conflict_service,
    ConflictCheckService,
    ConflictType,
    ConflictSeverity,
    EntityType,
    Entity,
    Matter,
    ConflictHit,
    ConflictCheck,
    Waiver
)
from app.services.billing_service import (
    billing_service,
    BillingService,
    BillingStatus,
    ExpenseType,
    TaskCode,
    Timekeeper,
    TimeEntry,
    Expense,
    BillingGuideline,
    Invoice,
    BudgetTracker
)
from app.services.research_service import (
    research_service,
    LegalResearchService,
    JurisdictionType,
    CaseStatus,
    ResearchType,
    LegalCitation,
    CitationAnalysis,
    ResearchMemo,
    KeyIssue,
    MotionBank
)

__all__ = [
    # AI Services
    "ai_service",
    "AIService", 
    "DocumentAnalysis",
    "AIConversation",
    
    # BWC Analysis
    "bwc_analysis_service",
    "BWCAnalysisService",
    "BWCVideoInfo",
    "BWCAnalysisResult",
    "RealTimeSyncManager",
    "SyncTimeline",
    
    # Transcription
    "transcription_service",
    "TranscriptionService",
    "TranscriptionResult",
    "BatchTranscriptionJob",
    "AudioPreprocessor",
    
    # Audio Analysis
    "audio_analyzer",
    "audio_enhancer",
    "multi_track_mixer",
    "AdvancedAudioAnalyzer",
    "AdvancedAudioEnhancer",
    "MultiTrackMixer",
    "AudioAnalysisResult",
    "AudioEnhancementResult",
    
    # Premium Legal: E-Discovery
    "discovery_service",
    "EDiscoveryService",
    "PrivilegeType",
    "ReviewStatus",
    "BatesRange",
    "PrivilegeLogEntry",
    "DocumentReview",
    "ProductionSet",
    
    # Premium Legal: Depositions
    "deposition_service",
    "DepositionService",
    "WitnessType",
    "CredibilityFactor",
    "WitnessProfile",
    "DepositionDigest",
    "ImpeachmentMaterial",
    "ExpertWitness",
    
    # Premium Legal: Case Strategy
    "case_strategy_service",
    "CaseStrategyService",
    "RiskLevel",
    "LiabilityAssessment",
    "DamagesAnalysis",
    "SettlementAnalysis",
    "CaseTheory",
    "JuryProfile",
    
    # Premium Legal: Deadlines
    "deadline_calculator",
    "DeadlineCalculator",
    "CourtRule",
    "Deadline",
    "DeadlineChain",
    
    # Premium Legal: Brady/Giglio
    "brady_service",
    "BradyTrackingService",
    "EvidenceClassification",
    "ViolationType",
    "BradyEvidence",
    "OfficerCredibilityRecord",
    "BradyViolation",
    
    # Firm Management: Conflicts
    "conflict_service",
    "ConflictCheckService",
    "ConflictType",
    "ConflictSeverity",
    "EntityType",
    "Entity",
    "Matter",
    "ConflictHit",
    "ConflictCheck",
    "Waiver",
    
    # Firm Management: Billing
    "billing_service",
    "BillingService",
    "BillingStatus",
    "ExpenseType",
    "TaskCode",
    "Timekeeper",
    "TimeEntry",
    "Expense",
    "BillingGuideline",
    "Invoice",
    "BudgetTracker",
    
    # Firm Management: Research
    "research_service",
    "LegalResearchService",
    "JurisdictionType",
    "CaseStatus",
    "ResearchType",
    "LegalCitation",
    "CitationAnalysis",
    "ResearchMemo",
    "KeyIssue",
    "MotionBank",
]
