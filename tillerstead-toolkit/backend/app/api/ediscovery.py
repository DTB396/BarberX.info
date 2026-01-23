"""
eDiscovery API Router - Complete Court-Defensible Evidence Management
======================================================================

Endpoints for:
- Evidence vault operations
- BWC/video processing
- CAD/dispatch processing
- Document discovery
- Unified search
- Production (clips, exhibits, load files)
- Monitoring and alerts

Author: BarberX Legal Case Management
Version: 1.0.0
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Query, Body
from fastapi.responses import FileResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from pathlib import Path

# Import services
from app.services.evidence_vault_service import evidence_vault, EvidenceType, SourceSystem, ExportMethod, AccessLevel, EvidenceProvenance, NegativeEvidenceType
from app.services.bwc_processor_service import bwc_processor, BWCPlatform
from app.services.cad_processor_service import cad_processor, CADSystem
from app.services.document_processor_service import document_processor, DocumentType
from app.services.ediscovery_platform_service import unified_search, production_service, monitoring_service, SearchQuery
from app.services.media_enhancement_service import media_enhancer, EnhancementQuality, AudioNoiseProfile


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class IngestEvidenceRequest(BaseModel):
    case_id: str
    evidence_type: str  # EvidenceType value
    source_system: str
    export_method: str
    received_from: str
    custodian: str
    collection_date: str
    description: str = ""
    tags: List[str] = []
    user: str = "System"
    user_role: str = "Administrator"


class ProcessBWCRequest(BaseModel):
    evidence_id: str
    perform_transcription: bool = True
    perform_scene_detection: bool = True
    perform_gps_extraction: bool = True


class ProcessCADRequest(BaseModel):
    evidence_id: str
    case_id: str
    expected_incident_id: Optional[str] = None


class ProcessDocumentRequest(BaseModel):
    evidence_id: str
    case_id: str
    perform_ocr: bool = True
    apply_bates: bool = False
    bates_prefix: str = "DEF"


class RecordNegativeEvidenceRequest(BaseModel):
    case_id: str
    evidence_type: str  # NegativeEvidenceType value
    responding_agency: str
    responding_custodian: str
    request_scope: str
    response_text: str
    request_date: Optional[str] = None
    response_date: Optional[str] = None


class UnifiedSearchRequest(BaseModel):
    query_text: str
    case_id: Optional[str] = None
    evidence_types: List[str] = []
    date_range: Optional[List[str]] = None
    custodians: List[str] = []
    tags: List[str] = []
    issue_codes: List[str] = []


class CreateVideoClipRequest(BaseModel):
    source_evidence_id: str
    start_time: float
    end_time: float
    description: str = ""
    created_by: str = "System"


class CreateExhibitPackRequest(BaseModel):
    case_id: str
    title: str
    clip_ids: List[str] = []
    document_ids: List[str] = []


class GenerateLoadFileRequest(BaseModel):
    case_id: str
    evidence_ids: List[str]
    production_name: str


# ============================================================================
# API ROUTER
# ============================================================================

router = APIRouter(
    prefix="/api/v1/ediscovery",
    tags=["eDiscovery"]
)


# ============================================================================
# EVIDENCE VAULT ENDPOINTS
# ============================================================================

@router.post("/vault/ingest")
async def ingest_evidence(
    file: UploadFile = File(...),
    request_data: str = Body(...)
):
    """
    Ingest evidence file with full chain of custody
    
    Returns:
        EvidenceItem with vault path and SHA-256 hash
    """
    import json
    req = IngestEvidenceRequest(**json.loads(request_data))
    
    # Save uploaded file temporarily
    temp_path = Path(f"/tmp/{file.filename}")
    with open(temp_path, 'wb') as f:
        f.write(await file.read())
    
    # Create provenance
    provenance = EvidenceProvenance(
        source_system=SourceSystem(req.source_system),
        export_method=ExportMethod(req.export_method),
        received_from=req.received_from,
        custodian=req.custodian,
        collection_date=req.collection_date,
        received_date=datetime.utcnow().isoformat(),
        original_filename=file.filename,
        original_location=str(temp_path)
    )
    
    # Ingest
    evidence = await evidence_vault.ingest_evidence(
        file_path=temp_path,
        case_id=req.case_id,
        evidence_type=EvidenceType(req.evidence_type),
        provenance=provenance,
        description=req.description,
        tags=req.tags,
        user=req.user,
        user_role=req.user_role
    )
    
    return evidence.to_dict()


@router.post("/vault/verify/{evidence_id}")
async def verify_evidence(evidence_id: str, user: str = "System"):
    """Verify evidence hash integrity"""
    is_valid = await evidence_vault.verify_evidence_integrity(evidence_id, user)
    return {
        "evidence_id": evidence_id,
        "integrity_valid": is_valid,
        "verified_date": datetime.utcnow().isoformat()
    }


@router.post("/vault/litigation-hold")
async def apply_litigation_hold(
    case_id: str,
    evidence_ids: List[str],
    reason: str,
    user: str,
    user_role: str = "Attorney"
):
    """Apply litigation hold to evidence items"""
    await evidence_vault.apply_litigation_hold(case_id, evidence_ids, reason, user, user_role)
    return {
        "case_id": case_id,
        "evidence_ids": evidence_ids,
        "hold_applied": True,
        "date": datetime.utcnow().isoformat()
    }


@router.get("/vault/chain-report/{evidence_id}")
async def get_chain_of_custody_report(evidence_id: str):
    """Generate printable chain of custody report"""
    report = await evidence_vault.generate_chain_report(evidence_id)
    return report


# ============================================================================
# BWC PROCESSING ENDPOINTS
# ============================================================================

@router.post("/bwc/process")
async def process_bwc_video(request: ProcessBWCRequest):
    """
    Process BWC video with transcription, scene detection, and GPS extraction
    
    Returns:
        Complete BWC processing result
    """
    # Get evidence from vault
    evidence = evidence_vault.evidence_items.get(request.evidence_id)
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")
    
    video_path = Path(evidence.vault_path)
    
    result = await bwc_processor.process_bwc_video(
        video_path=video_path,
        evidence_id=request.evidence_id,
        perform_transcription=request.perform_transcription,
        perform_scene_detection=request.perform_scene_detection,
        perform_gps_extraction=request.perform_gps_extraction
    )
    
    return result.to_dict()


@router.get("/bwc/key-utterances/{evidence_id}")
async def extract_key_utterances(
    evidence_id: str,
    keywords: Optional[List[str]] = Query(None)
):
    """Extract specific commands/statements from BWC transcript"""
    utterances = await bwc_processor.extract_key_utterances(evidence_id, keywords)
    return {"evidence_id": evidence_id, "utterances": utterances}


# ============================================================================
# CAD PROCESSING ENDPOINTS
# ============================================================================

@router.post("/cad/process")
async def process_cad_export(request: ProcessCADRequest):
    """
    Process CAD export file
    
    Returns:
        Parsed CAD events, MDT queries, and timestamp analysis
    """
    evidence = evidence_vault.evidence_items.get(request.evidence_id)
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")
    
    file_path = Path(evidence.vault_path)
    
    result = await cad_processor.process_cad_export(
        file_path=file_path,
        evidence_id=request.evidence_id,
        case_id=request.case_id,
        expected_incident_id=request.expected_incident_id
    )
    
    return result.to_dict()


@router.post("/cad/negative-evidence")
async def record_negative_evidence(request: RecordNegativeEvidenceRequest):
    """
    Record 'no responsive records' response from agency
    
    CRITICAL for proving suppression/destruction
    """
    negative = await cad_processor.record_negative_evidence(
        case_id=request.case_id,
        evidence_type=NegativeEvidenceType(request.evidence_type),
        responding_agency=request.responding_agency,
        responding_custodian=request.responding_custodian,
        request_scope=request.request_scope,
        response_text=request.response_text,
        request_date=request.request_date,
        response_date=request.response_date
    )
    
    return negative.to_dict()


@router.post("/cad/cross-validate")
async def cross_validate_timestamps(
    evidence_id: str,
    bwc_evidence_id: Optional[str] = None
):
    """Cross-validate CAD timestamps with BWC and other sources"""
    # Load CAD events
    cad_metadata = cad_processor.metadata_dir / f"{evidence_id}_cad_metadata.json"
    if not cad_metadata.exists():
        raise HTTPException(status_code=404, detail="CAD metadata not found")
    
    with open(cad_metadata, 'r') as f:
        import json
        cad_data = json.load(f)
    
    # Load BWC metadata if provided
    bwc_data = None
    if bwc_evidence_id:
        bwc_metadata = bwc_processor.metadata_dir / f"{bwc_evidence_id}_bwc_metadata.json"
        if bwc_metadata.exists():
            with open(bwc_metadata, 'r') as f:
                bwc_data = json.load(f)
    
    from app.services.cad_processor_service import CADEvent, EventType
    events = [CADEvent(**e) for e in cad_data.get('events', [])]
    
    discrepancies = await cad_processor.cross_validate_timestamps(
        cad_events=events,
        bwc_metadata=bwc_data
    )
    
    return {"discrepancies": [d.__dict__ for d in discrepancies]}


# ============================================================================
# DOCUMENT PROCESSING ENDPOINTS
# ============================================================================

@router.post("/documents/process")
async def process_document(request: ProcessDocumentRequest):
    """
    Process document with OCR, metadata extraction, and optional Bates stamping
    """
    evidence = evidence_vault.evidence_items.get(request.evidence_id)
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")
    
    file_path = Path(evidence.vault_path)
    
    result = await document_processor.process_document(
        file_path=file_path,
        evidence_id=request.evidence_id,
        case_id=request.case_id,
        perform_ocr=request.perform_ocr,
        apply_bates=request.apply_bates,
        bates_prefix=request.bates_prefix
    )
    
    return result.to_dict()


@router.post("/documents/redact/{evidence_id}")
async def apply_redaction(
    evidence_id: str,
    page_number: int,
    coordinates: Dict,
    reason: str,
    applied_by: str
):
    """Apply redaction to document page"""
    redaction = await document_processor.apply_redaction(
        evidence_id=evidence_id,
        page_number=page_number,
        coordinates=coordinates,
        reason=reason,
        applied_by=applied_by
    )
    
    return redaction.__dict__


# ============================================================================
# UNIFIED SEARCH ENDPOINTS
# ============================================================================

@router.post("/search")
async def unified_search_evidence(request: UnifiedSearchRequest):
    """
    Cross-media search across all evidence types
    
    Returns:
        Search results from BWC transcripts, PDFs, CAD events, emails
    """
    query = SearchQuery(
        query_text=request.query_text,
        case_id=request.case_id,
        evidence_types=request.evidence_types,
        date_range=tuple(request.date_range) if request.date_range else None,
        custodians=request.custodians,
        tags=request.tags,
        issue_codes=request.issue_codes
    )
    
    results = await unified_search.search(query)
    
    return {
        "query": request.query_text,
        "total_results": len(results),
        "results": [r.__dict__ for r in results]
    }


@router.get("/search/entities/{case_id}")
async def extract_entities(case_id: str):
    """Extract named entities from all case evidence"""
    entities = await unified_search.extract_entities(case_id)
    return {"case_id": case_id, "entities": [e.__dict__ for e in entities]}


@router.get("/search/chronology/{case_id}")
async def build_chronology(case_id: str):
    """Build master chronology/timeline from all evidence"""
    chronology = await unified_search.build_master_chronology(case_id)
    return {
        "case_id": case_id,
        "total_entries": len(chronology),
        "chronology": [e.__dict__ for e in chronology]
    }


# ============================================================================
# PRODUCTION ENDPOINTS
# ============================================================================

@router.post("/production/clip")
async def create_video_clip(request: CreateVideoClipRequest):
    """Create authenticated video clip with overlay timestamps"""
    # Get source evidence
    evidence = evidence_vault.evidence_items.get(request.source_evidence_id)
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")
    
    source_file = Path(evidence.vault_path)
    
    clip = await production_service.create_video_clip(
        source_evidence_id=request.source_evidence_id,
        source_file=source_file,
        start_time=request.start_time,
        end_time=request.end_time,
        description=request.description,
        created_by=request.created_by
    )
    
    return clip.__dict__


@router.post("/production/exhibit-pack")
async def create_exhibit_pack(request: CreateExhibitPackRequest):
    """Bundle exhibits into numbered pack with index"""
    pack = await production_service.create_exhibit_pack(
        case_id=request.case_id,
        title=request.title,
        clips=[],  # Would load from clip_ids
        document_ids=request.document_ids
    )
    
    return pack.__dict__


@router.post("/production/load-file")
async def generate_load_file(request: GenerateLoadFileRequest):
    """Generate DAT/OPT load file for standard eDiscovery production"""
    load_file_path = await production_service.generate_dat_load_file(
        case_id=request.case_id,
        evidence_ids=request.evidence_ids,
        production_name=request.production_name
    )
    
    return FileResponse(
        path=str(load_file_path),
        media_type="text/plain",
        filename=f"{request.production_name}.dat"
    )


# ============================================================================
# MONITORING ENDPOINTS
# ============================================================================

@router.get("/monitoring/alerts")
async def get_alerts(acknowledged: Optional[bool] = None):
    """Get all alerts"""
    alerts = monitoring_service.alerts
    
    if acknowledged is not None:
        alerts = [a for a in alerts if a.acknowledged == acknowledged]
    
    return {"total_alerts": len(alerts), "alerts": [a.__dict__ for a in alerts]}


@router.get("/monitoring/coverage/{case_id}")
async def get_coverage_report(case_id: str):
    """Generate coverage/gaps analysis"""
    report = await monitoring_service.generate_coverage_report(case_id)
    return report.__dict__


@router.get("/monitoring/stats")
async def get_system_stats():
    """Get overall system statistics"""
    return {
        "total_evidence_items": len(evidence_vault.evidence_items),
        "litigation_holds_active": len([e for e in evidence_vault.evidence_items.values() if hasattr(e, 'litigation_hold') and e.litigation_hold]),
        "total_chain_events": len(evidence_vault.chain_events),
        "total_alerts": len(monitoring_service.alerts),
        "unacknowledged_alerts": len([a for a in monitoring_service.alerts if not a.acknowledged])
    }


# ============================================================================
# MEDIA ENHANCEMENT ENDPOINTS
# ============================================================================

class EnhanceAudioRequest(BaseModel):
    evidence_id: str
    quality_level: str = "Moderate"  # Minimal, Moderate, Aggressive
    noise_profile: str = "General Noise Reduction"
    apply_noise_reduction: bool = True
    apply_normalization: bool = True
    apply_voice_enhancement: bool = True
    isolate_speaker: Optional[str] = None  # "SPEAKER_01", etc.


class EnhanceVideoRequest(BaseModel):
    evidence_id: str
    quality_level: str = "Moderate"
    apply_upscaling: bool = True
    apply_stabilization: bool = True
    apply_sharpening: bool = True
    apply_denoising: bool = True
    target_resolution: str = "1920x1080"


class EnhanceImageRequest(BaseModel):
    evidence_id: str
    quality_level: str = "Moderate"
    apply_super_resolution: bool = True
    apply_clarity: bool = True
    apply_contrast: bool = True
    target_scale: float = 2.0


@router.post("/enhancement/audio")
async def enhance_audio(request: EnhanceAudioRequest, background_tasks: BackgroundTasks):
    """
    Enhance audio quality (noise reduction, normalization, voice enhancement)
    
    NON-DESTRUCTIVE: Original evidence is never modified
    Creates enhanced version with complete audit trail
    """
    # Retrieve original evidence
    if request.evidence_id not in evidence_vault.evidence_items:
        raise HTTPException(status_code=404, detail=f"Evidence not found: {request.evidence_id}")
    
    original_evidence = evidence_vault.evidence_items[request.evidence_id]
    source_file = Path(original_evidence.file_path)
    
    # Map quality level
    quality_map = {
        "Minimal": EnhancementQuality.MINIMAL,
        "Moderate": EnhancementQuality.MODERATE,
        "Aggressive": EnhancementQuality.AGGRESSIVE
    }
    quality = quality_map.get(request.quality_level, EnhancementQuality.MODERATE)
    
    # Map noise profile
    noise_map = {
        "Traffic/Road Noise": AudioNoiseProfile.TRAFFIC_NOISE,
        "Wind Noise": AudioNoiseProfile.WIND_NOISE,
        "Radio Static/Interference": AudioNoiseProfile.RADIO_STATIC,
        "Background Crowd/Multiple Speakers": AudioNoiseProfile.BACKGROUND_CROWD,
        "Vehicle Engine Hum": AudioNoiseProfile.ENGINE_HUM,
        "HVAC/Air Conditioning": AudioNoiseProfile.HVAC_NOISE,
        "General Noise Reduction": AudioNoiseProfile.GENERAL
    }
    noise_profile = noise_map.get(request.noise_profile, AudioNoiseProfile.GENERAL)
    
    # Process enhancement
    result = await media_enhancer.enhance_audio(
        source_file=source_file,
        evidence_id=request.evidence_id,
        quality_level=quality,
        noise_profile=noise_profile,
        apply_noise_reduction=request.apply_noise_reduction,
        apply_normalization=request.apply_normalization,
        apply_voice_enhancement=request.apply_voice_enhancement,
        isolate_speaker=request.isolate_speaker
    )
    
    # Log enhancement in chain of custody
    evidence_vault.log_chain_event(
        evidence_id=request.evidence_id,
        event_type="Enhancement",
        performed_by="System",
        user_role="Administrator",
        notes=f"Audio enhanced: {request.quality_level}, {request.noise_profile}"
    )
    
    return {
        "enhancement_id": result.enhancement_id,
        "original_evidence_id": result.source_evidence_id,
        "enhanced_file": result.enhanced_file,
        "original_hash": result.original_hash,
        "enhanced_hash": result.enhanced_hash,
        "processing_duration_seconds": result.processing_duration_seconds,
        "court_admissible": result.court_admissible,
        "original_metrics": result.original_metrics.to_dict(),
        "enhanced_metrics": result.enhanced_metrics.to_dict(),
        "enhancement_notes": result.enhancement_notes,
        "warning": "Original evidence preserved in vault. Enhanced version is derivative work." if result.court_admissible else "CAUTION: Aggressive enhancement may affect court admissibility."
    }


@router.post("/enhancement/video")
async def enhance_video(request: EnhanceVideoRequest, background_tasks: BackgroundTasks):
    """
    Enhance video quality (upscaling, stabilization, sharpening, denoising)
    
    NON-DESTRUCTIVE: Original evidence is never modified
    """
    if request.evidence_id not in evidence_vault.evidence_items:
        raise HTTPException(status_code=404, detail=f"Evidence not found: {request.evidence_id}")
    
    original_evidence = evidence_vault.evidence_items[request.evidence_id]
    source_file = Path(original_evidence.file_path)
    
    quality_map = {
        "Minimal": EnhancementQuality.MINIMAL,
        "Moderate": EnhancementQuality.MODERATE,
        "Aggressive": EnhancementQuality.AGGRESSIVE
    }
    quality = quality_map.get(request.quality_level, EnhancementQuality.MODERATE)
    
    result = await media_enhancer.enhance_video(
        source_file=source_file,
        evidence_id=request.evidence_id,
        quality_level=quality,
        apply_upscaling=request.apply_upscaling,
        apply_stabilization=request.apply_stabilization,
        apply_sharpening=request.apply_sharpening,
        apply_denoising=request.apply_denoising,
        target_resolution=request.target_resolution
    )
    
    evidence_vault.log_chain_event(
        evidence_id=request.evidence_id,
        event_type="Enhancement",
        performed_by="System",
        user_role="Administrator",
        notes=f"Video enhanced: {request.quality_level}, target: {request.target_resolution}"
    )
    
    return {
        "enhancement_id": result.enhancement_id,
        "original_evidence_id": result.source_evidence_id,
        "enhanced_file": result.enhanced_file,
        "original_hash": result.original_hash,
        "enhanced_hash": result.enhanced_hash,
        "processing_duration_seconds": result.processing_duration_seconds,
        "court_admissible": result.court_admissible,
        "original_metrics": result.original_metrics.to_dict(),
        "enhanced_metrics": result.enhanced_metrics.to_dict(),
        "enhancement_notes": result.enhancement_notes,
        "warning": "Original evidence preserved in vault. Enhanced version is derivative work." if result.court_admissible else "CAUTION: Aggressive enhancement may affect court admissibility."
    }


@router.post("/enhancement/image")
async def enhance_image(request: EnhanceImageRequest, background_tasks: BackgroundTasks):
    """
    Enhance image quality (super-resolution, clarity, contrast)
    
    NON-DESTRUCTIVE: Original evidence is never modified
    """
    if request.evidence_id not in evidence_vault.evidence_items:
        raise HTTPException(status_code=404, detail=f"Evidence not found: {request.evidence_id}")
    
    original_evidence = evidence_vault.evidence_items[request.evidence_id]
    source_file = Path(original_evidence.file_path)
    
    quality_map = {
        "Minimal": EnhancementQuality.MINIMAL,
        "Moderate": EnhancementQuality.MODERATE,
        "Aggressive": EnhancementQuality.AGGRESSIVE
    }
    quality = quality_map.get(request.quality_level, EnhancementQuality.MODERATE)
    
    result = await media_enhancer.enhance_image(
        source_file=source_file,
        evidence_id=request.evidence_id,
        quality_level=quality,
        apply_super_resolution=request.apply_super_resolution,
        apply_clarity=request.apply_clarity,
        apply_contrast=request.apply_contrast,
        target_scale=request.target_scale
    )
    
    evidence_vault.log_chain_event(
        evidence_id=request.evidence_id,
        event_type="Enhancement",
        performed_by="System",
        user_role="Administrator",
        notes=f"Image enhanced: {request.quality_level}, scale: {request.target_scale}x"
    )
    
    return {
        "enhancement_id": result.enhancement_id,
        "original_evidence_id": result.source_evidence_id,
        "enhanced_file": result.enhanced_file,
        "original_hash": result.original_hash,
        "enhanced_hash": result.enhanced_hash,
        "processing_duration_seconds": result.processing_duration_seconds,
        "court_admissible": result.court_admissible,
        "original_metrics": result.original_metrics.to_dict(),
        "enhanced_metrics": result.enhanced_metrics.to_dict(),
        "enhancement_notes": result.enhancement_notes,
        "warning": "Original evidence preserved in vault. Enhanced version is derivative work." if result.court_admissible else "CAUTION: Aggressive enhancement may affect court admissibility."
    }


@router.get("/enhancement/compare/{enhancement_id}")
async def compare_enhancement(enhancement_id: str, original_evidence_id: str):
    """
    Compare original vs enhanced versions with quality metrics
    
    Returns side-by-side comparison and recommendations for court use
    """
    try:
        comparison = await media_enhancer.compare_versions(original_evidence_id, enhancement_id)
        return comparison
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/enhancement/download/{enhancement_id}")
async def download_enhanced_media(enhancement_id: str):
    """Download enhanced media file"""
    # Look for enhancement file in output directories
    for directory in [media_enhancer.audio_enhanced_dir, 
                      media_enhancer.video_enhanced_dir, 
                      media_enhancer.image_enhanced_dir]:
        for ext in ['.wav', '.mp4', '.png', '.jpg']:
            file_path = directory / f"{enhancement_id}{ext}"
            if file_path.exists():
                return FileResponse(
                    path=str(file_path),
                    filename=f"{enhancement_id}{ext}",
                    media_type="application/octet-stream"
                )
    
    raise HTTPException(status_code=404, detail=f"Enhanced file not found: {enhancement_id}")

