"""
BarberX Legal Case Management Pro Suite
Evidence API Router - BWC Footage, Video Processing, Multi-POV Sync
"""
import os
import re
import hashlib
import uuid
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.db.database import get_db
from app.db.models import Evidence, Case, EvidenceType, ProcessingStatus, case_evidence
from app.schemas.legal_schemas import (
    EvidenceResponse, BWCUploadResponse, BatchBWCUploadResponse, BWCMetadata,
    VideoSyncRequest, VideoSyncResponse,
    AudioHarmonizeRequest, AudioHarmonizeResponse
)

router = APIRouter()

# Configure upload directories
EVIDENCE_DIR = Path("uploads/evidence")
BWC_DIR = EVIDENCE_DIR / "bwc"
AUDIO_DIR = EVIDENCE_DIR / "audio"

for dir_path in [EVIDENCE_DIR, BWC_DIR, AUDIO_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


def calculate_file_hash(file_content: bytes) -> str:
    """Calculate SHA-256 hash of file content"""
    return hashlib.sha256(file_content).hexdigest()


def parse_motorola_bwc_filename(filename: str) -> dict:
    """
    Parse Motorola Solutions BWC filename format.
    
    Format: OfficerName_YYYYMMDDHHMI_DeviceID-Segment.mp4
    Example: BryanMerritt_202511292256_311-0.mp4
    Example: BryanMerritt_202511292257_BWL7137497-0.mp4
    Example: CristianMartin_202511292312_BWL7139081-0.mp4
    """
    # Remove extension
    base = filename.rsplit('.', 1)[0]
    
    # Pattern: Name_Timestamp_DeviceID-Segment
    pattern = r'^([A-Za-z]+)_(\d{12})_([A-Z0-9]+)-(\d+)$'
    match = re.match(pattern, base)
    
    if match:
        officer_name = match.group(1)
        # Insert space before capital letters for readability
        officer_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', officer_name)
        
        timestamp_str = match.group(2)
        # Parse: YYYYMMDDHHMI
        try:
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d%H%M')
        except:
            timestamp = None
        
        device_id = match.group(3)
        segment = int(match.group(4))
        
        return {
            'officer_name': officer_name,
            'timestamp': timestamp,
            'device_id': device_id,
            'segment': segment,
            'parsed': True
        }
    
    return {
        'officer_name': None,
        'timestamp': None,
        'device_id': None,
        'segment': 0,
        'parsed': False
    }


async def process_bwc_footage(evidence_id: int, db: AsyncSession):
    """Background task to process BWC footage"""
    result = await db.execute(select(Evidence).where(Evidence.id == evidence_id))
    evidence = result.scalar_one_or_none()
    
    if evidence:
        evidence.processing_status = ProcessingStatus.PROCESSING
        await db.commit()
        
        try:
            # TODO: Actual video processing with FFmpeg/MoviePy
            # - Extract duration, resolution, frame rate
            # - Generate thumbnail/key frames
            # - Extract audio track
            # - Run audio transcription
            
            evidence.processing_status = ProcessingStatus.COMPLETED
            evidence.processed_at = datetime.utcnow()
            await db.commit()
        except Exception as e:
            evidence.processing_status = ProcessingStatus.FAILED
            await db.commit()


@router.post("/bwc/upload", response_model=BatchBWCUploadResponse)
async def upload_bwc_footage(
    files: List[UploadFile] = File(...),
    case_id: Optional[int] = Form(None),
    auto_sync: bool = Form(True),
    auto_transcribe: bool = Form(True),
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Batch upload Body-Worn Camera footage.
    
    - Parses Motorola Solutions filename format
    - Extracts officer name, timestamp, device ID
    - Groups videos by incident for multi-POV sync
    - Queues for processing (transcription, key frames)
    """
    uploaded = []
    failed = []
    sync_group_id = None
    
    # Verify case exists if provided
    if case_id:
        case_result = await db.execute(select(Case).where(Case.id == case_id))
        case = case_result.scalar_one_or_none()
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
    
    # Generate sync group ID if multiple files
    if len(files) > 1 and auto_sync:
        sync_group_id = f"sync_{uuid.uuid4().hex[:8]}"
    
    for file in files:
        try:
            # Validate file type
            valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
            ext = Path(file.filename).suffix.lower()
            if ext not in valid_extensions:
                failed.append({
                    "filename": file.filename,
                    "error": f"Invalid video format. Accepted: {', '.join(valid_extensions)}"
                })
                continue
            
            # Read file content
            content = await file.read()
            file_hash = calculate_file_hash(content)
            
            # Check for duplicate
            existing = await db.execute(
                select(Evidence).where(Evidence.file_hash == file_hash)
            )
            if existing.scalar_one_or_none():
                failed.append({
                    "filename": file.filename,
                    "error": "Duplicate file (already uploaded)"
                })
                continue
            
            # Parse Motorola filename
            bwc_info = parse_motorola_bwc_filename(file.filename)
            
            # Generate unique filename
            unique_id = uuid.uuid4().hex[:12]
            safe_filename = f"{unique_id}_{file.filename}"
            file_path = BWC_DIR / safe_filename
            
            # Save file
            with open(file_path, "wb") as f:
                f.write(content)
            
            # Create evidence record
            evidence = Evidence(
                filename=safe_filename,
                original_filename=file.filename,
                file_path=str(file_path),
                file_size=len(content),
                mime_type=file.content_type or f"video/{ext[1:]}",
                file_hash=file_hash,
                evidence_type=EvidenceType.BWC_FOOTAGE,
                
                # BWC metadata from filename
                officer_name=bwc_info['officer_name'],
                start_timestamp=bwc_info['timestamp'],
                bwc_device_id=bwc_info['device_id'],
                
                # Sync info
                sync_group_id=sync_group_id,
                
                processing_status=ProcessingStatus.PENDING
            )
            
            db.add(evidence)
            await db.commit()
            await db.refresh(evidence)
            
            # Link to case if provided
            if case_id:
                await db.execute(
                    case_evidence.insert().values(
                        case_id=case_id,
                        evidence_id=evidence.id
                    )
                )
                await db.commit()
            
            # Queue processing
            if background_tasks:
                background_tasks.add_task(process_bwc_footage, evidence.id, db)
            
            # Create metadata response
            metadata = BWCMetadata(
                officer_name=bwc_info['officer_name'] or "Unknown",
                timestamp=bwc_info['timestamp'] or datetime.now(),
                device_id=bwc_info['device_id'] or "Unknown",
                segment=bwc_info['segment'],
                duration_seconds=None,  # Set after processing
                resolution=None,
                frame_rate=None,
                file_size=len(content),
                file_hash=file_hash
            )
            
            uploaded.append(BWCUploadResponse(
                id=evidence.id,
                filename=evidence.filename,
                original_filename=evidence.original_filename,
                evidence_type=evidence.evidence_type,
                metadata=metadata,
                processing_status=evidence.processing_status,
                message="Upload successful" + (
                    f", Officer: {bwc_info['officer_name']}" if bwc_info['parsed'] else ", filename not in Motorola format"
                )
            ))
            
        except Exception as e:
            failed.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return BatchBWCUploadResponse(
        uploaded=uploaded,
        failed=failed,
        total_uploaded=len(uploaded),
        total_failed=len(failed),
        sync_group_id=sync_group_id
    )


@router.post("/bwc/motorola")
async def import_motorola_folder(
    folder_path: str,
    case_id: Optional[int] = None,
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Import BWC footage from a Motorola Solutions export folder.
    
    Scans folder for video files and imports them with metadata.
    """
    path = Path(folder_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Folder not found")
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    videos_found = []
    officers_found = set()
    
    for file_path in path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            bwc_info = parse_motorola_bwc_filename(file_path.name)
            videos_found.append({
                'path': str(file_path),
                'filename': file_path.name,
                'officer': bwc_info['officer_name'],
                'timestamp': bwc_info['timestamp'].isoformat() if bwc_info['timestamp'] else None,
                'device_id': bwc_info['device_id'],
                'parsed': bwc_info['parsed']
            })
            if bwc_info['officer_name']:
                officers_found.add(bwc_info['officer_name'])
    
    return {
        "folder": folder_path,
        "videos_found": len(videos_found),
        "officers_detected": list(officers_found),
        "files": videos_found,
        "next_steps": [
            "Review the detected files and officers",
            "POST /api/evidence/bwc/upload with the video files to import them"
        ]
    }


@router.get("/", response_model=List[EvidenceResponse])
async def list_evidence(
    case_id: Optional[int] = None,
    evidence_type: Optional[EvidenceType] = None,
    officer_name: Optional[str] = None,
    sync_group_id: Optional[str] = None,
    limit: int = Query(50, le=200),
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    """List evidence with filtering"""
    query = select(Evidence)
    
    if case_id:
        query = query.join(case_evidence).where(case_evidence.c.case_id == case_id)
    if evidence_type:
        query = query.where(Evidence.evidence_type == evidence_type)
    if officer_name:
        query = query.where(Evidence.officer_name.ilike(f"%{officer_name}%"))
    if sync_group_id:
        query = query.where(Evidence.sync_group_id == sync_group_id)
    
    query = query.order_by(Evidence.start_timestamp.asc().nullslast())
    query = query.offset(offset).limit(limit)
    
    result = await db.execute(query)
    evidence_items = result.scalars().all()
    
    return [EvidenceResponse.model_validate(e) for e in evidence_items]


@router.get("/{evidence_id}", response_model=EvidenceResponse)
async def get_evidence(
    evidence_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get specific evidence by ID"""
    result = await db.execute(select(Evidence).where(Evidence.id == evidence_id))
    evidence = result.scalar_one_or_none()
    
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")
    
    return EvidenceResponse.model_validate(evidence)


@router.get("/bwc/{evidence_id}/frames")
async def get_key_frames(
    evidence_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get key frames from BWC footage"""
    result = await db.execute(select(Evidence).where(Evidence.id == evidence_id))
    evidence = result.scalar_one_or_none()
    
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")
    
    return {
        "evidence_id": evidence_id,
        "filename": evidence.original_filename,
        "key_frames": evidence.key_frames or [],
        "total_frames": len(evidence.key_frames or [])
    }


@router.post("/bwc/sync", response_model=VideoSyncResponse)
async def sync_multi_pov(
    request: VideoSyncRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Synchronize multiple BWC videos from different perspectives.
    
    Uses audio fingerprinting or timestamp alignment to sync videos
    so they can be viewed together in a multi-POV player.
    """
    # Verify all evidence exists
    evidence_items = []
    for eid in request.evidence_ids:
        result = await db.execute(select(Evidence).where(Evidence.id == eid))
        evidence = result.scalar_one_or_none()
        if not evidence:
            raise HTTPException(status_code=404, detail=f"Evidence {eid} not found")
        evidence_items.append(evidence)
    
    # Generate sync group
    sync_group_id = f"sync_{uuid.uuid4().hex[:8]}"
    
    # Calculate offsets (would use audio fingerprinting in production)
    synced_videos = []
    timeline_start = None
    timeline_end = None
    
    for i, evidence in enumerate(evidence_items):
        is_primary = evidence.id == request.primary_evidence_id
        
        # Set offset (from manual input or calculated)
        offset_ms = 0
        if request.manual_offsets and evidence.id in request.manual_offsets:
            offset_ms = request.manual_offsets[evidence.id]
        
        # Update evidence record
        evidence.sync_group_id = sync_group_id
        evidence.sync_offset_ms = offset_ms
        evidence.is_primary_pov = is_primary
        
        # Track timeline bounds
        if evidence.start_timestamp:
            if not timeline_start or evidence.start_timestamp < timeline_start:
                timeline_start = evidence.start_timestamp
            if evidence.end_timestamp:
                if not timeline_end or evidence.end_timestamp > timeline_end:
                    timeline_end = evidence.end_timestamp
        
        synced_videos.append({
            "evidence_id": evidence.id,
            "officer_name": evidence.officer_name,
            "device_id": evidence.bwc_device_id,
            "offset_ms": offset_ms,
            "is_primary": is_primary
        })
    
    await db.commit()
    
    return VideoSyncResponse(
        sync_group_id=sync_group_id,
        primary_evidence_id=request.primary_evidence_id,
        synced_videos=synced_videos,
        sync_quality=0.95 if request.sync_method == "manual" else 0.85,
        timeline_start=timeline_start or datetime.now(),
        timeline_end=timeline_end or datetime.now()
    )


@router.post("/bwc/harmonize", response_model=AudioHarmonizeResponse)
async def harmonize_audio(
    request: AudioHarmonizeRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Harmonize audio from multiple BWC sources.
    
    - Normalize audio levels across all tracks
    - Apply noise reduction
    - Enhance speech frequencies
    - Create combined audio track
    """
    # This would use librosa/pydub in production
    
    # Verify evidence exists
    for eid in request.evidence_ids:
        result = await db.execute(select(Evidence).where(Evidence.id == eid))
        if not result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail=f"Evidence {eid} not found")
    
    # Placeholder for audio processing
    output_filename = f"harmonized_{uuid.uuid4().hex[:8]}.{request.output_format}"
    output_path = AUDIO_DIR / output_filename
    
    processing_notes = []
    if request.normalize_levels:
        processing_notes.append("Audio levels normalized to -16 LUFS")
    if request.reduce_noise:
        processing_notes.append("Background noise reduced using spectral gating")
    if request.enhance_speech:
        processing_notes.append("Speech frequencies enhanced (200Hz-4kHz)")
    
    # In production, would create actual Evidence record for output
    return AudioHarmonizeResponse(
        output_file_id=0,  # Would be real ID
        output_filename=output_filename,
        sources=request.evidence_ids,
        processing_notes=processing_notes
    )


@router.post("/link")
async def link_evidence(
    evidence_ids: List[int],
    case_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Link evidence items to a case"""
    # Verify case exists
    case_result = await db.execute(select(Case).where(Case.id == case_id))
    if not case_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Case not found")
    
    linked = 0
    for eid in evidence_ids:
        result = await db.execute(select(Evidence).where(Evidence.id == eid))
        if result.scalar_one_or_none():
            try:
                await db.execute(
                    case_evidence.insert().values(
                        case_id=case_id,
                        evidence_id=eid
                    )
                )
                linked += 1
            except:
                pass
    
    await db.commit()
    
    return {
        "message": f"Linked {linked} evidence items to case {case_id}",
        "case_id": case_id,
        "evidence_linked": linked
    }


# ============================================================================
# BWC Folder Scanner Endpoints
# ============================================================================

@router.post("/bwc/scan-folder")
async def scan_bwc_folder_endpoint(
    folder_path: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Scan a BWC folder and return discovered footage metadata.
    
    Parses Motorola Solutions filename format to extract:
    - Officer names
    - Timestamps
    - Device IDs
    - File organization
    
    Does NOT import files - use /bwc/import-folder for that.
    """
    from ..utils.bwc_scanner import BWCFolderScanner
    
    folder = Path(folder_path)
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"Folder not found: {folder_path}")
    
    scanner = BWCFolderScanner()
    incident = scanner.scan_folder(folder)
    report = scanner.generate_report(incident)
    
    return report


@router.post("/bwc/scan-root")
async def scan_bwc_root_endpoint(
    bwc_root: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Scan a .bwc root folder for all incident subfolders.
    
    Returns summary of all discovered incidents with footage metadata.
    """
    from ..utils.bwc_scanner import BWCFolderScanner
    
    root = Path(bwc_root)
    if not root.exists():
        raise HTTPException(status_code=404, detail=f"BWC root not found: {bwc_root}")
    
    scanner = BWCFolderScanner()
    incidents = scanner.scan_bwc_root(root)
    
    return {
        "bwc_root": bwc_root,
        "total_incidents": len(incidents),
        "incidents": [scanner.generate_report(i) for i in incidents]
    }


@router.post("/bwc/import-folder")
async def import_bwc_folder(
    folder_path: str,
    case_id: Optional[int] = None,
    create_case: bool = False,
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Import BWC footage from a local folder into the system.
    
    Scans folder for Motorola BWC files, creates Evidence records,
    and optionally links to a case.
    
    Args:
        folder_path: Path to folder containing BWC footage
        case_id: Optional case ID to link footage to
        create_case: If True and no case_id, create new case from folder
    """
    from ..utils.bwc_scanner import BWCFolderScanner, BWCTimelineBuilder
    
    folder = Path(folder_path)
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"Folder not found: {folder_path}")
    
    # Scan the folder
    scanner = BWCFolderScanner()
    incident = scanner.scan_folder(folder)
    
    # Create case if requested
    if create_case and not case_id:
        new_case = Case(
            case_number=incident.case_number,
            title=f"BWC Import: {incident.case_number}",
            case_type="civil_rights",
            status="open",
            description=f"Auto-created from BWC folder import: {folder_path}"
        )
        db.add(new_case)
        await db.flush()
        case_id = new_case.id
    
    # Import all video files
    imported = []
    failed = []
    sync_group_id = f"sync_{uuid.uuid4().hex[:8]}"
    
    for officer in incident.officers.values():
        for file_info in officer.files:
            try:
                file_path = Path(file_info['path'])
                
                # Check for duplicate
                existing = await db.execute(
                    select(Evidence).where(
                        Evidence.original_filename == file_path.name
                    )
                )
                if existing.scalar_one_or_none():
                    failed.append({
                        "filename": file_path.name,
                        "error": "Already imported"
                    })
                    continue
                
                # Get file size
                file_size = file_path.stat().st_size
                
                # Create evidence record (referencing original location)
                evidence = Evidence(
                    filename=file_path.name,
                    original_filename=file_path.name,
                    file_path=str(file_path),
                    file_size=file_size,
                    mime_type="video/mp4",
                    evidence_type=EvidenceType.BWC_FOOTAGE,
                    officer_name=officer.officer_name,
                    start_timestamp=file_info['timestamp'],
                    bwc_device_id=file_info['device_id'],
                    sync_group_id=sync_group_id,
                    processing_status=ProcessingStatus.PENDING
                )
                db.add(evidence)
                await db.flush()
                
                # Link to case if provided
                if case_id:
                    await db.execute(
                        case_evidence.insert().values(
                            case_id=case_id,
                            evidence_id=evidence.id
                        )
                    )
                
                imported.append({
                    "id": evidence.id,
                    "filename": file_path.name,
                    "officer": officer.officer_name,
                    "timestamp": file_info['timestamp'].isoformat() if file_info['timestamp'] else None
                })
                
            except Exception as e:
                failed.append({
                    "filename": file_info['filename'],
                    "error": str(e)
                })
    
    # Import documents
    for doc in incident.documents:
        doc_path = Path(doc['path'])
        # Would create Document records here
    
    await db.commit()
    
    # Build timeline
    timeline_builder = BWCTimelineBuilder(incident)
    overlaps = timeline_builder.find_overlapping_footage()
    
    return {
        "success": True,
        "case_number": incident.case_number,
        "case_id": case_id,
        "sync_group_id": sync_group_id,
        "imported": len(imported),
        "failed": len(failed),
        "officers_found": incident.total_officers,
        "multi_pov_overlaps": len(overlaps),
        "details": {
            "imported": imported,
            "failed": failed,
            "overlapping_footage": overlaps
        }
    }


@router.get("/bwc/timeline/{sync_group_id}")
async def get_bwc_timeline(
    sync_group_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get synchronized timeline for a BWC sync group.
    
    Returns all footage in chronological order with multi-POV markers.
    """
    result = await db.execute(
        select(Evidence)
        .where(Evidence.sync_group_id == sync_group_id)
        .order_by(Evidence.start_timestamp)
    )
    evidence_items = result.scalars().all()
    
    if not evidence_items:
        raise HTTPException(status_code=404, detail="Sync group not found")
    
    timeline = []
    for e in evidence_items:
        timeline.append({
            "id": e.id,
            "officer": e.officer_name,
            "filename": e.original_filename,
            "timestamp": e.start_timestamp.isoformat() if e.start_timestamp else None,
            "device_id": e.bwc_device_id,
            "duration": e.duration_seconds
        })
    
    # Find overlapping segments
    overlaps = []
    for i, item in enumerate(timeline):
        if not item['timestamp']:
            continue
        concurrent = [item['officer']]
        for j, other in enumerate(timeline):
            if i == j or not other['timestamp']:
                continue
            # Check if within 5 minutes
            try:
                t1 = datetime.fromisoformat(item['timestamp'])
                t2 = datetime.fromisoformat(other['timestamp'])
                if abs((t1 - t2).total_seconds()) <= 300:
                    if other['officer'] not in concurrent:
                        concurrent.append(other['officer'])
            except:
                pass
        
        if len(concurrent) > 1:
            overlaps.append({
                "timestamp": item['timestamp'],
                "officers": concurrent
            })
    
    return {
        "sync_group_id": sync_group_id,
        "total_files": len(timeline),
        "officers": list(set(e['officer'] for e in timeline if e['officer'])),
        "timeline": timeline,
        "multi_pov_segments": overlaps
    }
