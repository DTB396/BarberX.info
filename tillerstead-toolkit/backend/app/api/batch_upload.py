"""
BarberX Legal Case Management Pro Suite
Batch Upload API - Enhanced Bulk Upload with Auto-Transcription
"""
import os
import uuid
import asyncio
import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
from dataclasses import asdict

from fastapi import (
    APIRouter, Depends, HTTPException, UploadFile, File, Form, 
    Query, BackgroundTasks, WebSocket, WebSocketDisconnect
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.db.database import get_db
from app.db.models import Evidence, Case, EvidenceType, ProcessingStatus, case_evidence
from app.services.transcription_service import (
    transcription_service, 
    TranscriptionResult,
    BatchTranscriptionJob
)

router = APIRouter()

# Upload directories
UPLOAD_BASE = Path("uploads")
BWC_DIR = UPLOAD_BASE / "bwc"
AUDIO_DIR = UPLOAD_BASE / "audio"
TEMP_DIR = UPLOAD_BASE / "temp"

for d in [UPLOAD_BASE, BWC_DIR, AUDIO_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# SCHEMAS
# ============================================================

class UploadConfig(BaseModel):
    """Configuration for batch upload"""
    case_id: Optional[int] = None
    auto_transcribe: bool = True
    enhance_audio: bool = True
    detect_speakers: bool = True
    word_timestamps: bool = True
    language: str = "en"
    auto_sync: bool = True
    priority: str = "normal"  # low, normal, high


class FileUploadResult(BaseModel):
    """Result of a single file upload"""
    evidence_id: int
    filename: str
    original_filename: str
    file_size: int
    file_hash: str
    officer_name: Optional[str] = None
    timestamp: Optional[str] = None
    device_id: Optional[str] = None
    status: str
    message: str


class BatchUploadResponse(BaseModel):
    """Response from batch upload"""
    batch_id: str
    total_files: int
    uploaded: int
    failed: int
    results: List[FileUploadResult]
    errors: List[Dict[str, Any]]
    transcription_job_id: Optional[str] = None
    sync_group_id: Optional[str] = None


class TranscriptionJobResponse(BaseModel):
    """Transcription job status"""
    job_id: str
    status: str
    total_files: int
    completed_files: int
    failed_files: int
    results: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    created_at: str
    completed_at: Optional[str] = None


class TranscriptionRequest(BaseModel):
    """Request to transcribe evidence"""
    evidence_ids: List[int]
    enhance_audio: bool = True
    detect_speakers: bool = True
    word_timestamps: bool = True
    language: str = "en"


class AudioEnhanceRequest(BaseModel):
    """Request to enhance audio"""
    evidence_id: int
    noise_reduction: bool = True
    normalize_volume: bool = True
    enhance_speech: bool = True


# ============================================================
# UPLOAD TRACKING
# ============================================================

class UploadTracker:
    """Track upload and processing progress"""
    
    def __init__(self):
        self.active_uploads: Dict[str, Dict[str, Any]] = {}
        self.websocket_connections: Dict[str, List[WebSocket]] = {}
    
    def create_batch(self, batch_id: str, total_files: int) -> Dict[str, Any]:
        """Create new batch tracking entry"""
        batch = {
            "batch_id": batch_id,
            "total_files": total_files,
            "uploaded": 0,
            "failed": 0,
            "processing": 0,
            "transcribed": 0,
            "status": "uploading",
            "results": [],
            "errors": [],
            "created_at": datetime.utcnow().isoformat()
        }
        self.active_uploads[batch_id] = batch
        return batch
    
    async def update_progress(self, batch_id: str, update: Dict[str, Any]):
        """Update batch progress and notify WebSocket clients"""
        if batch_id in self.active_uploads:
            self.active_uploads[batch_id].update(update)
            await self._broadcast_update(batch_id)
    
    async def _broadcast_update(self, batch_id: str):
        """Broadcast progress to WebSocket clients"""
        if batch_id in self.websocket_connections:
            batch = self.active_uploads.get(batch_id, {})
            for ws in self.websocket_connections[batch_id]:
                try:
                    await ws.send_json({
                        "type": "progress",
                        "data": batch
                    })
                except:
                    pass
    
    async def connect_websocket(self, batch_id: str, websocket: WebSocket):
        """Connect WebSocket for progress updates"""
        if batch_id not in self.websocket_connections:
            self.websocket_connections[batch_id] = []
        self.websocket_connections[batch_id].append(websocket)
    
    def disconnect_websocket(self, batch_id: str, websocket: WebSocket):
        """Disconnect WebSocket"""
        if batch_id in self.websocket_connections:
            try:
                self.websocket_connections[batch_id].remove(websocket)
            except:
                pass
    
    def get_batch(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get batch status"""
        return self.active_uploads.get(batch_id)


upload_tracker = UploadTracker()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_file_hash(content: bytes) -> str:
    """Calculate SHA-256 hash"""
    return hashlib.sha256(content).hexdigest()


def parse_motorola_filename(filename: str) -> Dict[str, Any]:
    """Parse Motorola BWC filename format"""
    import re
    
    base = filename.rsplit('.', 1)[0]
    pattern = r'^([A-Za-z]+)_(\d{12})_([A-Z0-9]+)-(\d+)$'
    match = re.match(pattern, base)
    
    if match:
        officer_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', match.group(1))
        try:
            timestamp = datetime.strptime(match.group(2), '%Y%m%d%H%M')
        except:
            timestamp = None
        
        return {
            'officer_name': officer_name,
            'timestamp': timestamp,
            'device_id': match.group(3),
            'segment': int(match.group(4)),
            'parsed': True
        }
    
    return {'officer_name': None, 'timestamp': None, 'device_id': None, 'segment': 0, 'parsed': False}


async def process_transcription_background(
    evidence_id: int,
    file_path: str,
    config: Dict[str, Any],
    db_url: str
):
    """Background task to transcribe a file"""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    
    engine = create_async_engine(db_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as db:
        try:
            # Update status to processing
            await db.execute(
                update(Evidence)
                .where(Evidence.id == evidence_id)
                .values(processing_status=ProcessingStatus.PROCESSING)
            )
            await db.commit()
            
            # Transcribe
            result = await transcription_service.transcribe_file(
                audio_path=file_path,
                evidence_id=evidence_id,
                language=config.get('language', 'en'),
                enhance_audio=config.get('enhance_audio', True),
                detect_speakers=config.get('detect_speakers', True),
                word_timestamps=config.get('word_timestamps', True)
            )
            
            # Update evidence record
            await db.execute(
                update(Evidence)
                .where(Evidence.id == evidence_id)
                .values(
                    transcript=result.full_text,
                    transcript_confidence=result.speech_ratio,
                    transcript_language=result.language,
                    transcript_uploaded_at=datetime.utcnow(),
                    processing_status=ProcessingStatus.COMPLETED,
                    processed_at=datetime.utcnow(),
                    detected_events={
                        "miranda_detected": result.miranda_detected,
                        "miranda_timestamp": result.miranda_timestamp,
                        "commands": result.commands_detected,
                        "threats": result.threats_detected
                    }
                )
            )
            await db.commit()
            
        except Exception as e:
            await db.execute(
                update(Evidence)
                .where(Evidence.id == evidence_id)
                .values(
                    processing_status=ProcessingStatus.FAILED,
                    processing_notes=str(e)
                )
            )
            await db.commit()


# ============================================================
# ENDPOINTS
# ============================================================

@router.post("/upload/batch", response_model=BatchUploadResponse)
async def batch_upload(
    files: List[UploadFile] = File(...),
    case_id: Optional[int] = Form(None),
    auto_transcribe: bool = Form(True),
    enhance_audio: bool = Form(True),
    detect_speakers: bool = Form(True),
    word_timestamps: bool = Form(True),
    language: str = Form("en"),
    auto_sync: bool = Form(True),
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Batch upload BWC footage with automatic transcription.
    
    Features:
    - Supports multiple file upload
    - Parses Motorola BWC filename format
    - Auto-transcribes audio with OpenAI Whisper
    - Detects Miranda rights, commands, threats
    - Creates sync groups for multi-POV playback
    - Real-time progress via WebSocket
    
    Supported formats: MP4, AVI, MOV, MKV, WEBM, MP3, WAV, M4A
    """
    batch_id = f"batch_{uuid.uuid4().hex[:8]}"
    
    # Validate case
    if case_id:
        case_result = await db.execute(select(Case).where(Case.id == case_id))
        if not case_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Case not found")
    
    # Create batch tracker
    upload_tracker.create_batch(batch_id, len(files))
    
    results = []
    errors = []
    evidence_ids = []
    sync_group_id = f"sync_{batch_id}" if auto_sync and len(files) > 1 else None
    
    valid_video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    valid_audio_exts = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac'}
    valid_exts = valid_video_exts | valid_audio_exts
    
    for file in files:
        try:
            ext = Path(file.filename).suffix.lower()
            
            # Validate format
            if ext not in valid_exts:
                errors.append({
                    "filename": file.filename,
                    "error": f"Unsupported format. Use: {', '.join(valid_exts)}"
                })
                await upload_tracker.update_progress(batch_id, {"failed": len(errors)})
                continue
            
            # Read file
            content = await file.read()
            file_hash = calculate_file_hash(content)
            
            # Check duplicate
            existing = await db.execute(
                select(Evidence).where(Evidence.file_hash == file_hash)
            )
            if existing.scalar_one_or_none():
                errors.append({
                    "filename": file.filename,
                    "error": "Duplicate file already uploaded"
                })
                await upload_tracker.update_progress(batch_id, {"failed": len(errors)})
                continue
            
            # Parse Motorola filename
            bwc_info = parse_motorola_filename(file.filename)
            
            # Determine evidence type
            is_video = ext in valid_video_exts
            evidence_type = EvidenceType.BWC_FOOTAGE if is_video else EvidenceType.AUDIO_RECORDING
            
            # Generate safe filename
            unique_id = uuid.uuid4().hex[:12]
            safe_filename = f"{unique_id}_{file.filename}"
            
            # Choose directory
            save_dir = BWC_DIR if is_video else AUDIO_DIR
            file_path = save_dir / safe_filename
            
            # Save file
            with open(file_path, "wb") as f:
                f.write(content)
            
            # Create evidence record
            evidence = Evidence(
                filename=safe_filename,
                original_filename=file.filename,
                file_path=str(file_path),
                file_size=len(content),
                mime_type=file.content_type or f"{'video' if is_video else 'audio'}/{ext[1:]}",
                file_hash=file_hash,
                evidence_type=evidence_type,
                
                # BWC metadata
                officer_name=bwc_info.get('officer_name'),
                start_timestamp=bwc_info.get('timestamp'),
                bwc_device_id=bwc_info.get('device_id'),
                
                # Sync
                sync_group_id=sync_group_id,
                
                # Processing
                processing_status=ProcessingStatus.PENDING if auto_transcribe else ProcessingStatus.COMPLETED
            )
            
            db.add(evidence)
            await db.commit()
            await db.refresh(evidence)
            
            evidence_ids.append(evidence.id)
            
            # Link to case
            if case_id:
                await db.execute(
                    case_evidence.insert().values(
                        case_id=case_id,
                        evidence_id=evidence.id
                    )
                )
                await db.commit()
            
            # Queue transcription
            if auto_transcribe and background_tasks:
                from app.core.config import settings
                background_tasks.add_task(
                    process_transcription_background,
                    evidence.id,
                    str(file_path),
                    {
                        'language': language,
                        'enhance_audio': enhance_audio,
                        'detect_speakers': detect_speakers,
                        'word_timestamps': word_timestamps
                    },
                    settings.DATABASE_URL
                )
            
            results.append(FileUploadResult(
                evidence_id=evidence.id,
                filename=safe_filename,
                original_filename=file.filename,
                file_size=len(content),
                file_hash=file_hash,
                officer_name=bwc_info.get('officer_name'),
                timestamp=bwc_info['timestamp'].isoformat() if bwc_info.get('timestamp') else None,
                device_id=bwc_info.get('device_id'),
                status="uploaded",
                message="Queued for transcription" if auto_transcribe else "Upload complete"
            ))
            
            await upload_tracker.update_progress(batch_id, {
                "uploaded": len(results),
                "results": [r.dict() for r in results]
            })
            
        except Exception as e:
            errors.append({
                "filename": file.filename,
                "error": str(e)
            })
            await upload_tracker.update_progress(batch_id, {"failed": len(errors)})
    
    # Update final status
    await upload_tracker.update_progress(batch_id, {
        "status": "completed",
        "completed_at": datetime.utcnow().isoformat()
    })
    
    return BatchUploadResponse(
        batch_id=batch_id,
        total_files=len(files),
        uploaded=len(results),
        failed=len(errors),
        results=results,
        errors=errors,
        transcription_job_id=batch_id if auto_transcribe else None,
        sync_group_id=sync_group_id
    )


@router.get("/upload/{batch_id}/status")
async def get_upload_status(batch_id: str):
    """Get batch upload status"""
    batch = upload_tracker.get_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    return batch


@router.websocket("/upload/{batch_id}/ws")
async def upload_progress_websocket(websocket: WebSocket, batch_id: str):
    """WebSocket for real-time upload progress"""
    await websocket.accept()
    await upload_tracker.connect_websocket(batch_id, websocket)
    
    # Send current status
    batch = upload_tracker.get_batch(batch_id)
    if batch:
        await websocket.send_json({"type": "status", "data": batch})
    
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        upload_tracker.disconnect_websocket(batch_id, websocket)


# ============================================================
# TRANSCRIPTION ENDPOINTS
# ============================================================

@router.post("/transcribe", response_model=TranscriptionJobResponse)
async def transcribe_evidence(
    request: TranscriptionRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Transcribe multiple evidence files.
    
    Creates a background job and returns job ID for tracking.
    """
    # Fetch evidence
    result = await db.execute(
        select(Evidence).where(Evidence.id.in_(request.evidence_ids))
    )
    evidence_items = result.scalars().all()
    
    if not evidence_items:
        raise HTTPException(status_code=404, detail="No evidence found")
    
    # Get file paths
    file_paths = [ev.file_path for ev in evidence_items if ev.file_path]
    
    # Start batch transcription
    job = await transcription_service.transcribe_batch(
        file_paths=file_paths,
        evidence_ids=request.evidence_ids,
        enhance_audio=request.enhance_audio,
        detect_speakers=request.detect_speakers,
        word_timestamps=request.word_timestamps,
        language=request.language
    )
    
    # Update evidence records with transcripts
    for result in job.results:
        if result.evidence_id:
            await db.execute(
                update(Evidence)
                .where(Evidence.id == result.evidence_id)
                .values(
                    transcript=result.full_text,
                    transcript_confidence=result.speech_ratio,
                    transcript_language=result.language,
                    transcript_uploaded_at=datetime.utcnow(),
                    processing_status=ProcessingStatus.COMPLETED,
                    detected_events={
                        "miranda_detected": result.miranda_detected,
                        "miranda_timestamp": result.miranda_timestamp,
                        "commands": result.commands_detected,
                        "threats": result.threats_detected
                    }
                )
            )
    
    await db.commit()
    
    return TranscriptionJobResponse(
        job_id=job.job_id,
        status=job.status,
        total_files=job.total_files,
        completed_files=job.completed_files,
        failed_files=job.failed_files,
        results=[asdict(r) for r in job.results],
        errors=job.errors,
        created_at=job.created_at,
        completed_at=job.completed_at
    )


@router.get("/transcribe/{job_id}", response_model=TranscriptionJobResponse)
async def get_transcription_status(job_id: str):
    """Get transcription job status"""
    job = transcription_service.get_job_status(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return TranscriptionJobResponse(
        job_id=job.job_id,
        status=job.status,
        total_files=job.total_files,
        completed_files=job.completed_files,
        failed_files=job.failed_files,
        results=[asdict(r) for r in job.results],
        errors=job.errors,
        created_at=job.created_at,
        completed_at=job.completed_at
    )


@router.post("/transcribe/single/{evidence_id}")
async def transcribe_single(
    evidence_id: int,
    enhance_audio: bool = Query(True),
    detect_speakers: bool = Query(True),
    word_timestamps: bool = Query(True),
    language: str = Query("en"),
    db: AsyncSession = Depends(get_db)
):
    """Transcribe a single evidence file"""
    result = await db.execute(select(Evidence).where(Evidence.id == evidence_id))
    evidence = result.scalar_one_or_none()
    
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")
    
    if not evidence.file_path:
        raise HTTPException(status_code=400, detail="No file path for evidence")
    
    # Transcribe
    transcription = await transcription_service.transcribe_file(
        audio_path=evidence.file_path,
        evidence_id=evidence_id,
        language=language,
        enhance_audio=enhance_audio,
        detect_speakers=detect_speakers,
        word_timestamps=word_timestamps
    )
    
    # Update evidence
    evidence.transcript = transcription.full_text
    evidence.transcript_confidence = transcription.speech_ratio
    evidence.transcript_language = transcription.language
    evidence.transcript_uploaded_at = datetime.utcnow()
    evidence.processing_status = ProcessingStatus.COMPLETED
    evidence.detected_events = {
        "miranda_detected": transcription.miranda_detected,
        "miranda_timestamp": transcription.miranda_timestamp,
        "commands": transcription.commands_detected,
        "threats": transcription.threats_detected
    }
    
    await db.commit()
    
    return asdict(transcription)


# ============================================================
# AUDIO ENHANCEMENT ENDPOINTS
# ============================================================

@router.post("/audio/enhance")
async def enhance_audio(
    request: AudioEnhanceRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Enhance audio quality of evidence file.
    
    Processing:
    - Noise reduction (spectral gating)
    - Volume normalization
    - Speech frequency enhancement
    """
    from app.services.transcription_service import AudioPreprocessor
    
    result = await db.execute(select(Evidence).where(Evidence.id == request.evidence_id))
    evidence = result.scalar_one_or_none()
    
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")
    
    if not evidence.file_path:
        raise HTTPException(status_code=400, detail="No file path")
    
    preprocessor = AudioPreprocessor()
    
    # Extract audio if video
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    input_path = evidence.file_path
    
    if Path(input_path).suffix.lower() in video_exts:
        input_path = preprocessor.extract_audio_from_video(input_path)
    
    # Enhance
    enhanced_path = AUDIO_DIR / f"enhanced_{evidence.id}_{Path(input_path).stem}.wav"
    
    preprocessor.enhance_for_transcription(
        audio_path=input_path,
        output_path=str(enhanced_path),
        noise_reduction=request.noise_reduction,
        normalize_volume=request.normalize_volume,
        enhance_speech=request.enhance_speech
    )
    
    return {
        "evidence_id": request.evidence_id,
        "original_path": evidence.file_path,
        "enhanced_path": str(enhanced_path),
        "processing": {
            "noise_reduction": request.noise_reduction,
            "normalize_volume": request.normalize_volume,
            "enhance_speech": request.enhance_speech
        },
        "message": "Audio enhanced successfully"
    }


@router.get("/audio/{evidence_id}/info")
async def get_audio_info(
    evidence_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get audio file information"""
    from app.services.transcription_service import AudioPreprocessor
    
    result = await db.execute(select(Evidence).where(Evidence.id == evidence_id))
    evidence = result.scalar_one_or_none()
    
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")
    
    if not evidence.file_path:
        raise HTTPException(status_code=400, detail="No file path")
    
    preprocessor = AudioPreprocessor()
    
    # Extract audio if video for analysis
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    analyze_path = evidence.file_path
    
    if Path(analyze_path).suffix.lower() in video_exts:
        analyze_path = preprocessor.extract_audio_from_video(analyze_path)
    
    info = preprocessor.get_audio_info(analyze_path)
    
    return {
        "evidence_id": evidence_id,
        "filename": evidence.original_filename,
        "audio_info": info
    }


@router.get("/status")
async def get_batch_upload_status():
    """Get service status"""
    return {
        "available": True,
        "transcription_available": transcription_service.is_available,
        "transcription_model": transcription_service.model,
        "active_jobs": len(transcription_service.active_jobs),
        "active_uploads": len(upload_tracker.active_uploads),
        "supported_formats": {
            "video": [".mp4", ".avi", ".mov", ".mkv", ".webm"],
            "audio": [".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"]
        },
        "features": {
            "auto_transcribe": True,
            "noise_reduction": True,
            "speech_enhancement": True,
            "miranda_detection": True,
            "command_detection": True,
            "speaker_diarization": True,
            "word_timestamps": True
        }
    }
