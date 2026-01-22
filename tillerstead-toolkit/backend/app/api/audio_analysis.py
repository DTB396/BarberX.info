"""
BarberX Legal Case Management Pro Suite
Audio Analysis API - Advanced Audio Processing Endpoints
"""
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.database import get_db
from app.db.models import Evidence
from app.services.audio_analysis_service import (
    audio_analyzer,
    audio_enhancer,
    multi_track_mixer,
    AudioAnalysisResult,
    AudioEnhancementResult
)

router = APIRouter()

UPLOAD_DIR = Path("uploads/audio")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SCHEMAS
# ============================================================

class AnalyzeRequest(BaseModel):
    """Request to analyze audio"""
    evidence_id: int


class EnhanceRequest(BaseModel):
    """Request to enhance audio"""
    evidence_id: int
    noise_reduction: bool = True
    normalize_volume: bool = True
    compress_dynamics: bool = True
    enhance_speech: bool = True
    remove_dc_offset: bool = True
    equalize: bool = False


class MixTrack(BaseModel):
    """Track for mixing"""
    evidence_id: int
    offset_ms: int = 0
    label: str = ""


class MixRequest(BaseModel):
    """Request to mix multiple audio tracks"""
    tracks: List[MixTrack]
    normalize_levels: bool = True
    pan_spread: bool = True


class CompareRequest(BaseModel):
    """Request to create comparison track"""
    evidence_id_a: int
    evidence_id_b: int
    mode: str = "lr"  # "lr" = left/right, "ab" = alternating


class AudioAnalysisResponse(BaseModel):
    """Audio analysis response"""
    analysis_id: str
    filename: str
    duration_seconds: float
    sample_rate: int
    channels: int
    total_speech_time: float
    speech_percentage: float
    average_volume_db: float
    peak_volume_db: float
    noise_floor_db: float
    signal_to_noise_ratio: float
    clarity_score: float
    estimated_speaker_count: int
    voice_segments: List[Dict[str, Any]]
    shouting_detected: List[Dict[str, float]]
    gunshot_detected: List[float]
    siren_detected: List[float]
    processing_time_seconds: float


class EnhanceResponse(BaseModel):
    """Audio enhancement response"""
    original_path: str
    enhanced_path: str
    operations_applied: List[str]
    quality_improvement: float
    before_metrics: Dict[str, float]
    after_metrics: Dict[str, float]


# ============================================================
# ENDPOINTS
# ============================================================

@router.get("/status")
async def get_service_status():
    """Get audio analysis service status"""
    from app.services.audio_analysis_service import HAS_PYDUB, HAS_LIBROSA
    
    return {
        "available": True,
        "pydub_available": HAS_PYDUB,
        "librosa_available": HAS_LIBROSA,
        "features": {
            "voice_activity_detection": True,
            "speaker_estimation": HAS_LIBROSA,
            "shouting_detection": HAS_LIBROSA,
            "gunshot_detection": HAS_LIBROSA,
            "siren_detection": HAS_LIBROSA,
            "noise_reduction": HAS_LIBROSA,
            "speech_enhancement": HAS_PYDUB,
            "dynamic_compression": HAS_PYDUB,
            "multi_track_mixing": HAS_PYDUB
        }
    }


@router.post("/analyze", response_model=AudioAnalysisResponse)
async def analyze_audio(
    request: AnalyzeRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Perform comprehensive audio analysis on evidence.
    
    Analysis includes:
    - Voice activity detection (speech vs silence)
    - Volume analysis (average, peak, variance)
    - Noise floor and signal-to-noise ratio
    - Speaker count estimation
    - Event detection (shouting, gunshots, sirens)
    """
    # Get evidence
    result = await db.execute(select(Evidence).where(Evidence.id == request.evidence_id))
    evidence = result.scalar_one_or_none()
    
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")
    
    if not evidence.file_path:
        raise HTTPException(status_code=400, detail="No file path for evidence")
    
    # Extract audio if video
    audio_path = evidence.file_path
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    
    if Path(audio_path).suffix.lower() in video_exts:
        from app.services.transcription_service import AudioPreprocessor
        preprocessor = AudioPreprocessor()
        audio_path = preprocessor.extract_audio_from_video(audio_path)
    
    # Analyze
    analysis = audio_analyzer.analyze_audio(audio_path)
    
    return AudioAnalysisResponse(
        analysis_id=analysis.analysis_id,
        filename=analysis.filename,
        duration_seconds=analysis.duration_seconds,
        sample_rate=analysis.sample_rate,
        channels=analysis.channels,
        total_speech_time=analysis.total_speech_time,
        speech_percentage=analysis.speech_percentage,
        average_volume_db=analysis.average_volume_db,
        peak_volume_db=analysis.peak_volume_db,
        noise_floor_db=analysis.noise_floor_db,
        signal_to_noise_ratio=analysis.signal_to_noise_ratio,
        clarity_score=analysis.clarity_score,
        estimated_speaker_count=analysis.estimated_speaker_count,
        voice_segments=[asdict(v) for v in analysis.voice_segments],
        shouting_detected=analysis.shouting_detected,
        gunshot_detected=analysis.gunshot_detected,
        siren_detected=analysis.siren_detected,
        processing_time_seconds=analysis.processing_time_seconds
    )


@router.post("/analyze/upload")
async def analyze_uploaded_audio(
    file: UploadFile = File(...)
):
    """
    Analyze an uploaded audio file without creating evidence record.
    
    Useful for quick analysis of audio files.
    """
    valid_exts = {'.mp3', '.wav', '.m4a', '.mp4', '.avi', '.mov', '.mkv', '.webm', '.aac', '.ogg'}
    ext = Path(file.filename).suffix.lower()
    
    if ext not in valid_exts:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported format. Use: {', '.join(valid_exts)}"
        )
    
    # Save temporarily
    temp_path = UPLOAD_DIR / f"temp_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    content = await file.read()
    
    with open(temp_path, "wb") as f:
        f.write(content)
    
    try:
        # Extract audio if video
        audio_path = str(temp_path)
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        
        if ext in video_exts:
            from app.services.transcription_service import AudioPreprocessor
            preprocessor = AudioPreprocessor()
            audio_path = preprocessor.extract_audio_from_video(audio_path)
        
        # Analyze
        analysis = audio_analyzer.analyze_audio(audio_path)
        
        return asdict(analysis)
    
    finally:
        # Cleanup
        if temp_path.exists():
            os.remove(temp_path)


@router.post("/enhance", response_model=EnhanceResponse)
async def enhance_audio(
    request: EnhanceRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Enhance audio quality for better playback and transcription.
    
    Enhancement options:
    - noise_reduction: Remove background noise (spectral subtraction)
    - normalize_volume: Standardize volume levels
    - compress_dynamics: Reduce dynamic range for clearer speech
    - enhance_speech: Boost speech frequencies (400Hz-4kHz)
    - remove_dc_offset: Fix DC bias issues
    - equalize: Apply speech-optimized EQ
    """
    # Get evidence
    result = await db.execute(select(Evidence).where(Evidence.id == request.evidence_id))
    evidence = result.scalar_one_or_none()
    
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")
    
    if not evidence.file_path:
        raise HTTPException(status_code=400, detail="No file path for evidence")
    
    # Extract audio if video
    audio_path = evidence.file_path
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    
    if Path(audio_path).suffix.lower() in video_exts:
        from app.services.transcription_service import AudioPreprocessor
        preprocessor = AudioPreprocessor()
        audio_path = preprocessor.extract_audio_from_video(audio_path)
    
    # Enhance
    enhanced = audio_enhancer.enhance_audio(
        audio_path=audio_path,
        noise_reduction=request.noise_reduction,
        normalize_volume=request.normalize_volume,
        compress_dynamics=request.compress_dynamics,
        enhance_speech=request.enhance_speech,
        remove_dc_offset=request.remove_dc_offset,
        equalize=request.equalize
    )
    
    return EnhanceResponse(
        original_path=enhanced.original_path,
        enhanced_path=enhanced.enhanced_path,
        operations_applied=enhanced.operations_applied,
        quality_improvement=enhanced.quality_improvement,
        before_metrics=enhanced.before_metrics,
        after_metrics=enhanced.after_metrics
    )


@router.post("/mix")
async def mix_audio_tracks(
    request: MixRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Mix multiple audio tracks for synchronized playback.
    
    Features:
    - Time-aligned mixing with offset support
    - Volume normalization across tracks
    - Stereo panning for separation
    
    Use cases:
    - Combine multiple officer BWC audio
    - Create master timeline audio
    - Sync civilian/officer perspectives
    """
    if not request.tracks:
        raise HTTPException(status_code=400, detail="No tracks provided")
    
    # Fetch evidence and build track list
    tracks = []
    
    for track_req in request.tracks:
        result = await db.execute(
            select(Evidence).where(Evidence.id == track_req.evidence_id)
        )
        evidence = result.scalar_one_or_none()
        
        if not evidence:
            raise HTTPException(
                status_code=404, 
                detail=f"Evidence {track_req.evidence_id} not found"
            )
        
        if not evidence.file_path:
            raise HTTPException(
                status_code=400,
                detail=f"Evidence {track_req.evidence_id} has no file path"
            )
        
        # Extract audio if video
        audio_path = evidence.file_path
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        
        if Path(audio_path).suffix.lower() in video_exts:
            from app.services.transcription_service import AudioPreprocessor
            preprocessor = AudioPreprocessor()
            audio_path = preprocessor.extract_audio_from_video(audio_path)
        
        tracks.append({
            'path': audio_path,
            'offset_ms': track_req.offset_ms,
            'label': track_req.label or evidence.officer_name or f"Track {evidence.id}"
        })
    
    # Mix tracks
    output_path = multi_track_mixer.mix_tracks(
        tracks=tracks,
        normalize_levels=request.normalize_levels,
        pan_spread=request.pan_spread
    )
    
    return {
        "mixed_file": output_path,
        "track_count": len(tracks),
        "tracks": [
            {"label": t['label'], "offset_ms": t['offset_ms']} 
            for t in tracks
        ],
        "settings": {
            "normalize_levels": request.normalize_levels,
            "pan_spread": request.pan_spread
        }
    }


@router.post("/compare")
async def create_comparison_track(
    request: CompareRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Create comparison track for A/B audio analysis.
    
    Modes:
    - "lr": Track A on left channel, Track B on right channel
    - "ab": Alternating 5-second segments of A and B
    
    Use cases:
    - Compare officer statements to BWC audio
    - Verify transcript accuracy
    - Analyze audio tampering
    """
    # Get evidence A
    result_a = await db.execute(
        select(Evidence).where(Evidence.id == request.evidence_id_a)
    )
    evidence_a = result_a.scalar_one_or_none()
    
    if not evidence_a:
        raise HTTPException(status_code=404, detail=f"Evidence A not found")
    
    # Get evidence B
    result_b = await db.execute(
        select(Evidence).where(Evidence.id == request.evidence_id_b)
    )
    evidence_b = result_b.scalar_one_or_none()
    
    if not evidence_b:
        raise HTTPException(status_code=404, detail=f"Evidence B not found")
    
    # Extract audio paths
    def get_audio_path(evidence):
        audio_path = evidence.file_path
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        
        if Path(audio_path).suffix.lower() in video_exts:
            from app.services.transcription_service import AudioPreprocessor
            preprocessor = AudioPreprocessor()
            audio_path = preprocessor.extract_audio_from_video(audio_path)
        
        return audio_path
    
    path_a = get_audio_path(evidence_a)
    path_b = get_audio_path(evidence_b)
    
    # Create comparison
    comparison_path = multi_track_mixer.create_comparison_track(
        track_a=path_a,
        track_b=path_b,
        mode=request.mode
    )
    
    return {
        "comparison_file": comparison_path,
        "track_a": {
            "evidence_id": request.evidence_id_a,
            "filename": evidence_a.original_filename
        },
        "track_b": {
            "evidence_id": request.evidence_id_b,
            "filename": evidence_b.original_filename
        },
        "mode": request.mode,
        "description": "Left=Track A, Right=Track B" if request.mode == "lr" else "Alternating 5-second segments"
    }


@router.get("/{evidence_id}/voice-segments")
async def get_voice_segments(
    evidence_id: int,
    min_duration: float = Query(0.1, description="Minimum segment duration in seconds"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get voice activity segments from evidence audio.
    
    Returns list of speech segments with:
    - Start/end times
    - Duration
    - Volume level
    - Shouting/whisper detection
    """
    # Get evidence
    result = await db.execute(select(Evidence).where(Evidence.id == evidence_id))
    evidence = result.scalar_one_or_none()
    
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")
    
    if not evidence.file_path:
        raise HTTPException(status_code=400, detail="No file path for evidence")
    
    # Extract audio if video
    audio_path = evidence.file_path
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    
    if Path(audio_path).suffix.lower() in video_exts:
        from app.services.transcription_service import AudioPreprocessor
        preprocessor = AudioPreprocessor()
        audio_path = preprocessor.extract_audio_from_video(audio_path)
    
    # Analyze
    analysis = audio_analyzer.analyze_audio(audio_path)
    
    # Filter by duration
    segments = [
        asdict(v) for v in analysis.voice_segments 
        if v.duration >= min_duration
    ]
    
    return {
        "evidence_id": evidence_id,
        "filename": evidence.original_filename,
        "total_duration": analysis.duration_seconds,
        "speech_percentage": analysis.speech_percentage,
        "segment_count": len(segments),
        "segments": segments
    }


@router.get("/{evidence_id}/events")
async def get_detected_events(
    evidence_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get detected audio events from evidence.
    
    Events detected:
    - Shouting/raised voices
    - Gunshots (impulsive sounds)
    - Sirens
    """
    # Get evidence
    result = await db.execute(select(Evidence).where(Evidence.id == evidence_id))
    evidence = result.scalar_one_or_none()
    
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")
    
    if not evidence.file_path:
        raise HTTPException(status_code=400, detail="No file path for evidence")
    
    # Extract audio if video
    audio_path = evidence.file_path
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    
    if Path(audio_path).suffix.lower() in video_exts:
        from app.services.transcription_service import AudioPreprocessor
        preprocessor = AudioPreprocessor()
        audio_path = preprocessor.extract_audio_from_video(audio_path)
    
    # Analyze
    analysis = audio_analyzer.analyze_audio(audio_path)
    
    return {
        "evidence_id": evidence_id,
        "filename": evidence.original_filename,
        "duration_seconds": analysis.duration_seconds,
        "events": {
            "shouting": {
                "count": len(analysis.shouting_detected),
                "instances": analysis.shouting_detected
            },
            "gunshots": {
                "count": len(analysis.gunshot_detected),
                "timestamps": analysis.gunshot_detected
            },
            "sirens": {
                "count": len(analysis.siren_detected),
                "timestamps": analysis.siren_detected
            }
        }
    }
