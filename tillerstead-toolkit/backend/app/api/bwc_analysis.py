"""
BarberX Legal Case Management Pro Suite
BWC Analysis API - Bulk Video Analysis & Real-Time Multi-POV Sync
"""
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.database import get_db
from app.db.models import Evidence, Case, AnalysisJob, ProcessingStatus
from app.services.bwc_analysis_service import (
    bwc_analysis_service, 
    BWCVideoInfo, 
    AnalysisType,
    BWCAnalysisResult
)

router = APIRouter()


# ============================================================
# PYDANTIC SCHEMAS
# ============================================================

class BWCVideoInput(BaseModel):
    """Input for a BWC video"""
    evidence_id: int
    transcript: Optional[str] = None


class BulkAnalysisRequest(BaseModel):
    """Request for bulk BWC analysis"""
    evidence_ids: List[int] = Field(..., description="List of evidence IDs to analyze")
    case_id: Optional[int] = Field(None, description="Associated case for context")
    analysis_types: Optional[List[str]] = Field(
        None,
        description="Types: incident_summary, excessive_force, miranda_compliance, use_of_force_continuum, officer_conduct"
    )
    include_transcripts: bool = Field(True, description="Include any available transcripts")
    create_sync_session: bool = Field(True, description="Create real-time sync session")


class BulkAnalysisResponse(BaseModel):
    """Response from bulk BWC analysis"""
    analysis_id: str
    videos_analyzed: int
    incident_summary: str
    officers_involved: List[dict]
    force_incidents: List[dict]
    constitutional_concerns: List[dict]
    liability_score: float
    liability_factors: List[str]
    recommended_actions: List[str]
    sync_session_id: Optional[str] = None
    confidence_score: float
    model_used: str


class CreateSyncSessionRequest(BaseModel):
    """Request to create sync session"""
    evidence_ids: List[int]
    primary_evidence_id: Optional[int] = None


class SyncSessionResponse(BaseModel):
    """Sync session info"""
    sync_group_id: str
    video_count: int
    total_duration_ms: int
    total_duration_formatted: str
    timeline_start: str
    timeline_end: str
    overlap_segments: List[List[int]]
    videos: List[dict]


class SyncSeekRequest(BaseModel):
    """Request to seek in sync session"""
    position_ms: int


class SyncStateResponse(BaseModel):
    """Current sync session state"""
    sync_group_id: str
    current_position_ms: int
    current_position_formatted: str
    playback_speed: float
    is_playing: bool
    total_duration_ms: int
    active_videos: List[dict]
    in_overlap: bool
    officers_recording: int


class TranscriptUploadRequest(BaseModel):
    """Upload transcript for video"""
    evidence_id: int
    transcript: str
    language: str = "en"


# ============================================================
# BULK ANALYSIS ENDPOINTS
# ============================================================

@router.post("/analyze/bulk", response_model=BulkAnalysisResponse)
async def analyze_bulk_bwc(
    request: BulkAnalysisRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze multiple BWC videos together with GPT-5.2.
    
    Features:
    - Combined incident reconstruction
    - Use of force continuum assessment
    - Constitutional violation detection
    - Officer conduct review
    - Timeline extraction
    - Liability scoring
    
    Optionally creates a real-time sync session for multi-POV playback.
    """
    # Fetch evidence records
    result = await db.execute(
        select(Evidence).where(Evidence.id.in_(request.evidence_ids))
    )
    evidence_items = result.scalars().all()
    
    if not evidence_items:
        raise HTTPException(status_code=404, detail="No evidence found")
    
    # Build BWCVideoInfo list
    videos = []
    for ev in evidence_items:
        videos.append(BWCVideoInfo(
            evidence_id=ev.id,
            filename=ev.original_filename or ev.filename,
            officer_name=ev.officer_name or "Unknown Officer",
            device_id=ev.bwc_device_id or "Unknown",
            timestamp=ev.start_timestamp or datetime.now(),
            duration_seconds=ev.duration_seconds or 0,
            file_path=ev.file_path or "",
            transcript=ev.transcript if request.include_transcripts else None
        ))
    
    # Get case context
    case_context = None
    if request.case_id:
        case_result = await db.execute(select(Case).where(Case.id == request.case_id))
        case = case_result.scalar_one_or_none()
        if case:
            case_context = f"Case: {case.docket_number}. {case.title}. {case.description or ''}"
    
    # Parse analysis types
    analysis_types = None
    if request.analysis_types:
        analysis_types = [AnalysisType(t) for t in request.analysis_types]
    
    # Run analysis
    analysis = await bwc_analysis_service.analyze_bulk_footage(
        videos=videos,
        analysis_types=analysis_types,
        case_context=case_context
    )
    
    # Create sync session if requested
    sync_session_id = None
    if request.create_sync_session and len(videos) > 1:
        sync_session_id = f"sync_{uuid.uuid4().hex[:8]}"
        bwc_analysis_service.create_sync_session(sync_session_id, videos)
    
    # Convert force incidents to dicts
    force_incidents = [
        {
            "timestamp_formatted": fi.timestamp_formatted,
            "officer_name": fi.officer_name,
            "force_type": fi.force_type,
            "force_level": fi.force_level,
            "description": fi.description,
            "subject_resistance_level": fi.subject_resistance_level,
            "appears_justified": fi.appears_justified,
            "legal_concerns": fi.legal_concerns
        }
        for fi in analysis.force_incidents
    ]
    
    return BulkAnalysisResponse(
        analysis_id=analysis.analysis_id,
        videos_analyzed=len(analysis.videos_analyzed),
        incident_summary=analysis.incident_summary,
        officers_involved=analysis.officers_involved,
        force_incidents=force_incidents,
        constitutional_concerns=analysis.constitutional_concerns,
        liability_score=analysis.liability_score,
        liability_factors=analysis.liability_factors,
        recommended_actions=analysis.recommended_actions,
        sync_session_id=sync_session_id,
        confidence_score=analysis.confidence_score,
        model_used=analysis.model_used
    )


@router.post("/analyze/single/{evidence_id}")
async def analyze_single_bwc(
    evidence_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Analyze a single BWC video"""
    result = await db.execute(select(Evidence).where(Evidence.id == evidence_id))
    evidence = result.scalar_one_or_none()
    
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")
    
    video = BWCVideoInfo(
        evidence_id=evidence.id,
        filename=evidence.original_filename or evidence.filename,
        officer_name=evidence.officer_name or "Unknown Officer",
        device_id=evidence.bwc_device_id or "Unknown",
        timestamp=evidence.start_timestamp or datetime.now(),
        duration_seconds=evidence.duration_seconds or 0,
        file_path=evidence.file_path or "",
        transcript=evidence.transcript
    )
    
    analysis = await bwc_analysis_service.analyze_single_video(video)
    
    return asdict(analysis)


# ============================================================
# REAL-TIME SYNC ENDPOINTS
# ============================================================

@router.post("/sync/create", response_model=SyncSessionResponse)
async def create_sync_session(
    request: CreateSyncSessionRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a real-time synchronized playback session.
    
    Aligns multiple BWC videos on a master timeline based on
    their embedded timestamps for multi-POV viewing.
    """
    # Fetch evidence
    result = await db.execute(
        select(Evidence).where(Evidence.id.in_(request.evidence_ids))
    )
    evidence_items = result.scalars().all()
    
    if not evidence_items:
        raise HTTPException(status_code=404, detail="No evidence found")
    
    if len(evidence_items) < 2:
        raise HTTPException(
            status_code=400, 
            detail="Need at least 2 videos for sync session"
        )
    
    # Build video info
    videos = [
        BWCVideoInfo(
            evidence_id=ev.id,
            filename=ev.original_filename or ev.filename,
            officer_name=ev.officer_name or "Unknown Officer",
            device_id=ev.bwc_device_id or "Unknown",
            timestamp=ev.start_timestamp or datetime.now(),
            duration_seconds=ev.duration_seconds or 300,  # Default 5 min
            file_path=ev.file_path or ""
        )
        for ev in evidence_items
    ]
    
    # Set primary if specified
    if request.primary_evidence_id:
        for v in videos:
            v.is_primary = (v.evidence_id == request.primary_evidence_id)
    
    # Create session
    sync_group_id = f"sync_{uuid.uuid4().hex[:8]}"
    timeline = bwc_analysis_service.create_sync_session(sync_group_id, videos)
    
    # Update evidence records with sync group
    for ev in evidence_items:
        ev.sync_group_id = sync_group_id
        matching_video = next((v for v in timeline.videos if v.evidence_id == ev.id), None)
        if matching_video:
            ev.sync_offset_ms = matching_video.sync_offset_ms
            ev.is_primary_pov = matching_video.is_primary
    
    await db.commit()
    
    def format_time(ms: int) -> str:
        seconds = ms // 1000
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    return SyncSessionResponse(
        sync_group_id=sync_group_id,
        video_count=len(timeline.videos),
        total_duration_ms=timeline.total_duration_ms,
        total_duration_formatted=format_time(timeline.total_duration_ms),
        timeline_start=timeline.timeline_start.isoformat(),
        timeline_end=timeline.timeline_end.isoformat(),
        overlap_segments=[list(seg) for seg in timeline.overlap_segments],
        videos=[
            {
                "evidence_id": v.evidence_id,
                "officer_name": v.officer_name,
                "filename": v.filename,
                "offset_ms": v.sync_offset_ms,
                "duration_ms": int(v.duration_seconds * 1000),
                "is_primary": v.is_primary,
                "start_formatted": format_time(v.sync_offset_ms)
            }
            for v in timeline.videos
        ]
    )


@router.get("/sync/{sync_group_id}/state", response_model=SyncStateResponse)
async def get_sync_state(sync_group_id: str):
    """Get current state of a sync session"""
    state = bwc_analysis_service.get_sync_state(sync_group_id)
    
    if not state:
        raise HTTPException(status_code=404, detail="Sync session not found")
    
    # Get active videos at current position
    active = bwc_analysis_service.get_videos_at_time(
        sync_group_id, 
        state["current_position_ms"]
    )
    
    # Check overlap
    timeline = bwc_analysis_service.sync_manager.active_sessions[sync_group_id]["timeline"]
    in_overlap = any(
        start <= state["current_position_ms"] < end
        for start, end in timeline.overlap_segments
    )
    
    return SyncStateResponse(
        sync_group_id=sync_group_id,
        current_position_ms=state["current_position_ms"],
        current_position_formatted=state["current_position_formatted"],
        playback_speed=state["playback_speed"],
        is_playing=state["is_playing"],
        total_duration_ms=state["total_duration_ms"],
        active_videos=active,
        in_overlap=in_overlap,
        officers_recording=len(active)
    )


@router.post("/sync/{sync_group_id}/seek")
async def seek_sync_session(
    sync_group_id: str,
    request: SyncSeekRequest
):
    """
    Seek to a specific position in synchronized timeline.
    
    Returns the state of all videos at that position.
    """
    try:
        result = bwc_analysis_service.seek_sync(sync_group_id, request.position_ms)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/sync/{sync_group_id}/at/{position_ms}")
async def get_videos_at_position(
    sync_group_id: str,
    position_ms: int
):
    """
    Get all videos active at a specific timeline position.
    
    Useful for determining which officers were recording at any moment.
    """
    videos = bwc_analysis_service.get_videos_at_time(sync_group_id, position_ms)
    
    if not videos:
        # Check if session exists
        state = bwc_analysis_service.get_sync_state(sync_group_id)
        if not state:
            raise HTTPException(status_code=404, detail="Sync session not found")
    
    return {
        "position_ms": position_ms,
        "active_videos": videos,
        "officer_count": len(videos)
    }


@router.get("/sync/{sync_group_id}/overlaps")
async def get_overlap_segments(sync_group_id: str):
    """
    Get all time segments where multiple officers were recording.
    
    Useful for identifying moments to compare perspectives.
    """
    state = bwc_analysis_service.get_sync_state(sync_group_id)
    
    if not state:
        raise HTTPException(status_code=404, detail="Sync session not found")
    
    timeline = bwc_analysis_service.sync_manager.active_sessions[sync_group_id]["timeline"]
    
    def format_time(ms: int) -> str:
        seconds = ms // 1000
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    overlaps = []
    for start, end in timeline.overlap_segments:
        # Get officers at this overlap
        officers = bwc_analysis_service.get_videos_at_time(sync_group_id, start)
        overlaps.append({
            "start_ms": start,
            "end_ms": end,
            "start_formatted": format_time(start),
            "end_formatted": format_time(end),
            "duration_ms": end - start,
            "officers": [v["officer_name"] for v in officers]
        })
    
    return {
        "sync_group_id": sync_group_id,
        "total_overlaps": len(overlaps),
        "overlaps": overlaps
    }


@router.delete("/sync/{sync_group_id}")
async def delete_sync_session(sync_group_id: str):
    """Delete a sync session"""
    if sync_group_id in bwc_analysis_service.sync_manager.active_sessions:
        del bwc_analysis_service.sync_manager.active_sessions[sync_group_id]
        return {"status": "deleted", "sync_group_id": sync_group_id}
    
    raise HTTPException(status_code=404, detail="Sync session not found")


# ============================================================
# WEBSOCKET FOR REAL-TIME SYNC
# ============================================================

class ConnectionManager:
    """Manage WebSocket connections for sync sessions"""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, sync_group_id: str):
        await websocket.accept()
        if sync_group_id not in self.active_connections:
            self.active_connections[sync_group_id] = []
        self.active_connections[sync_group_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, sync_group_id: str):
        if sync_group_id in self.active_connections:
            self.active_connections[sync_group_id].remove(websocket)
    
    async def broadcast(self, sync_group_id: str, message: dict):
        if sync_group_id in self.active_connections:
            for connection in self.active_connections[sync_group_id]:
                await connection.send_json(message)


ws_manager = ConnectionManager()


@router.websocket("/sync/{sync_group_id}/ws")
async def sync_websocket(websocket: WebSocket, sync_group_id: str):
    """
    WebSocket for real-time sync playback coordination.
    
    Messages:
    - seek: {"type": "seek", "position_ms": 12345}
    - play: {"type": "play"}
    - pause: {"type": "pause"}
    - speed: {"type": "speed", "value": 1.5}
    
    Server broadcasts state changes to all connected clients.
    """
    state = bwc_analysis_service.get_sync_state(sync_group_id)
    if not state:
        await websocket.close(code=4004, reason="Sync session not found")
        return
    
    await ws_manager.connect(websocket, sync_group_id)
    
    # Send initial state
    await websocket.send_json({
        "type": "state",
        "data": state
    })
    
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            
            session = bwc_analysis_service.sync_manager.active_sessions.get(sync_group_id)
            if not session:
                await websocket.send_json({"type": "error", "message": "Session expired"})
                break
            
            if msg_type == "seek":
                position_ms = data.get("position_ms", 0)
                result = bwc_analysis_service.seek_sync(sync_group_id, position_ms)
                await ws_manager.broadcast(sync_group_id, {
                    "type": "seek",
                    "data": result
                })
            
            elif msg_type == "play":
                session["is_playing"] = True
                await ws_manager.broadcast(sync_group_id, {
                    "type": "play",
                    "position_ms": session["current_position_ms"]
                })
            
            elif msg_type == "pause":
                session["is_playing"] = False
                await ws_manager.broadcast(sync_group_id, {
                    "type": "pause",
                    "position_ms": session["current_position_ms"]
                })
            
            elif msg_type == "speed":
                session["playback_speed"] = data.get("value", 1.0)
                await ws_manager.broadcast(sync_group_id, {
                    "type": "speed",
                    "value": session["playback_speed"]
                })
            
            elif msg_type == "get_state":
                current_state = bwc_analysis_service.get_sync_state(sync_group_id)
                await websocket.send_json({
                    "type": "state",
                    "data": current_state
                })
    
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, sync_group_id)


# ============================================================
# TRANSCRIPT MANAGEMENT
# ============================================================

@router.post("/transcript/upload")
async def upload_transcript(
    request: TranscriptUploadRequest,
    db: AsyncSession = Depends(get_db)
):
    """Upload or update transcript for a BWC video"""
    result = await db.execute(
        select(Evidence).where(Evidence.id == request.evidence_id)
    )
    evidence = result.scalar_one_or_none()
    
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")
    
    evidence.transcript = request.transcript
    evidence.transcript_language = request.language
    evidence.transcript_uploaded_at = datetime.utcnow()
    
    await db.commit()
    
    return {
        "evidence_id": request.evidence_id,
        "transcript_length": len(request.transcript),
        "language": request.language,
        "message": "Transcript uploaded successfully"
    }


@router.get("/transcript/{evidence_id}")
async def get_transcript(
    evidence_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get transcript for a BWC video"""
    result = await db.execute(
        select(Evidence).where(Evidence.id == evidence_id)
    )
    evidence = result.scalar_one_or_none()
    
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")
    
    return {
        "evidence_id": evidence_id,
        "filename": evidence.original_filename,
        "transcript": evidence.transcript,
        "language": evidence.transcript_language,
        "has_transcript": bool(evidence.transcript)
    }


@router.get("/status")
async def get_bwc_analysis_status():
    """Check BWC analysis service status"""
    return {
        "available": bwc_analysis_service.is_available,
        "model": bwc_analysis_service.model,
        "active_sync_sessions": len(bwc_analysis_service.sync_manager.active_sessions),
        "features": {
            "bulk_analysis": True,
            "realtime_sync": True,
            "force_continuum": True,
            "constitutional_analysis": True,
            "miranda_compliance": True,
            "websocket_sync": True
        }
    }
