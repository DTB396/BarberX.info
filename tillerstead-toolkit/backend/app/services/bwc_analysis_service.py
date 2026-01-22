"""
BarberX Legal Case Management Pro Suite
BWC Analysis Service - AI-Powered Video Analysis & Real-Time Multi-POV Sync
"""
import os
import json
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import uuid

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class AnalysisType(str, Enum):
    """Types of BWC analysis"""
    INCIDENT_SUMMARY = "incident_summary"
    EXCESSIVE_FORCE = "excessive_force"
    MIRANDA_COMPLIANCE = "miranda_compliance"
    USE_OF_FORCE_CONTINUUM = "use_of_force_continuum"
    OFFICER_CONDUCT = "officer_conduct"
    TIMELINE_EXTRACTION = "timeline_extraction"
    TRANSCRIPT_ANALYSIS = "transcript_analysis"


@dataclass
class BWCVideoInfo:
    """Information about a BWC video file"""
    evidence_id: int
    filename: str
    officer_name: str
    device_id: str
    timestamp: datetime
    duration_seconds: float
    file_path: str
    transcript: Optional[str] = None
    sync_offset_ms: int = 0
    is_primary: bool = False


@dataclass
class SyncPoint:
    """A point in synchronized time across all videos"""
    sync_time_ms: int  # Master timeline position
    videos_at_point: List[Dict[str, Any]] = field(default_factory=list)
    events: List[str] = field(default_factory=list)
    is_overlap: bool = False  # True if multiple officers recording


@dataclass
class SyncTimeline:
    """Synchronized timeline for multi-POV playback"""
    sync_group_id: str
    videos: List[BWCVideoInfo]
    timeline_start: datetime
    timeline_end: datetime
    total_duration_ms: int
    sync_points: List[SyncPoint] = field(default_factory=list)
    overlap_segments: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class ForceIncident:
    """Detected use of force incident"""
    timestamp_ms: int
    timestamp_formatted: str
    officer_name: str
    force_type: str  # verbal, soft hands, hard hands, taser, firearm, etc
    force_level: int  # 1-5 on continuum
    description: str
    subject_resistance_level: int  # 1-5
    appears_justified: Optional[bool] = None
    legal_concerns: List[str] = field(default_factory=list)


@dataclass
class BWCAnalysisResult:
    """Complete analysis result for BWC footage"""
    analysis_id: str
    analysis_type: AnalysisType
    sync_group_id: Optional[str] = None
    videos_analyzed: List[str] = field(default_factory=list)
    
    # Summary
    incident_summary: str = ""
    incident_type: str = ""
    incident_date: Optional[datetime] = None
    incident_location: str = ""
    
    # Officers & Subjects
    officers_involved: List[Dict[str, Any]] = field(default_factory=list)
    subjects_involved: List[Dict[str, Any]] = field(default_factory=list)
    
    # Timeline
    key_events: List[Dict[str, Any]] = field(default_factory=list)
    timeline_summary: str = ""
    
    # Force Analysis
    force_incidents: List[ForceIncident] = field(default_factory=list)
    force_justified: Optional[bool] = None
    force_concerns: List[str] = field(default_factory=list)
    
    # Constitutional Issues
    constitutional_concerns: List[Dict[str, Any]] = field(default_factory=list)
    miranda_given: Optional[bool] = None
    miranda_timing: Optional[str] = None
    
    # Legal Assessment
    liability_score: float = 0.0
    liability_factors: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    
    # Metadata
    confidence_score: float = 0.0
    model_used: str = ""
    analysis_timestamp: str = ""


class RealTimeSyncManager:
    """
    Manages real-time synchronization of multiple BWC perspectives.
    
    Provides:
    - Master timeline coordination
    - Frame-accurate sync across videos
    - WebSocket state management for live playback
    - Event-based timeline markers
    """
    
    def __init__(self):
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    def create_sync_session(
        self,
        sync_group_id: str,
        videos: List[BWCVideoInfo]
    ) -> SyncTimeline:
        """
        Create a new synchronized playback session.
        
        Calculates offsets and creates master timeline.
        """
        if not videos:
            raise ValueError("No videos provided for sync session")
        
        # Sort by timestamp
        sorted_videos = sorted(videos, key=lambda v: v.timestamp)
        
        # Find timeline bounds
        timeline_start = sorted_videos[0].timestamp
        latest_end = max(
            v.timestamp + timedelta(seconds=v.duration_seconds)
            for v in sorted_videos
        )
        timeline_end = latest_end
        total_duration_ms = int((timeline_end - timeline_start).total_seconds() * 1000)
        
        # Calculate sync offsets relative to timeline start
        for video in sorted_videos:
            video.sync_offset_ms = int((video.timestamp - timeline_start).total_seconds() * 1000)
        
        # Find primary (earliest with longest duration)
        primary = max(sorted_videos, key=lambda v: v.duration_seconds)
        primary.is_primary = True
        
        # Calculate overlap segments
        overlap_segments = self._find_overlaps(sorted_videos, timeline_start)
        
        timeline = SyncTimeline(
            sync_group_id=sync_group_id,
            videos=sorted_videos,
            timeline_start=timeline_start,
            timeline_end=timeline_end,
            total_duration_ms=total_duration_ms,
            overlap_segments=overlap_segments
        )
        
        # Store session state
        self.active_sessions[sync_group_id] = {
            "timeline": timeline,
            "current_position_ms": 0,
            "playback_speed": 1.0,
            "is_playing": False,
            "connected_clients": set(),
            "created_at": datetime.utcnow().isoformat()
        }
        
        return timeline
    
    def _find_overlaps(
        self,
        videos: List[BWCVideoInfo],
        timeline_start: datetime
    ) -> List[Tuple[int, int]]:
        """Find time ranges where multiple officers were recording"""
        events = []
        
        for video in videos:
            start_ms = int((video.timestamp - timeline_start).total_seconds() * 1000)
            end_ms = start_ms + int(video.duration_seconds * 1000)
            events.append((start_ms, 1, video.officer_name))  # Start recording
            events.append((end_ms, -1, video.officer_name))   # Stop recording
        
        # Sort by time
        events.sort(key=lambda x: (x[0], -x[1]))
        
        overlaps = []
        active_count = 0
        overlap_start = None
        
        for time_ms, delta, _ in events:
            prev_count = active_count
            active_count += delta
            
            if prev_count <= 1 and active_count > 1:
                # Overlap started
                overlap_start = time_ms
            elif prev_count > 1 and active_count <= 1 and overlap_start is not None:
                # Overlap ended
                overlaps.append((overlap_start, time_ms))
                overlap_start = None
        
        return overlaps
    
    def get_videos_at_position(
        self,
        sync_group_id: str,
        position_ms: int
    ) -> List[Dict[str, Any]]:
        """Get all videos active at a specific timeline position"""
        if sync_group_id not in self.active_sessions:
            return []
        
        timeline = self.active_sessions[sync_group_id]["timeline"]
        active_videos = []
        
        for video in timeline.videos:
            video_start = video.sync_offset_ms
            video_end = video_start + int(video.duration_seconds * 1000)
            
            if video_start <= position_ms < video_end:
                local_position = position_ms - video_start
                active_videos.append({
                    "evidence_id": video.evidence_id,
                    "officer_name": video.officer_name,
                    "filename": video.filename,
                    "local_position_ms": local_position,
                    "is_primary": video.is_primary,
                    "progress_percent": (local_position / (video.duration_seconds * 1000)) * 100
                })
        
        return active_videos
    
    def seek(self, sync_group_id: str, position_ms: int) -> Dict[str, Any]:
        """Seek to position in synchronized timeline"""
        if sync_group_id not in self.active_sessions:
            raise ValueError(f"Session {sync_group_id} not found")
        
        session = self.active_sessions[sync_group_id]
        timeline = session["timeline"]
        
        # Clamp to valid range
        position_ms = max(0, min(position_ms, timeline.total_duration_ms))
        session["current_position_ms"] = position_ms
        
        # Get state for all videos
        active_videos = self.get_videos_at_position(sync_group_id, position_ms)
        
        # Check if in overlap
        in_overlap = any(
            start <= position_ms < end
            for start, end in timeline.overlap_segments
        )
        
        return {
            "position_ms": position_ms,
            "position_formatted": self._format_time(position_ms),
            "total_duration_ms": timeline.total_duration_ms,
            "active_videos": active_videos,
            "in_overlap": in_overlap,
            "officers_recording": len(active_videos)
        }
    
    def _format_time(self, ms: int) -> str:
        """Format milliseconds as HH:MM:SS.mmm"""
        seconds = ms // 1000
        millis = ms % 1000
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    
    def get_session_state(self, sync_group_id: str) -> Optional[Dict[str, Any]]:
        """Get current state of a sync session"""
        if sync_group_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[sync_group_id]
        timeline = session["timeline"]
        
        return {
            "sync_group_id": sync_group_id,
            "current_position_ms": session["current_position_ms"],
            "current_position_formatted": self._format_time(session["current_position_ms"]),
            "playback_speed": session["playback_speed"],
            "is_playing": session["is_playing"],
            "total_duration_ms": timeline.total_duration_ms,
            "total_duration_formatted": self._format_time(timeline.total_duration_ms),
            "timeline_start": timeline.timeline_start.isoformat(),
            "timeline_end": timeline.timeline_end.isoformat(),
            "video_count": len(timeline.videos),
            "overlap_count": len(timeline.overlap_segments),
            "videos": [
                {
                    "evidence_id": v.evidence_id,
                    "officer_name": v.officer_name,
                    "filename": v.filename,
                    "offset_ms": v.sync_offset_ms,
                    "duration_ms": int(v.duration_seconds * 1000),
                    "is_primary": v.is_primary
                }
                for v in timeline.videos
            ]
        }


class BWCAnalysisService:
    """
    AI-powered BWC footage analysis using GPT-5.2.
    
    Capabilities:
    - Bulk video analysis from transcripts
    - Use of force continuum assessment
    - Miranda compliance checking
    - Constitutional violation detection
    - Multi-POV incident reconstruction
    - Timeline extraction and synchronization
    """
    
    ANALYSIS_PROMPT = """You are an expert law enforcement video analyst and civil rights attorney.
Analyze the following Body-Worn Camera (BWC) footage transcripts and metadata.

Your analysis should cover:
1. INCIDENT SUMMARY: What happened, when, where, who was involved
2. USE OF FORCE: Every instance of force used, categorized by level
3. CONSTITUTIONAL CONCERNS: Any potential 4th/5th/6th/14th Amendment violations
4. MIRANDA: Whether Miranda warnings were given, when, and if properly administered
5. OFFICER CONDUCT: Professional conduct assessment for each officer
6. TIMELINE: Chronological sequence of key events
7. LIABILITY ASSESSMENT: Potential civil liability with estimated severity

Use of Force Continuum Levels:
1 - Officer Presence / Verbal Commands
2 - Soft Hands / Control Techniques  
3 - Hard Hands / Strikes / Takedowns
4 - OC Spray / Taser / Less-Lethal
5 - Deadly Force / Firearms

Be thorough and cite specific moments from the transcripts."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5.2"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None
        self.sync_manager = RealTimeSyncManager()
        
        if OPENAI_AVAILABLE and self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key)
    
    @property
    def is_available(self) -> bool:
        return OPENAI_AVAILABLE and self.client is not None
    
    async def analyze_bulk_footage(
        self,
        videos: List[BWCVideoInfo],
        analysis_types: List[AnalysisType] = None,
        case_context: Optional[str] = None
    ) -> BWCAnalysisResult:
        """
        Analyze multiple BWC videos together.
        
        Args:
            videos: List of BWCVideoInfo with transcripts
            analysis_types: Types of analysis to perform
            case_context: Additional context about the case
        
        Returns:
            Comprehensive BWCAnalysisResult
        """
        if not analysis_types:
            analysis_types = [
                AnalysisType.INCIDENT_SUMMARY,
                AnalysisType.EXCESSIVE_FORCE,
                AnalysisType.MIRANDA_COMPLIANCE,
                AnalysisType.OFFICER_CONDUCT
            ]
        
        analysis_id = f"bwc_analysis_{uuid.uuid4().hex[:12]}"
        
        # Build context from all videos
        video_context = self._build_video_context(videos)
        
        if not self.is_available:
            return self._fallback_analysis(analysis_id, videos, video_context)
        
        prompt = f"""{self.ANALYSIS_PROMPT}

{f'CASE CONTEXT: {case_context}' if case_context else ''}

VIDEO FOOTAGE TO ANALYZE:
{video_context}

Provide analysis in the following JSON format:
{{
    "incident_summary": "Detailed summary of what occurred",
    "incident_type": "traffic_stop|arrest|search|pursuit|other",
    "incident_location": "Location if mentioned",
    
    "officers_involved": [
        {{"name": "Officer Name", "role": "primary|backup|supervisor", "conduct_assessment": "summary"}}
    ],
    "subjects_involved": [
        {{"description": "Subject description", "charges": [], "injuries": []}}
    ],
    
    "key_events": [
        {{"timestamp": "HH:MM:SS", "officer": "Name", "description": "What happened", "significance": "high|medium|low"}}
    ],
    
    "force_incidents": [
        {{
            "timestamp": "HH:MM:SS",
            "officer": "Name",
            "force_type": "verbal|soft_hands|hard_hands|taser|firearm|other",
            "force_level": 1-5,
            "description": "What force was used",
            "subject_resistance": 1-5,
            "appears_justified": true/false,
            "legal_concerns": ["concern1", "concern2"]
        }}
    ],
    
    "constitutional_concerns": [
        {{
            "amendment": "4th|5th|6th|14th",
            "type": "excessive_force|unlawful_search|miranda|due_process",
            "description": "What happened",
            "severity": 1-5,
            "relevant_case_law": ["Case Name, Citation"]
        }}
    ],
    
    "miranda_given": true/false/null,
    "miranda_timing": "Timestamp or 'not applicable'",
    
    "liability_score": 0-100,
    "liability_factors": ["factor1", "factor2"],
    "recommended_actions": ["action1", "action2"],
    "confidence_score": 0.0-1.0
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert BWC footage analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
                max_tokens=4000
            )
            
            result_json = json.loads(response.choices[0].message.content)
            
            # Parse force incidents
            force_incidents = []
            for fi in result_json.get("force_incidents", []):
                force_incidents.append(ForceIncident(
                    timestamp_ms=self._parse_timestamp_to_ms(fi.get("timestamp", "00:00:00")),
                    timestamp_formatted=fi.get("timestamp", "00:00:00"),
                    officer_name=fi.get("officer", "Unknown"),
                    force_type=fi.get("force_type", "unknown"),
                    force_level=fi.get("force_level", 1),
                    description=fi.get("description", ""),
                    subject_resistance_level=fi.get("subject_resistance", 1),
                    appears_justified=fi.get("appears_justified"),
                    legal_concerns=fi.get("legal_concerns", [])
                ))
            
            return BWCAnalysisResult(
                analysis_id=analysis_id,
                analysis_type=analysis_types[0],
                videos_analyzed=[v.filename for v in videos],
                incident_summary=result_json.get("incident_summary", ""),
                incident_type=result_json.get("incident_type", ""),
                incident_location=result_json.get("incident_location", ""),
                officers_involved=result_json.get("officers_involved", []),
                subjects_involved=result_json.get("subjects_involved", []),
                key_events=result_json.get("key_events", []),
                force_incidents=force_incidents,
                force_justified=all(fi.appears_justified for fi in force_incidents) if force_incidents else None,
                force_concerns=[
                    concern
                    for fi in force_incidents
                    for concern in fi.legal_concerns
                ],
                constitutional_concerns=result_json.get("constitutional_concerns", []),
                miranda_given=result_json.get("miranda_given"),
                miranda_timing=result_json.get("miranda_timing"),
                liability_score=result_json.get("liability_score", 0),
                liability_factors=result_json.get("liability_factors", []),
                recommended_actions=result_json.get("recommended_actions", []),
                confidence_score=result_json.get("confidence_score", 0.0),
                model_used=self.model,
                analysis_timestamp=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            print(f"BWC Analysis error: {e}")
            return self._fallback_analysis(analysis_id, videos, video_context)
    
    def _build_video_context(self, videos: List[BWCVideoInfo]) -> str:
        """Build context string from all videos"""
        parts = []
        
        # Sort by timestamp
        sorted_videos = sorted(videos, key=lambda v: v.timestamp)
        
        for video in sorted_videos:
            part = f"""
--- VIDEO: {video.filename} ---
Officer: {video.officer_name}
Device: {video.device_id}
Start Time: {video.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {video.duration_seconds:.1f} seconds

TRANSCRIPT:
{video.transcript or '[No transcript available]'}
"""
            parts.append(part)
        
        return "\n".join(parts)
    
    def _parse_timestamp_to_ms(self, timestamp: str) -> int:
        """Parse HH:MM:SS to milliseconds"""
        try:
            parts = timestamp.split(":")
            hours = int(parts[0]) if len(parts) > 2 else 0
            minutes = int(parts[-2]) if len(parts) > 1 else 0
            seconds = float(parts[-1])
            return int((hours * 3600 + minutes * 60 + seconds) * 1000)
        except:
            return 0
    
    def _fallback_analysis(
        self,
        analysis_id: str,
        videos: List[BWCVideoInfo],
        video_context: str
    ) -> BWCAnalysisResult:
        """Fallback when AI is unavailable"""
        # Simple keyword-based analysis
        force_keywords = ["struck", "punched", "tased", "deployed", "force", "grabbed", "pushed"]
        miranda_keywords = ["right to remain silent", "miranda", "you have the right"]
        
        context_lower = video_context.lower()
        
        force_detected = any(kw in context_lower for kw in force_keywords)
        miranda_detected = any(kw in context_lower for kw in miranda_keywords)
        
        return BWCAnalysisResult(
            analysis_id=analysis_id,
            analysis_type=AnalysisType.INCIDENT_SUMMARY,
            videos_analyzed=[v.filename for v in videos],
            incident_summary="[Automated analysis - AI unavailable] Manual review required.",
            officers_involved=[
                {"name": v.officer_name, "role": "unknown"}
                for v in videos
            ],
            force_incidents=[
                ForceIncident(
                    timestamp_ms=0,
                    timestamp_formatted="00:00:00",
                    officer_name="Unknown",
                    force_type="detected",
                    force_level=3,
                    description="Force keywords detected - manual review required",
                    subject_resistance_level=0
                )
            ] if force_detected else [],
            miranda_given=miranda_detected if miranda_keywords else None,
            recommended_actions=[
                "Manual review of all footage required",
                "Transcribe audio for detailed analysis",
                "Configure OPENAI_API_KEY for AI-powered analysis"
            ],
            confidence_score=0.2,
            model_used="keyword_fallback",
            analysis_timestamp=datetime.utcnow().isoformat()
        )
    
    async def analyze_single_video(
        self,
        video: BWCVideoInfo,
        analysis_types: List[AnalysisType] = None
    ) -> BWCAnalysisResult:
        """Analyze a single video"""
        return await self.analyze_bulk_footage([video], analysis_types)
    
    def create_sync_session(
        self,
        sync_group_id: str,
        videos: List[BWCVideoInfo]
    ) -> SyncTimeline:
        """Create synchronized playback session"""
        return self.sync_manager.create_sync_session(sync_group_id, videos)
    
    def seek_sync(self, sync_group_id: str, position_ms: int) -> Dict[str, Any]:
        """Seek in synchronized timeline"""
        return self.sync_manager.seek(sync_group_id, position_ms)
    
    def get_sync_state(self, sync_group_id: str) -> Optional[Dict[str, Any]]:
        """Get current sync session state"""
        return self.sync_manager.get_session_state(sync_group_id)
    
    def get_videos_at_time(
        self,
        sync_group_id: str,
        position_ms: int
    ) -> List[Dict[str, Any]]:
        """Get all active videos at a specific time"""
        return self.sync_manager.get_videos_at_position(sync_group_id, position_ms)


# Singleton instance
bwc_analysis_service = BWCAnalysisService()
