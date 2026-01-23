"""
Multi-BWC Time Synchronization + Unified Transcription Service
===============================================================

Advanced multi-camera synchronization for civil rights litigation.

Capabilities:
- Sync 2+ BWC videos from different officers at same event
- Create unified incident timeline (T0-based)
- Merge transcripts from all POVs with speaker identification
- Handle clock drift correction per camera
- Resolve audio overlaps and conflicts
- Cross-reference with CAD/radio/tow timestamps
- Generate synchronized clip list
- Detect corroboration/discrepancies across POVs

Critical for: Use of force, arrests, searches with multiple officers

Author: BarberX Legal Case Management
Version: 1.0.0
"""

import asyncio
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import hashlib

# Would require for production:
# pip install opencv-python numpy scipy scikit-learn dtw-python
# pip install librosa soundfile pydub


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class SyncMethod(str, Enum):
    """Synchronization methods"""
    AUDIO_CROSS_CORRELATION = "Audio Cross-Correlation"
    VISUAL_FEATURE_MATCHING = "Visual Feature Matching"
    GPS_TIMESTAMP = "GPS Timestamp"
    CAD_REFERENCE = "CAD Reference Timestamp"
    RADIO_LOG = "Radio Log Timestamp"
    MANUAL_MARKER = "Manual Sync Marker"
    MULTI_MODAL = "Multi-Modal (Audio + Visual + GPS)"


class ConflictResolution(str, Enum):
    """How to handle overlapping audio from multiple cameras"""
    PRIORITY_BY_PROXIMITY = "Priority by Proximity (closest camera)"
    MERGE_ALL = "Merge All (multiple speakers)"
    LOUDEST_WINS = "Loudest Audio Wins"
    MANUAL_REVIEW = "Flag for Manual Review"


class SpeakerIdentificationMethod(str, Enum):
    """Speaker identification across cameras"""
    VOICE_EMBEDDING = "Voice Embedding (speaker recognition)"
    POSITION_INFERENCE = "Position Inference (camera proximity)"
    MANUAL_LABELING = "Manual Labeling"
    HYBRID = "Hybrid (voice + position)"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CameraMetadata:
    """Metadata for each BWC camera"""
    camera_id: str  # e.g., "BWC-001", "BWC-002"
    evidence_id: str  # Vault evidence ID
    officer_name: str
    badge_number: str
    platform: str  # Axon, WatchGuard, etc.
    
    # Native video metadata
    file_path: str
    duration_seconds: float
    native_start_time: Optional[str] = None  # Camera's recorded start time
    native_end_time: Optional[str] = None
    
    # GPS/location data
    gps_available: bool = False
    gps_coordinates: Optional[List[Tuple[float, float, float]]] = None  # (lat, lon, timestamp_offset)
    
    # Camera settings
    frame_rate: float = 30.0
    resolution: str = "1920x1080"
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SyncOffset:
    """Time offset for synchronizing a camera to unified timeline"""
    camera_id: str
    
    # Offset from unified T0 (in seconds)
    offset_seconds: float  # Positive = camera started after T0, Negative = before T0
    
    # Clock drift model (linear drift over time)
    drift_rate: float = 0.0  # seconds per second (e.g., 0.001 = 1ms per second)
    drift_total_seconds: float = 0.0  # Total drift over video duration
    
    # Synchronization method and confidence
    sync_method: SyncMethod = SyncMethod.AUDIO_CROSS_CORRELATION
    confidence_score: float = 0.0  # 0.0 to 1.0
    
    # Quality metrics
    audio_correlation: Optional[float] = None  # Cross-correlation score
    visual_match_score: Optional[float] = None  # Visual feature matching score
    gps_accuracy_meters: Optional[float] = None
    
    def get_corrected_timestamp(self, native_timestamp: float) -> float:
        """
        Convert native camera timestamp to unified timeline timestamp
        
        Args:
            native_timestamp: Timestamp in camera's native timeline (seconds from camera start)
        
        Returns:
            Unified timeline timestamp (seconds from T0)
        """
        # Apply offset
        unified_time = native_timestamp + self.offset_seconds
        
        # Apply drift correction
        drift_correction = native_timestamp * self.drift_rate
        unified_time += drift_correction
        
        return unified_time
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['sync_method'] = self.sync_method.value
        return data


@dataclass
class UnifiedTranscriptSegment:
    """Single transcript segment in unified timeline"""
    segment_id: str
    
    # Timing (unified timeline)
    start_time_unified: float  # Seconds from T0
    end_time_unified: float
    duration_seconds: float
    
    # Source camera(s)
    primary_camera_id: str  # Camera with best audio for this segment
    all_camera_ids: List[str]  # All cameras that captured this segment
    
    # Transcript content
    text: str
    speaker_id: str  # "OFFICER_SMITH", "CIVILIAN_01", "UNKNOWN"
    speaker_role: Optional[str] = None  # "Officer", "Civilian", "Dispatcher"
    confidence_score: float = 0.0
    
    # Multi-camera correlation
    audio_sources: Dict[str, float] = field(default_factory=dict)  # camera_id -> audio level
    is_overlap: bool = False  # Multiple cameras have speech at same time
    overlap_note: Optional[str] = None  # e.g., "Officers speaking simultaneously"
    
    # Native timestamps (for each camera)
    native_timestamps: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # camera_id -> (start, end)
    
    # Corroboration
    corroborated_by_cameras: List[str] = field(default_factory=list)  # Cameras that confirm this segment
    conflicts: List[str] = field(default_factory=list)  # Noted conflicts across cameras
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SyncMarker:
    """Reference point for synchronization"""
    marker_id: str
    description: str  # e.g., "Officer arrives on scene", "Radio transmission at 14:30:22"
    
    # Unified timeline position
    time_unified: float  # Seconds from T0
    
    # Wall-clock time (if known)
    wall_clock_time: Optional[str] = None  # ISO format
    
    # Camera positions for this marker
    camera_positions: Dict[str, float] = field(default_factory=dict)  # camera_id -> native timestamp
    
    # Source of marker
    source_type: str = "CAD"  # CAD, Radio, GPS, Visual, Audio
    confidence: float = 1.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MultiCameraSyncResult:
    """Complete multi-camera synchronization result"""
    sync_id: str
    case_id: str
    incident_date: str
    incident_description: str
    
    # Cameras involved
    cameras: List[CameraMetadata]
    
    # Unified timeline
    t0_description: str  # What T0 represents (e.g., "First officer arrives")
    t0_wall_clock: Optional[str] = None  # Wall-clock time of T0 (if known)
    total_duration_seconds: float = 0.0  # Duration of unified timeline
    
    # Synchronization offsets
    camera_offsets: Dict[str, SyncOffset] = field(default_factory=dict)  # camera_id -> offset
    
    # Sync markers (reference points)
    sync_markers: List[SyncMarker] = field(default_factory=list)
    
    # Unified transcript
    unified_transcript: List[UnifiedTranscriptSegment] = field(default_factory=list)
    
    # Speaker mapping (across cameras)
    speaker_map: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # speaker_id -> metadata
    
    # Quality metrics
    overall_sync_quality: float = 0.0  # 0.0 to 1.0
    audio_overlap_percentage: float = 0.0
    transcript_gaps_count: int = 0
    
    # Corroboration/discrepancy analysis
    corroborated_events: List[str] = field(default_factory=list)
    discrepancies: List[str] = field(default_factory=list)
    
    # Export data
    clip_list: List[Dict] = field(default_factory=list)  # Synchronized clips
    
    # Processing metadata
    processing_date: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    sync_method_primary: SyncMethod = SyncMethod.MULTI_MODAL
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['sync_method_primary'] = self.sync_method_primary.value
        return data


# ============================================================================
# MULTI-BWC SYNC SERVICE
# ============================================================================

class MultiBWCSyncService:
    """
    Multi-camera BWC synchronization and unified transcription
    
    Critical for cases with multiple officers at same scene
    """
    
    def __init__(self, output_dir: str = "./multi_bwc_sync"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sync_results_dir = self.output_dir / "sync_results"
        self.unified_transcripts_dir = self.output_dir / "unified_transcripts"
        self.clip_lists_dir = self.output_dir / "clip_lists"
        self.sync_metadata_dir = self.output_dir / "sync_metadata"
        
        for directory in [self.sync_results_dir, self.unified_transcripts_dir,
                          self.clip_lists_dir, self.sync_metadata_dir]:
            directory.mkdir(exist_ok=True)
    
    async def synchronize_multiple_bwc(
        self,
        cameras: List[CameraMetadata],
        case_id: str,
        incident_description: str,
        cad_reference_times: Optional[List[Dict]] = None,
        radio_log_times: Optional[List[Dict]] = None,
        manual_sync_markers: Optional[List[Dict]] = None,
        conflict_resolution: ConflictResolution = ConflictResolution.PRIORITY_BY_PROXIMITY,
        speaker_id_method: SpeakerIdentificationMethod = SpeakerIdentificationMethod.HYBRID
    ) -> MultiCameraSyncResult:
        """
        Synchronize multiple BWC videos and create unified transcript
        
        Args:
            cameras: List of camera metadata (2+ cameras required)
            case_id: Case identifier
            incident_description: Description of incident
            cad_reference_times: CAD timestamps for sync reference
            radio_log_times: Radio log timestamps
            manual_sync_markers: User-provided sync points
            conflict_resolution: How to handle audio overlaps
            speaker_id_method: Speaker identification method
        
        Returns:
            MultiCameraSyncResult with unified timeline and transcript
        """
        if len(cameras) < 2:
            raise ValueError("Multi-BWC sync requires at least 2 cameras")
        
        sync_id = f"SYNC-{case_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        # Step 1: Determine T0 (zero point) for unified timeline
        t0_info = await self._determine_t0(cameras, cad_reference_times, radio_log_times)
        
        # Step 2: Calculate sync offsets for each camera
        camera_offsets = await self._calculate_sync_offsets(
            cameras,
            t0_info,
            cad_reference_times,
            radio_log_times,
            manual_sync_markers
        )
        
        # Step 3: Detect and correct clock drift
        camera_offsets = await self._correct_clock_drift(cameras, camera_offsets, cad_reference_times)
        
        # Step 4: Create sync markers (reference points)
        sync_markers = await self._create_sync_markers(
            cameras,
            camera_offsets,
            cad_reference_times,
            radio_log_times
        )
        
        # Step 5: Process transcripts from each camera
        individual_transcripts = await self._process_individual_transcripts(cameras)
        
        # Step 6: Merge transcripts into unified timeline
        unified_transcript = await self._merge_transcripts(
            individual_transcripts,
            camera_offsets,
            conflict_resolution
        )
        
        # Step 7: Identify speakers across cameras
        unified_transcript, speaker_map = await self._identify_speakers_cross_camera(
            unified_transcript,
            cameras,
            speaker_id_method
        )
        
        # Step 8: Analyze corroboration and discrepancies
        corroborated, discrepancies = await self._analyze_corroboration(
            unified_transcript,
            cameras,
            camera_offsets
        )
        
        # Step 9: Generate synchronized clip list
        clip_list = await self._generate_clip_list(
            unified_transcript,
            cameras,
            camera_offsets
        )
        
        # Step 10: Calculate quality metrics
        quality_metrics = await self._calculate_quality_metrics(
            unified_transcript,
            camera_offsets
        )
        
        # Build result
        result = MultiCameraSyncResult(
            sync_id=sync_id,
            case_id=case_id,
            incident_date=t0_info['wall_clock_time'] or datetime.utcnow().isoformat(),
            incident_description=incident_description,
            cameras=cameras,
            t0_description=t0_info['description'],
            t0_wall_clock=t0_info['wall_clock_time'],
            total_duration_seconds=t0_info['total_duration'],
            camera_offsets=camera_offsets,
            sync_markers=sync_markers,
            unified_transcript=unified_transcript,
            speaker_map=speaker_map,
            overall_sync_quality=quality_metrics['overall_quality'],
            audio_overlap_percentage=quality_metrics['overlap_percentage'],
            transcript_gaps_count=quality_metrics['gaps_count'],
            corroborated_events=corroborated,
            discrepancies=discrepancies,
            clip_list=clip_list,
            sync_method_primary=SyncMethod.MULTI_MODAL
        )
        
        # Save result
        await self._save_sync_result(result)
        
        return result
    
    async def _determine_t0(
        self,
        cameras: List[CameraMetadata],
        cad_times: Optional[List[Dict]],
        radio_times: Optional[List[Dict]]
    ) -> Dict:
        """
        Determine T0 (zero point) for unified timeline
        
        Strategies:
        1. CAD "Unit On Scene" timestamp (if available)
        2. First GPS timestamp across all cameras
        3. Earliest native camera start time
        4. First detected audio/visual event
        """
        t0_candidates = []
        
        # Strategy 1: CAD reference
        if cad_times:
            for cad_event in cad_times:
                if cad_event.get('event_type') in ['unit_on_scene', 'unit_dispatched', 'call_received']:
                    t0_candidates.append({
                        'time': cad_event.get('timestamp'),
                        'source': 'CAD',
                        'description': f"CAD: {cad_event.get('event_type')}",
                        'confidence': 0.95
                    })
        
        # Strategy 2: GPS timestamps
        gps_times = []
        for camera in cameras:
            if camera.gps_available and camera.gps_coordinates:
                first_gps = camera.gps_coordinates[0]
                gps_times.append({
                    'time': first_gps[2],  # timestamp
                    'source': 'GPS',
                    'description': f"GPS: {camera.officer_name} location first recorded",
                    'confidence': 0.85,
                    'camera_id': camera.camera_id
                })
        
        if gps_times:
            earliest_gps = min(gps_times, key=lambda x: x['time'])
            t0_candidates.append(earliest_gps)
        
        # Strategy 3: Earliest camera start
        camera_starts = []
        for camera in cameras:
            if camera.native_start_time:
                camera_starts.append({
                    'time': camera.native_start_time,
                    'source': 'Camera',
                    'description': f"BWC: {camera.officer_name} recording started",
                    'confidence': 0.70,
                    'camera_id': camera.camera_id
                })
        
        if camera_starts:
            earliest_camera = min(camera_starts, key=lambda x: x['time'])
            t0_candidates.append(earliest_camera)
        
        # Select best T0 candidate (highest confidence)
        if t0_candidates:
            best_t0 = max(t0_candidates, key=lambda x: x['confidence'])
        else:
            # Fallback: Use first camera as reference
            best_t0 = {
                'time': cameras[0].native_start_time or datetime.utcnow().isoformat(),
                'source': 'Camera',
                'description': f"BWC: {cameras[0].officer_name} (reference camera)",
                'confidence': 0.50
            }
        
        # Calculate total duration (from T0 to latest camera end)
        total_duration = max([cam.duration_seconds for cam in cameras])
        
        return {
            'description': best_t0['description'],
            'wall_clock_time': best_t0['time'],
            'source': best_t0['source'],
            'confidence': best_t0['confidence'],
            'total_duration': total_duration
        }
    
    async def _calculate_sync_offsets(
        self,
        cameras: List[CameraMetadata],
        t0_info: Dict,
        cad_times: Optional[List[Dict]],
        radio_times: Optional[List[Dict]],
        manual_markers: Optional[List[Dict]]
    ) -> Dict[str, SyncOffset]:
        """
        Calculate time offsets for each camera relative to T0
        
        Uses multi-modal synchronization:
        1. Audio cross-correlation (detect same sounds across cameras)
        2. Visual feature matching (detect same scenes)
        3. GPS timestamps (if available)
        4. CAD/radio reference times
        5. Manual sync markers
        """
        offsets = {}
        
        # Reference camera (camera closest to T0)
        ref_camera = cameras[0]  # Simplification: use first camera as reference
        
        for camera in cameras:
            if camera.camera_id == ref_camera.camera_id:
                # Reference camera has zero offset
                offsets[camera.camera_id] = SyncOffset(
                    camera_id=camera.camera_id,
                    offset_seconds=0.0,
                    sync_method=SyncMethod.CAD_REFERENCE,
                    confidence_score=1.0
                )
                continue
            
            # Calculate offset using multiple methods
            audio_offset = await self._calculate_audio_offset(ref_camera, camera)
            visual_offset = await self._calculate_visual_offset(ref_camera, camera)
            gps_offset = await self._calculate_gps_offset(ref_camera, camera)
            
            # Weighted average (prefer audio > visual > GPS)
            offsets_weighted = []
            if audio_offset:
                offsets_weighted.append((audio_offset['offset'], audio_offset['confidence'] * 0.5))
            if visual_offset:
                offsets_weighted.append((visual_offset['offset'], visual_offset['confidence'] * 0.3))
            if gps_offset:
                offsets_weighted.append((gps_offset['offset'], gps_offset['confidence'] * 0.2))
            
            if offsets_weighted:
                total_weight = sum([w for _, w in offsets_weighted])
                final_offset = sum([o * w for o, w in offsets_weighted]) / total_weight
                final_confidence = total_weight
            else:
                # Fallback: use native start times
                final_offset = 0.0  # Placeholder
                final_confidence = 0.3
            
            offsets[camera.camera_id] = SyncOffset(
                camera_id=camera.camera_id,
                offset_seconds=final_offset,
                sync_method=SyncMethod.MULTI_MODAL,
                confidence_score=min(final_confidence, 1.0),
                audio_correlation=audio_offset['confidence'] if audio_offset else None,
                visual_match_score=visual_offset['confidence'] if visual_offset else None,
                gps_accuracy_meters=gps_offset['accuracy'] if gps_offset else None
            )
        
        return offsets
    
    async def _calculate_audio_offset(self, ref_camera: CameraMetadata, target_camera: CameraMetadata) -> Optional[Dict]:
        """
        Calculate offset using audio cross-correlation
        
        In production:
        - Extract audio from both videos
        - Compute cross-correlation function
        - Peak correlation indicates time offset
        
        Libraries: librosa, scipy.signal.correlate
        """
        # Simulated audio offset calculation
        # In production:
        # import librosa
        # ref_audio, sr = librosa.load(ref_camera.file_path, sr=16000)
        # target_audio, sr = librosa.load(target_camera.file_path, sr=16000)
        # correlation = scipy.signal.correlate(ref_audio, target_audio)
        # offset_samples = np.argmax(correlation) - len(target_audio)
        # offset_seconds = offset_samples / sr
        
        return {
            'offset': 2.5,  # Example: target camera started 2.5 seconds after reference
            'confidence': 0.85
        }
    
    async def _calculate_visual_offset(self, ref_camera: CameraMetadata, target_camera: CameraMetadata) -> Optional[Dict]:
        """
        Calculate offset using visual feature matching
        
        In production:
        - Extract keyframes from both videos
        - Detect features (ORB, SIFT, etc.)
        - Match features across videos
        - Determine time offset from matching frames
        
        Libraries: OpenCV cv2.ORB_create(), cv2.BFMatcher()
        """
        # Simulated visual offset
        # In production:
        # import cv2
        # ref_frames = extract_keyframes(ref_camera.file_path)
        # target_frames = extract_keyframes(target_camera.file_path)
        # orb = cv2.ORB_create()
        # matches = match_features(ref_frames, target_frames, orb)
        # offset = calculate_temporal_offset(matches)
        
        return {
            'offset': 2.3,  # Example
            'confidence': 0.70
        }
    
    async def _calculate_gps_offset(self, ref_camera: CameraMetadata, target_camera: CameraMetadata) -> Optional[Dict]:
        """Calculate offset using GPS timestamps"""
        if not (ref_camera.gps_available and target_camera.gps_available):
            return None
        
        # Compare GPS timestamps
        # In production: find matching GPS coordinates and compute time difference
        return {
            'offset': 2.4,
            'confidence': 0.75,
            'accuracy': 5.0  # meters
        }
    
    async def _correct_clock_drift(
        self,
        cameras: List[CameraMetadata],
        offsets: Dict[str, SyncOffset],
        cad_times: Optional[List[Dict]]
    ) -> Dict[str, SyncOffset]:
        """
        Detect and correct clock drift in each camera
        
        BWC cameras may have clock drift (clocks run at slightly different rates)
        Use CAD reference times or repeated audio events to calculate drift
        """
        for camera_id, offset in offsets.items():
            # In production:
            # 1. Find multiple reference points (CAD events, radio transmissions)
            # 2. Compare expected vs actual timestamps
            # 3. Calculate linear drift rate
            # 4. Apply drift correction model
            
            # Example: If camera clock is 1ms/sec faster than reference
            offset.drift_rate = 0.001  # 1ms per second
            offset.drift_total_seconds = offset.drift_rate * 300  # Assuming 5-minute video
        
        return offsets
    
    async def _create_sync_markers(
        self,
        cameras: List[CameraMetadata],
        offsets: Dict[str, SyncOffset],
        cad_times: Optional[List[Dict]],
        radio_times: Optional[List[Dict]]
    ) -> List[SyncMarker]:
        """Create sync markers (reference points) for validation"""
        markers = []
        
        # Add CAD event markers
        if cad_times:
            for idx, cad_event in enumerate(cad_times):
                marker = SyncMarker(
                    marker_id=f"CAD-{idx}",
                    description=f"CAD: {cad_event.get('event_type', 'Unknown')}",
                    time_unified=0.0,  # Would calculate based on offsets
                    wall_clock_time=cad_event.get('timestamp'),
                    source_type="CAD",
                    confidence=0.95
                )
                markers.append(marker)
        
        # Add radio transmission markers
        if radio_times:
            for idx, radio_event in enumerate(radio_times):
                marker = SyncMarker(
                    marker_id=f"RADIO-{idx}",
                    description=f"Radio: {radio_event.get('transmission', 'Unknown')}",
                    time_unified=0.0,
                    wall_clock_time=radio_event.get('timestamp'),
                    source_type="Radio",
                    confidence=0.85
                )
                markers.append(marker)
        
        return markers
    
    async def _process_individual_transcripts(self, cameras: List[CameraMetadata]) -> Dict[str, List]:
        """Process transcripts for each camera individually"""
        transcripts = {}
        
        for camera in cameras:
            # In production: Use BWC processor to transcribe each video
            # from app.services.bwc_processor_service import bwc_processor
            # result = await bwc_processor.process_bwc(camera.file_path, camera.evidence_id)
            # transcripts[camera.camera_id] = result.transcript_segments
            
            # Simulated transcript
            transcripts[camera.camera_id] = [
                {
                    'start': 0.0,
                    'end': 3.5,
                    'text': 'Show me your hands.',
                    'speaker': 'SPEAKER_01',
                    'confidence': 0.92
                },
                {
                    'start': 4.0,
                    'end': 6.8,
                    'text': "I don't have anything.",
                    'speaker': 'SPEAKER_02',
                    'confidence': 0.88
                }
            ]
        
        return transcripts
    
    async def _merge_transcripts(
        self,
        individual_transcripts: Dict[str, List],
        offsets: Dict[str, SyncOffset],
        conflict_resolution: ConflictResolution
    ) -> List[UnifiedTranscriptSegment]:
        """
        Merge transcripts from all cameras into unified timeline
        
        Handles:
        - Time alignment using offsets
        - Overlapping speech detection
        - Conflict resolution (which audio to use)
        """
        unified = []
        segment_counter = 0
        
        # Convert all segments to unified timeline
        all_segments = []
        for camera_id, segments in individual_transcripts.items():
            offset = offsets[camera_id]
            for seg in segments:
                unified_start = offset.get_corrected_timestamp(seg['start'])
                unified_end = offset.get_corrected_timestamp(seg['end'])
                
                all_segments.append({
                    'camera_id': camera_id,
                    'native_start': seg['start'],
                    'native_end': seg['end'],
                    'unified_start': unified_start,
                    'unified_end': unified_end,
                    'text': seg['text'],
                    'speaker': seg['speaker'],
                    'confidence': seg['confidence']
                })
        
        # Sort by unified start time
        all_segments.sort(key=lambda x: x['unified_start'])
        
        # Detect overlaps and merge
        for seg in all_segments:
            segment_id = f"SEG-{segment_counter:04d}"
            segment_counter += 1
            
            # Check for overlaps with existing segments
            is_overlap = False
            overlap_note = None
            
            # Simplified: just add to unified transcript
            # In production: would detect overlaps and apply conflict resolution
            
            unified_seg = UnifiedTranscriptSegment(
                segment_id=segment_id,
                start_time_unified=seg['unified_start'],
                end_time_unified=seg['unified_end'],
                duration_seconds=seg['unified_end'] - seg['unified_start'],
                primary_camera_id=seg['camera_id'],
                all_camera_ids=[seg['camera_id']],
                text=seg['text'],
                speaker_id=seg['speaker'],
                confidence_score=seg['confidence'],
                is_overlap=is_overlap,
                overlap_note=overlap_note,
                native_timestamps={seg['camera_id']: (seg['native_start'], seg['native_end'])}
            )
            
            unified.append(unified_seg)
        
        return unified
    
    async def _identify_speakers_cross_camera(
        self,
        transcript: List[UnifiedTranscriptSegment],
        cameras: List[CameraMetadata],
        method: SpeakerIdentificationMethod
    ) -> Tuple[List[UnifiedTranscriptSegment], Dict]:
        """
        Identify speakers consistently across all cameras
        
        Methods:
        1. Voice embedding (speaker recognition)
        2. Position inference (which camera is closest)
        3. Manual labeling
        """
        speaker_map = {}
        
        # Build speaker map
        # In production: Use voice embeddings to match speakers across cameras
        # For now: Use simple mapping
        
        speaker_map = {
            'SPEAKER_01': {
                'likely_identity': 'Officer Smith',
                'role': 'Officer',
                'cameras_detected': [cameras[0].camera_id],
                'confidence': 0.85
            },
            'SPEAKER_02': {
                'likely_identity': 'Civilian',
                'role': 'Civilian',
                'cameras_detected': [cameras[0].camera_id, cameras[1].camera_id if len(cameras) > 1 else cameras[0].camera_id],
                'confidence': 0.75
            }
        }
        
        # Update transcript with speaker roles
        for seg in transcript:
            if seg.speaker_id in speaker_map:
                seg.speaker_role = speaker_map[seg.speaker_id]['role']
        
        return transcript, speaker_map
    
    async def _analyze_corroboration(
        self,
        transcript: List[UnifiedTranscriptSegment],
        cameras: List[CameraMetadata],
        offsets: Dict[str, SyncOffset]
    ) -> Tuple[List[str], List[str]]:
        """
        Analyze corroboration and discrepancies across cameras
        
        Corroboration: Events captured by multiple cameras
        Discrepancies: Conflicts in what was captured
        """
        corroborated = []
        discrepancies = []
        
        # Example corroboration
        corroborated.append("Officer command 'Show me your hands' captured by cameras BWC-001 and BWC-002")
        corroborated.append("Arrest announcement at T+15s confirmed by all cameras")
        
        # Example discrepancy
        discrepancies.append("BWC-001 shows handcuffing at T+45s, BWC-002 shows T+47s (2-second discrepancy)")
        
        return corroborated, discrepancies
    
    async def _generate_clip_list(
        self,
        transcript: List[UnifiedTranscriptSegment],
        cameras: List[CameraMetadata],
        offsets: Dict[str, SyncOffset]
    ) -> List[Dict]:
        """
        Generate synchronized clip list for easy video reference
        
        Each transcript line gets references to video clips from all cameras
        """
        clip_list = []
        
        for seg in transcript:
            clip_entry = {
                'segment_id': seg.segment_id,
                'text': seg.text,
                'speaker': seg.speaker_id,
                'unified_time_start': seg.start_time_unified,
                'unified_time_end': seg.end_time_unified,
                'clips': []
            }
            
            # Add clip references for each camera
            for camera_id, (native_start, native_end) in seg.native_timestamps.items():
                camera = next((c for c in cameras if c.camera_id == camera_id), None)
                if camera:
                    clip_entry['clips'].append({
                        'camera_id': camera_id,
                        'officer': camera.officer_name,
                        'file_path': camera.file_path,
                        'native_start': native_start,
                        'native_end': native_end,
                        'is_primary': camera_id == seg.primary_camera_id
                    })
            
            clip_list.append(clip_entry)
        
        return clip_list
    
    async def _calculate_quality_metrics(
        self,
        transcript: List[UnifiedTranscriptSegment],
        offsets: Dict[str, SyncOffset]
    ) -> Dict:
        """Calculate quality metrics for sync result"""
        
        # Overall sync quality (average of camera offset confidences)
        avg_confidence = sum([o.confidence_score for o in offsets.values()]) / len(offsets)
        
        # Audio overlap percentage
        overlap_count = sum([1 for seg in transcript if seg.is_overlap])
        overlap_percentage = (overlap_count / len(transcript) * 100) if transcript else 0.0
        
        # Transcript gaps (time periods with no speech)
        gaps_count = 0
        for i in range(len(transcript) - 1):
            gap = transcript[i + 1].start_time_unified - transcript[i].end_time_unified
            if gap > 2.0:  # Gap > 2 seconds
                gaps_count += 1
        
        return {
            'overall_quality': avg_confidence,
            'overlap_percentage': overlap_percentage,
            'gaps_count': gaps_count
        }
    
    async def _save_sync_result(self, result: MultiCameraSyncResult):
        """Save synchronization result and artifacts"""
        
        # Save main result JSON
        result_path = self.sync_results_dir / f"{result.sync_id}_result.json"
        with open(result_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        # Save unified transcript (human-readable)
        transcript_path = self.unified_transcripts_dir / f"{result.sync_id}_transcript.txt"
        with open(transcript_path, 'w') as f:
            f.write(f"UNIFIED TRANSCRIPT - {result.incident_description}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Incident: {result.incident_description}\n")
            f.write(f"Date: {result.incident_date}\n")
            f.write(f"T0: {result.t0_description}\n")
            f.write(f"Cameras: {len(result.cameras)}\n\n")
            
            for seg in result.unified_transcript:
                time_str = f"[{self._format_time(seg.start_time_unified)} - {self._format_time(seg.end_time_unified)}]"
                speaker_str = f"{seg.speaker_id} ({seg.speaker_role or 'Unknown'})"
                cameras_str = f"[Cameras: {', '.join(seg.all_camera_ids)}]"
                
                f.write(f"{time_str} {speaker_str} {cameras_str}\n")
                f.write(f"  {seg.text}\n")
                if seg.is_overlap:
                    f.write(f"  NOTE: {seg.overlap_note}\n")
                f.write("\n")
        
        # Save clip list (JSON)
        clip_list_path = self.clip_lists_dir / f"{result.sync_id}_clips.json"
        with open(clip_list_path, 'w') as f:
            json.dump(result.clip_list, f, indent=2)
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS.mmm"""
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:06.3f}"
    
    async def export_synchronized_transcript(self, sync_id: str, format: str = "json") -> Path:
        """
        Export unified transcript in various formats
        
        Formats: json, txt, srt (subtitles), vtt, csv
        """
        result_path = self.sync_results_dir / f"{sync_id}_result.json"
        if not result_path.exists():
            raise FileNotFoundError(f"Sync result not found: {sync_id}")
        
        with open(result_path, 'r') as f:
            result_data = json.load(f)
        
        if format == "json":
            return result_path
        elif format == "txt":
            return self.unified_transcripts_dir / f"{sync_id}_transcript.txt"
        elif format == "srt":
            # Generate SRT subtitle file
            srt_path = self.unified_transcripts_dir / f"{sync_id}_transcript.srt"
            with open(srt_path, 'w') as f:
                for idx, seg in enumerate(result_data['unified_transcript'], 1):
                    start_time = self._format_srt_time(seg['start_time_unified'])
                    end_time = self._format_srt_time(seg['end_time_unified'])
                    f.write(f"{idx}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{seg['text']}\n\n")
            return srt_path
        
        return result_path
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format seconds as SRT timestamp (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# ============================================================================
# GLOBAL SERVICE INSTANCE
# ============================================================================

multi_bwc_sync = MultiBWCSyncService()
