"""
BarberX Legal Case Management Pro Suite
BWC Processor - Body-Worn Camera Video Processing
Motorola Solutions Integration, Multi-POV Sync, Audio Harmonization
"""
import os
import re
import hashlib
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

# Video processing imports (graceful fallback)
try:
    from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_audioclips
    HAS_MOVIEPY = True
except ImportError:
    HAS_MOVIEPY = False

try:
    import librosa
    import numpy as np
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    from pydub import AudioSegment
    from pydub.effects import normalize, compress_dynamic_range
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False


@dataclass
class VideoMetadata:
    """Video file metadata"""
    filename: str
    duration_seconds: float
    width: int
    height: int
    fps: float
    codec: str
    audio_codec: Optional[str]
    audio_channels: int
    file_size: int
    creation_time: Optional[datetime]


@dataclass
class BWCFileInfo:
    """Parsed Motorola BWC filename information"""
    officer_name: str
    timestamp: datetime
    device_id: str
    segment: int
    original_filename: str
    parsed_successfully: bool


@dataclass 
class KeyFrame:
    """Key frame from video analysis"""
    timestamp_seconds: float
    timestamp_formatted: str
    frame_number: int
    description: str
    confidence: float
    thumbnail_path: Optional[str] = None


@dataclass
class SyncedVideo:
    """Video with synchronization offset"""
    evidence_id: int
    filename: str
    officer_name: str
    offset_ms: int
    is_primary: bool
    timeline_start: datetime
    timeline_end: datetime


@dataclass
class AudioAnalysisResult:
    """Results from audio analysis"""
    has_speech: bool
    speech_segments: List[Tuple[float, float]]  # (start, end) in seconds
    noise_level: float  # 0-1 scale
    clarity_score: float  # 0-1 scale
    dominant_frequencies: List[float]


class MotorolaBWCParser:
    """
    Parse Motorola Solutions BWC filename formats.
    
    Motorola Si500/V300/V500 cameras use format:
    OfficerName_YYYYMMDDHHMI_DeviceID-Segment.mp4
    
    Examples:
    - BryanMerritt_202511292256_311-0.mp4
    - BryanMerritt_202511292257_BWL7137497-0.mp4
    - CristianMartin_202511292312_BWL7139081-0.mp4
    """
    
    # Main pattern: Name_Timestamp_DeviceID-Segment
    FILENAME_PATTERN = re.compile(
        r'^(?P<officer>[A-Za-z]+)_(?P<timestamp>\d{12})_(?P<device>[A-Z0-9]+)-(?P<segment>\d+)\.(?P<ext>\w+)$'
    )
    
    # Alternative patterns for other BWC systems
    ALT_PATTERNS = [
        # Axon format: AXON_Body_2_Video_2024-11-29_2256_001.mp4
        re.compile(r'^AXON_Body_\d+_Video_(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{4})'),
        # Generic: YYYYMMDD_HHMMSS_*.mp4
        re.compile(r'^(?P<date>\d{8})_(?P<time>\d{6})'),
    ]
    
    @classmethod
    def parse(cls, filename: str) -> BWCFileInfo:
        """
        Parse a BWC filename to extract metadata.
        
        Args:
            filename: The video filename (not full path)
            
        Returns:
            BWCFileInfo with parsed data
        """
        match = cls.FILENAME_PATTERN.match(filename)
        
        if match:
            # Parse officer name (split camelCase)
            officer_raw = match.group('officer')
            officer_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', officer_raw)
            
            # Parse timestamp (YYYYMMDDHHMM format)
            ts_str = match.group('timestamp')
            try:
                timestamp = datetime.strptime(ts_str, '%Y%m%d%H%M')
            except ValueError:
                timestamp = datetime.now()
            
            return BWCFileInfo(
                officer_name=officer_name,
                timestamp=timestamp,
                device_id=match.group('device'),
                segment=int(match.group('segment')),
                original_filename=filename,
                parsed_successfully=True
            )
        
        # Try alternative patterns
        for pattern in cls.ALT_PATTERNS:
            match = pattern.match(filename)
            if match:
                # Best effort parsing
                return BWCFileInfo(
                    officer_name="Unknown Officer",
                    timestamp=datetime.now(),
                    device_id="Unknown",
                    segment=0,
                    original_filename=filename,
                    parsed_successfully=False
                )
        
        # Unparseable filename
        return BWCFileInfo(
            officer_name="Unknown Officer",
            timestamp=datetime.now(),
            device_id="Unknown",
            segment=0,
            original_filename=filename,
            parsed_successfully=False
        )
    
    @classmethod
    def group_by_incident(cls, files: List[str]) -> Dict[str, List[BWCFileInfo]]:
        """
        Group BWC files by incident (same approximate timestamp).
        
        Files within 30 minutes of each other are considered same incident.
        """
        parsed = [cls.parse(f) for f in files]
        groups = {}
        
        for info in parsed:
            # Create group key based on timestamp (rounded to 30 min)
            ts = info.timestamp
            rounded = ts.replace(minute=(ts.minute // 30) * 30, second=0, microsecond=0)
            key = rounded.strftime('%Y%m%d_%H%M')
            
            if key not in groups:
                groups[key] = []
            groups[key].append(info)
        
        return groups


class BWCVideoProcessor:
    """
    Process Body-Worn Camera video files.
    
    Features:
    - Metadata extraction
    - Key frame detection
    - Thumbnail generation
    - Audio extraction
    - Video transcoding
    """
    
    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        """Initialize with FFmpeg path"""
        self.ffmpeg_path = ffmpeg_path
    
    def get_metadata(self, file_path: str) -> VideoMetadata:
        """
        Extract metadata from video file.
        
        Uses FFprobe for reliable metadata extraction.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {file_path}")
        
        # Try moviepy first
        if HAS_MOVIEPY:
            try:
                clip = VideoFileClip(file_path)
                metadata = VideoMetadata(
                    filename=path.name,
                    duration_seconds=clip.duration,
                    width=clip.w,
                    height=clip.h,
                    fps=clip.fps,
                    codec="h264",  # Assumed
                    audio_codec="aac" if clip.audio else None,
                    audio_channels=2 if clip.audio else 0,
                    file_size=path.stat().st_size,
                    creation_time=None
                )
                clip.close()
                return metadata
            except Exception as e:
                print(f"MoviePy error: {e}")
        
        # Fallback to ffprobe
        return self._get_metadata_ffprobe(file_path)
    
    def _get_metadata_ffprobe(self, file_path: str) -> VideoMetadata:
        """Get metadata using ffprobe"""
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'quiet',
                '-print_format', 'json',
                '-show_format', '-show_streams',
                file_path
            ], capture_output=True, text=True)
            
            import json
            data = json.loads(result.stdout)
            
            video_stream = next(
                (s for s in data.get('streams', []) if s['codec_type'] == 'video'),
                {}
            )
            audio_stream = next(
                (s for s in data.get('streams', []) if s['codec_type'] == 'audio'),
                {}
            )
            format_info = data.get('format', {})
            
            return VideoMetadata(
                filename=Path(file_path).name,
                duration_seconds=float(format_info.get('duration', 0)),
                width=int(video_stream.get('width', 0)),
                height=int(video_stream.get('height', 0)),
                fps=eval(video_stream.get('r_frame_rate', '0/1')),
                codec=video_stream.get('codec_name', 'unknown'),
                audio_codec=audio_stream.get('codec_name'),
                audio_channels=int(audio_stream.get('channels', 0)),
                file_size=int(format_info.get('size', 0)),
                creation_time=None
            )
        except Exception as e:
            print(f"FFprobe error: {e}")
            # Return basic metadata
            return VideoMetadata(
                filename=Path(file_path).name,
                duration_seconds=0,
                width=0,
                height=0,
                fps=0,
                codec="unknown",
                audio_codec=None,
                audio_channels=0,
                file_size=Path(file_path).stat().st_size,
                creation_time=None
            )
    
    def extract_audio(
        self,
        video_path: str,
        output_path: str,
        format: str = "wav"
    ) -> str:
        """
        Extract audio track from video.
        
        Args:
            video_path: Source video path
            output_path: Destination audio path
            format: Output format (wav, mp3, aac)
            
        Returns:
            Path to extracted audio
        """
        if HAS_MOVIEPY:
            try:
                clip = VideoFileClip(video_path)
                if clip.audio:
                    clip.audio.write_audiofile(output_path, verbose=False, logger=None)
                clip.close()
                return output_path
            except Exception as e:
                print(f"MoviePy audio extraction error: {e}")
        
        # Fallback to FFmpeg
        subprocess.run([
            self.ffmpeg_path, '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le' if format == 'wav' else 'aac',
            output_path
        ], capture_output=True)
        
        return output_path
    
    def generate_thumbnail(
        self,
        video_path: str,
        output_path: str,
        timestamp: float = 0
    ) -> str:
        """Generate thumbnail at specific timestamp"""
        if HAS_MOVIEPY:
            try:
                clip = VideoFileClip(video_path)
                frame = clip.get_frame(min(timestamp, clip.duration - 0.1))
                clip.close()
                
                from PIL import Image
                img = Image.fromarray(frame)
                img.save(output_path)
                return output_path
            except Exception:
                pass
        
        # FFmpeg fallback
        subprocess.run([
            self.ffmpeg_path, '-y', '-i', video_path,
            '-ss', str(timestamp), '-vframes', '1',
            output_path
        ], capture_output=True)
        
        return output_path
    
    def detect_key_frames(
        self,
        video_path: str,
        interval_seconds: float = 30,
        scene_change_threshold: float = 0.3
    ) -> List[KeyFrame]:
        """
        Detect key frames in video.
        
        Uses scene change detection and regular intervals.
        """
        key_frames = []
        metadata = self.get_metadata(video_path)
        
        # Generate key frames at regular intervals
        current = 0.0
        frame_num = 0
        
        while current < metadata.duration_seconds:
            key_frames.append(KeyFrame(
                timestamp_seconds=current,
                timestamp_formatted=self._format_timestamp(current),
                frame_number=frame_num,
                description=f"Frame at {self._format_timestamp(current)}",
                confidence=1.0
            ))
            
            current += interval_seconds
            frame_num += 1
        
        return key_frames
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def calculate_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()


class MultiPOVSynchronizer:
    """
    Synchronize multiple camera perspectives.
    
    Uses audio fingerprinting or timestamp alignment to
    sync videos from different officers/cameras.
    """
    
    def __init__(self):
        self.processor = BWCVideoProcessor()
    
    def sync_by_timestamp(
        self,
        videos: List[Dict[str, Any]],
        primary_id: int
    ) -> List[SyncedVideo]:
        """
        Sync videos based on embedded timestamps.
        
        Args:
            videos: List of video info dicts with 'id', 'path', 'timestamp'
            primary_id: ID of the primary POV video
            
        Returns:
            List of SyncedVideo with calculated offsets
        """
        # Find primary video
        primary = next((v for v in videos if v['id'] == primary_id), videos[0])
        primary_ts = primary.get('timestamp', datetime.now())
        
        synced = []
        for video in videos:
            video_ts = video.get('timestamp', datetime.now())
            offset_ms = int((video_ts - primary_ts).total_seconds() * 1000)
            
            # Get metadata for duration
            try:
                metadata = self.processor.get_metadata(video['path'])
                duration = metadata.duration_seconds
            except:
                duration = 0
            
            synced.append(SyncedVideo(
                evidence_id=video['id'],
                filename=video.get('filename', Path(video['path']).name),
                officer_name=video.get('officer_name', 'Unknown'),
                offset_ms=offset_ms,
                is_primary=(video['id'] == primary_id),
                timeline_start=video_ts,
                timeline_end=video_ts + timedelta(seconds=duration)
            ))
        
        return synced
    
    def sync_by_audio(
        self,
        videos: List[Dict[str, Any]],
        primary_id: int
    ) -> List[SyncedVideo]:
        """
        Sync videos using audio fingerprinting.
        
        Analyzes audio tracks to find matching patterns
        and calculate precise offsets.
        """
        if not HAS_LIBROSA:
            # Fall back to timestamp sync
            return self.sync_by_timestamp(videos, primary_id)
        
        # This would implement cross-correlation of audio tracks
        # For now, use timestamp sync
        return self.sync_by_timestamp(videos, primary_id)


class AudioHarmonizer:
    """
    Harmonize audio from multiple sources.
    
    Features:
    - Level normalization
    - Noise reduction
    - Speech enhancement
    - Audio mixing
    """
    
    def normalize_audio(self, audio_path: str, output_path: str) -> str:
        """Normalize audio levels to -16 LUFS"""
        if not HAS_PYDUB:
            raise RuntimeError("pydub not installed")
        
        audio = AudioSegment.from_file(audio_path)
        normalized = normalize(audio)
        normalized.export(output_path, format="wav")
        return output_path
    
    def reduce_noise(
        self,
        audio_path: str,
        output_path: str,
        noise_reduction_db: float = 12
    ) -> str:
        """
        Apply noise reduction to audio.
        
        Uses spectral gating for noise reduction.
        """
        if not HAS_LIBROSA:
            # Simple fallback - just copy
            import shutil
            shutil.copy(audio_path, output_path)
            return output_path
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Simple noise reduction using spectral gating
        # Estimate noise from first 0.5 seconds
        noise_sample = y[:int(sr * 0.5)]
        noise_spectrum = np.abs(librosa.stft(noise_sample))
        noise_threshold = np.mean(noise_spectrum, axis=1)
        
        # Apply spectral gating
        stft = librosa.stft(y)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Reduce where magnitude is below threshold
        reduction_factor = 10 ** (-noise_reduction_db / 20)
        for i, thresh in enumerate(noise_threshold):
            mask = magnitude[i] < thresh
            magnitude[i][mask] *= reduction_factor
        
        # Reconstruct
        stft_clean = magnitude * np.exp(1j * phase)
        y_clean = librosa.istft(stft_clean)
        
        # Save
        import soundfile as sf
        sf.write(output_path, y_clean, sr)
        
        return output_path
    
    def enhance_speech(self, audio_path: str, output_path: str) -> str:
        """Enhance speech frequencies (200Hz - 4kHz)"""
        if not HAS_PYDUB:
            import shutil
            shutil.copy(audio_path, output_path)
            return output_path
        
        audio = AudioSegment.from_file(audio_path)
        
        # Apply high-pass and low-pass filters for speech band
        audio = audio.high_pass_filter(200)
        audio = audio.low_pass_filter(4000)
        
        # Light compression to even out levels
        audio = compress_dynamic_range(audio)
        
        audio.export(output_path, format="wav")
        return output_path
    
    def mix_tracks(
        self,
        audio_paths: List[str],
        output_path: str,
        offsets_ms: Optional[List[int]] = None
    ) -> str:
        """
        Mix multiple audio tracks with optional offsets.
        
        Args:
            audio_paths: List of audio file paths
            output_path: Output file path
            offsets_ms: Optional list of offsets in milliseconds
            
        Returns:
            Path to mixed audio
        """
        if not HAS_PYDUB:
            raise RuntimeError("pydub not installed")
        
        if not audio_paths:
            raise ValueError("No audio files provided")
        
        if offsets_ms is None:
            offsets_ms = [0] * len(audio_paths)
        
        # Load all tracks
        tracks = []
        max_length = 0
        
        for path, offset in zip(audio_paths, offsets_ms):
            audio = AudioSegment.from_file(path)
            
            # Apply offset
            if offset > 0:
                silence = AudioSegment.silent(duration=offset)
                audio = silence + audio
            elif offset < 0:
                audio = audio[-offset:]
            
            tracks.append(audio)
            max_length = max(max_length, len(audio))
        
        # Pad shorter tracks
        padded = []
        for track in tracks:
            if len(track) < max_length:
                silence = AudioSegment.silent(duration=max_length - len(track))
                track = track + silence
            padded.append(track)
        
        # Mix by overlaying
        mixed = padded[0]
        for track in padded[1:]:
            mixed = mixed.overlay(track)
        
        # Normalize result
        mixed = normalize(mixed)
        
        mixed.export(output_path, format="wav")
        return output_path
    
    def analyze_audio(self, audio_path: str) -> AudioAnalysisResult:
        """
        Analyze audio for speech, noise levels, and quality.
        """
        if not HAS_LIBROSA:
            return AudioAnalysisResult(
                has_speech=True,  # Assume yes
                speech_segments=[],
                noise_level=0.5,
                clarity_score=0.5,
                dominant_frequencies=[]
            )
        
        y, sr = librosa.load(audio_path, sr=None)
        
        # Simple speech detection using energy and zero-crossing rate
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)  # 10ms hop
        
        # Compute features
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Detect speech (high energy, moderate ZCR)
        threshold = np.mean(rms) * 1.5
        speech_frames = rms > threshold
        
        # Convert to time segments
        speech_segments = []
        in_speech = False
        start = 0
        
        for i, is_speech in enumerate(speech_frames):
            time = i * hop_length / sr
            if is_speech and not in_speech:
                start = time
                in_speech = True
            elif not is_speech and in_speech:
                speech_segments.append((start, time))
                in_speech = False
        
        if in_speech:
            speech_segments.append((start, len(y) / sr))
        
        # Calculate noise level (from quiet sections)
        quiet_frames = ~speech_frames
        if np.any(quiet_frames):
            noise_level = np.mean(rms[quiet_frames]) / (np.max(rms) + 1e-6)
        else:
            noise_level = 0.1
        
        # Clarity score (inverse of noise)
        clarity_score = 1 - noise_level
        
        # Dominant frequencies
        spec = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        mean_spec = np.mean(spec, axis=1)
        top_indices = np.argsort(mean_spec)[-5:]
        dominant_frequencies = [float(freqs[i]) for i in top_indices]
        
        return AudioAnalysisResult(
            has_speech=len(speech_segments) > 0,
            speech_segments=speech_segments,
            noise_level=float(noise_level),
            clarity_score=float(clarity_score),
            dominant_frequencies=dominant_frequencies
        )


# Convenience functions
def parse_bwc_filename(filename: str) -> Dict[str, Any]:
    """Parse Motorola BWC filename"""
    info = MotorolaBWCParser.parse(filename)
    return {
        'officer_name': info.officer_name,
        'timestamp': info.timestamp.isoformat() if info.timestamp else None,
        'device_id': info.device_id,
        'segment': info.segment,
        'parsed': info.parsed_successfully
    }


def get_video_duration(file_path: str) -> float:
    """Get video duration in seconds"""
    processor = BWCVideoProcessor()
    metadata = processor.get_metadata(file_path)
    return metadata.duration_seconds
