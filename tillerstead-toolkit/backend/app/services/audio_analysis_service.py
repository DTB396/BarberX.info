"""
BarberX Legal Case Management Pro Suite
Advanced Audio Analysis Service - Deep Audio Processing for BWC Evidence
"""
import os
import asyncio
import tempfile
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import uuid

try:
    from pydub import AudioSegment
    from pydub.effects import normalize, compress_dynamic_range
    from pydub.silence import split_on_silence, detect_nonsilent
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

try:
    import librosa
    import numpy as np
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class VoiceActivity:
    """Voice activity detection result"""
    start_time: float  # seconds
    end_time: float
    duration: float
    is_speech: bool
    volume_level: float  # dBFS
    is_shouting: bool = False
    is_whisper: bool = False


@dataclass
class AudioSegmentInfo:
    """Information about an audio segment"""
    segment_id: str
    start_time: float
    end_time: float
    duration: float
    volume_db: float
    noise_level: float
    speech_detected: bool
    speaker_count: int = 0
    clarity_score: float = 0.0


@dataclass
class SpeakerProfile:
    """Detected speaker profile"""
    speaker_id: str
    label: str  # "Officer", "Civilian", "Unknown"
    total_speaking_time: float
    average_volume: float
    pitch_estimate: float  # Hz
    segments: List[Tuple[float, float]] = field(default_factory=list)
    is_primary: bool = False
    voice_characteristics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioAnalysisResult:
    """Complete audio analysis result"""
    analysis_id: str
    filename: str
    duration_seconds: float
    sample_rate: int
    channels: int
    
    # Voice activity
    voice_segments: List[VoiceActivity] = field(default_factory=list)
    total_speech_time: float = 0.0
    speech_percentage: float = 0.0
    
    # Volume analysis
    average_volume_db: float = 0.0
    peak_volume_db: float = 0.0
    volume_variance: float = 0.0
    
    # Audio quality
    noise_floor_db: float = 0.0
    signal_to_noise_ratio: float = 0.0
    clarity_score: float = 0.0
    
    # Speaker detection
    speakers: List[SpeakerProfile] = field(default_factory=list)
    estimated_speaker_count: int = 0
    
    # Events
    shouting_detected: List[Dict[str, float]] = field(default_factory=list)
    gunshot_detected: List[float] = field(default_factory=list)
    siren_detected: List[float] = field(default_factory=list)
    
    # Metadata
    created_at: str = ""
    processing_time_seconds: float = 0.0


@dataclass
class AudioEnhancementResult:
    """Result of audio enhancement"""
    original_path: str
    enhanced_path: str
    operations_applied: List[str]
    quality_improvement: float  # percentage
    before_metrics: Dict[str, float] = field(default_factory=dict)
    after_metrics: Dict[str, float] = field(default_factory=dict)


# ============================================================
# AUDIO ANALYZER
# ============================================================

class AdvancedAudioAnalyzer:
    """Deep audio analysis for BWC evidence"""
    
    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "audio_analysis"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_audio(self, audio_path: str) -> AudioAnalysisResult:
        """
        Perform comprehensive audio analysis.
        
        Analysis includes:
        - Voice activity detection
        - Volume analysis
        - Noise floor estimation
        - Speaker counting
        - Event detection (gunshots, sirens, shouting)
        """
        start_time = datetime.utcnow()
        analysis_id = f"analysis_{uuid.uuid4().hex[:8]}"
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load audio
        if HAS_LIBROSA:
            y, sr = librosa.load(str(audio_path), sr=None)
            duration = len(y) / sr
            channels = 1  # librosa converts to mono
        elif HAS_PYDUB:
            audio = AudioSegment.from_file(str(audio_path))
            duration = len(audio) / 1000.0
            sr = audio.frame_rate
            channels = audio.channels
            y = np.array(audio.get_array_of_samples())
            if channels == 2:
                y = y.reshape((-1, 2)).mean(axis=1)
            y = y / np.max(np.abs(y))
        else:
            raise RuntimeError("Neither librosa nor pydub available")
        
        result = AudioAnalysisResult(
            analysis_id=analysis_id,
            filename=audio_path.name,
            duration_seconds=duration,
            sample_rate=sr,
            channels=channels,
            created_at=datetime.utcnow().isoformat()
        )
        
        # Voice activity detection
        if HAS_LIBROSA:
            result.voice_segments = self._detect_voice_activity_librosa(y, sr)
        elif HAS_PYDUB:
            result.voice_segments = self._detect_voice_activity_pydub(str(audio_path))
        
        result.total_speech_time = sum(v.duration for v in result.voice_segments if v.is_speech)
        result.speech_percentage = (result.total_speech_time / duration * 100) if duration > 0 else 0
        
        # Volume analysis
        volume_stats = self._analyze_volume(y, sr)
        result.average_volume_db = volume_stats['average_db']
        result.peak_volume_db = volume_stats['peak_db']
        result.volume_variance = volume_stats['variance']
        
        # Noise analysis
        noise_stats = self._analyze_noise(y, sr)
        result.noise_floor_db = noise_stats['noise_floor']
        result.signal_to_noise_ratio = noise_stats['snr']
        result.clarity_score = min(100, max(0, noise_stats['snr'] * 5))
        
        # Speaker estimation
        if HAS_LIBROSA:
            result.estimated_speaker_count = self._estimate_speaker_count(y, sr)
        
        # Event detection
        if HAS_LIBROSA:
            result.shouting_detected = self._detect_shouting(y, sr)
            result.gunshot_detected = self._detect_impulses(y, sr, "gunshot")
            result.siren_detected = self._detect_siren(y, sr)
        
        result.processing_time_seconds = (datetime.utcnow() - start_time).total_seconds()
        
        return result
    
    def _detect_voice_activity_librosa(self, y: np.ndarray, sr: int) -> List[VoiceActivity]:
        """Detect voice activity using librosa"""
        segments = []
        
        # Compute RMS energy
        frame_length = int(sr * 0.025)  # 25ms
        hop_length = int(sr * 0.010)    # 10ms
        
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Dynamic threshold based on percentile
        threshold = np.percentile(rms, 30)
        
        # Find speech regions
        is_speech = rms > threshold
        
        # Convert to segments
        in_segment = False
        segment_start = 0
        
        for i, speech in enumerate(is_speech):
            time = i * hop_length / sr
            
            if speech and not in_segment:
                in_segment = True
                segment_start = time
            elif not speech and in_segment:
                in_segment = False
                duration = time - segment_start
                if duration > 0.1:  # Minimum 100ms
                    avg_rms = np.mean(rms[int(segment_start * sr / hop_length):i])
                    volume_db = 20 * np.log10(avg_rms + 1e-10)
                    
                    segments.append(VoiceActivity(
                        start_time=segment_start,
                        end_time=time,
                        duration=duration,
                        is_speech=True,
                        volume_level=volume_db,
                        is_shouting=volume_db > -10,
                        is_whisper=volume_db < -40
                    ))
        
        return segments
    
    def _detect_voice_activity_pydub(self, audio_path: str) -> List[VoiceActivity]:
        """Detect voice activity using pydub silence detection"""
        segments = []
        audio = AudioSegment.from_file(audio_path)
        
        # Detect non-silent chunks
        nonsilent = detect_nonsilent(
            audio,
            min_silence_len=300,   # 300ms silence
            silence_thresh=-40     # -40 dBFS threshold
        )
        
        for start_ms, end_ms in nonsilent:
            duration = (end_ms - start_ms) / 1000.0
            if duration > 0.1:
                chunk = audio[start_ms:end_ms]
                volume_db = chunk.dBFS
                
                segments.append(VoiceActivity(
                    start_time=start_ms / 1000.0,
                    end_time=end_ms / 1000.0,
                    duration=duration,
                    is_speech=True,
                    volume_level=volume_db,
                    is_shouting=volume_db > -10,
                    is_whisper=volume_db < -40
                ))
        
        return segments
    
    def _analyze_volume(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze volume characteristics"""
        # Compute RMS in dB
        rms = np.sqrt(np.mean(y ** 2))
        average_db = 20 * np.log10(rms + 1e-10)
        
        # Peak
        peak = np.max(np.abs(y))
        peak_db = 20 * np.log10(peak + 1e-10)
        
        # Variance
        frame_length = int(sr * 0.1)  # 100ms frames
        hop = frame_length // 2
        
        frame_rms = []
        for i in range(0, len(y) - frame_length, hop):
            frame = y[i:i + frame_length]
            frame_rms.append(np.sqrt(np.mean(frame ** 2)))
        
        variance = np.var(frame_rms) if frame_rms else 0
        
        return {
            'average_db': float(average_db),
            'peak_db': float(peak_db),
            'variance': float(variance)
        }
    
    def _analyze_noise(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Analyze noise characteristics"""
        # Use first and last portions as noise estimate
        noise_duration = min(int(sr * 0.5), len(y) // 10)
        noise_samples = np.concatenate([y[:noise_duration], y[-noise_duration:]])
        
        noise_rms = np.sqrt(np.mean(noise_samples ** 2))
        noise_floor = 20 * np.log10(noise_rms + 1e-10)
        
        # Signal power (using louder portions)
        signal_rms = np.sqrt(np.mean(y ** 2))
        signal_db = 20 * np.log10(signal_rms + 1e-10)
        
        snr = signal_db - noise_floor
        
        return {
            'noise_floor': float(noise_floor),
            'snr': float(max(0, snr))
        }
    
    def _estimate_speaker_count(self, y: np.ndarray, sr: int) -> int:
        """Estimate number of speakers using pitch analysis"""
        # Extract pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        
        # Get valid pitches
        valid_pitches = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if 80 < pitch < 400:  # Human voice range
                valid_pitches.append(pitch)
        
        if len(valid_pitches) < 10:
            return 1
        
        # Cluster pitches to estimate speakers
        pitches_array = np.array(valid_pitches)
        
        # Simple clustering: count distinct pitch ranges
        hist, bins = np.histogram(pitches_array, bins=10)
        
        # Count significant clusters
        threshold = np.max(hist) * 0.2
        speaker_count = sum(1 for h in hist if h > threshold)
        
        return max(1, min(speaker_count, 5))
    
    def _detect_shouting(self, y: np.ndarray, sr: int) -> List[Dict[str, float]]:
        """Detect shouting/raised voice segments"""
        shouting = []
        
        # Compute spectral centroid and RMS
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
        
        # Shouting: high energy + high spectral centroid
        rms_threshold = np.percentile(rms, 85)
        cent_threshold = np.percentile(cent, 75)
        
        hop_length = 512
        in_shout = False
        shout_start = 0
        
        for i, (r, c) in enumerate(zip(rms, cent)):
            time = i * hop_length / sr
            is_shouting = r > rms_threshold and c > cent_threshold
            
            if is_shouting and not in_shout:
                in_shout = True
                shout_start = time
            elif not is_shouting and in_shout:
                in_shout = False
                duration = time - shout_start
                if duration > 0.3:  # Min 300ms
                    shouting.append({
                        'start_time': shout_start,
                        'end_time': time,
                        'duration': duration
                    })
        
        return shouting
    
    def _detect_impulses(self, y: np.ndarray, sr: int, event_type: str) -> List[float]:
        """Detect impulsive sounds (gunshots, door slams)"""
        impulses = []
        
        # Compute onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        # Find peaks
        peaks = librosa.util.peak_pick(
            onset_env,
            pre_max=3, post_max=3,
            pre_avg=3, post_avg=5,
            delta=0.5, wait=10
        )
        
        hop_length = 512
        
        # Filter for impulsive characteristics
        for peak in peaks:
            time = peak * hop_length / sr
            
            # Get local energy
            start_sample = max(0, peak * hop_length - sr // 10)
            end_sample = min(len(y), peak * hop_length + sr // 10)
            local_energy = np.sqrt(np.mean(y[start_sample:end_sample] ** 2))
            
            # High energy spike
            if local_energy > np.mean(np.abs(y)) * 5:
                impulses.append(float(time))
        
        return impulses
    
    def _detect_siren(self, y: np.ndarray, sr: int) -> List[float]:
        """Detect siren sounds"""
        sirens = []
        
        # Sirens have characteristic frequency sweeps
        # Compute spectrogram
        D = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        
        # Siren frequencies typically 500-2000 Hz
        siren_band = (freqs > 500) & (freqs < 2000)
        siren_energy = D[siren_band].sum(axis=0)
        
        # Detect oscillating energy in siren band
        threshold = np.percentile(siren_energy, 90)
        
        hop_length = 512
        for i, energy in enumerate(siren_energy):
            if energy > threshold:
                time = i * hop_length / sr
                if not sirens or time - sirens[-1] > 2:  # 2 second gap
                    sirens.append(float(time))
        
        return sirens


# ============================================================
# AUDIO ENHANCER
# ============================================================

class AdvancedAudioEnhancer:
    """Advanced audio enhancement for BWC evidence"""
    
    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "audio_enhanced"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def enhance_audio(
        self,
        audio_path: str,
        output_path: Optional[str] = None,
        noise_reduction: bool = True,
        normalize_volume: bool = True,
        compress_dynamics: bool = True,
        enhance_speech: bool = True,
        remove_dc_offset: bool = True,
        equalize: bool = False
    ) -> AudioEnhancementResult:
        """
        Comprehensive audio enhancement.
        
        Pipeline:
        1. Remove DC offset
        2. Noise reduction (spectral subtraction)
        3. Speech frequency enhancement (400Hz-4kHz boost)
        4. Dynamic compression
        5. Volume normalization
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")
        
        if output_path is None:
            output_path = self.temp_dir / f"enhanced_{audio_path.stem}.wav"
        
        operations = []
        before_metrics = {}
        
        # Load audio
        if HAS_PYDUB:
            audio = AudioSegment.from_file(str(audio_path))
            before_metrics['volume_db'] = audio.dBFS
            before_metrics['duration'] = len(audio) / 1000.0
        else:
            raise RuntimeError("pydub required for enhancement")
        
        # Remove DC offset
        if remove_dc_offset:
            audio = self._remove_dc_offset(audio)
            operations.append("dc_offset_removal")
        
        # Noise reduction
        if noise_reduction and HAS_LIBROSA:
            audio = self._apply_noise_reduction(audio)
            operations.append("noise_reduction")
        
        # Speech enhancement
        if enhance_speech:
            audio = self._enhance_speech_frequencies(audio)
            operations.append("speech_enhancement")
        
        # Dynamic compression
        if compress_dynamics:
            audio = compress_dynamic_range(
                audio,
                threshold=-20.0,
                ratio=4.0,
                attack=5.0,
                release=50.0
            )
            operations.append("dynamic_compression")
        
        # Equalization
        if equalize:
            audio = self._apply_equalization(audio)
            operations.append("equalization")
        
        # Normalize
        if normalize_volume:
            audio = normalize(audio)
            operations.append("normalization")
        
        # Export
        audio.export(str(output_path), format="wav")
        
        # Calculate improvement
        after_metrics = {
            'volume_db': audio.dBFS,
            'duration': len(audio) / 1000.0
        }
        
        quality_improvement = max(0, (after_metrics['volume_db'] - before_metrics['volume_db']) / abs(before_metrics['volume_db']) * 100) if before_metrics['volume_db'] != 0 else 0
        
        return AudioEnhancementResult(
            original_path=str(audio_path),
            enhanced_path=str(output_path),
            operations_applied=operations,
            quality_improvement=quality_improvement,
            before_metrics=before_metrics,
            after_metrics=after_metrics
        )
    
    def _remove_dc_offset(self, audio: 'AudioSegment') -> 'AudioSegment':
        """Remove DC offset from audio"""
        samples = np.array(audio.get_array_of_samples())
        samples = samples - np.mean(samples)
        
        # Reconstruct
        return audio._spawn(samples.astype(np.int16).tobytes())
    
    def _apply_noise_reduction(self, audio: 'AudioSegment') -> 'AudioSegment':
        """Apply spectral subtraction noise reduction"""
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        samples = samples / np.max(np.abs(samples))
        
        sr = audio.frame_rate
        
        # STFT
        stft = librosa.stft(samples)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise from quiet portions
        noise_frames = magnitude[:, :10].mean(axis=1)
        
        # Spectral subtraction
        alpha = 2.0  # Over-subtraction factor
        beta = 0.01  # Spectral floor
        
        magnitude_clean = np.maximum(magnitude - alpha * noise_frames[:, np.newaxis], beta * magnitude)
        
        # Reconstruct
        stft_clean = magnitude_clean * np.exp(1j * phase)
        samples_clean = librosa.istft(stft_clean)
        
        # Scale back to int16
        samples_clean = (samples_clean * 32767).astype(np.int16)
        
        return audio._spawn(samples_clean.tobytes())
    
    def _enhance_speech_frequencies(self, audio: 'AudioSegment') -> 'AudioSegment':
        """Enhance speech frequencies (400Hz-4kHz)"""
        # Apply bandpass emphasis
        audio = audio.high_pass_filter(200)
        
        # Boost mid frequencies (where consonants live)
        # Using a combination of filters
        low_cut = audio.low_pass_filter(4000)
        high_boost = audio.high_pass_filter(1000)
        
        # Mix for speech emphasis
        enhanced = audio.overlay(low_cut, gain_during_overlay=-3)
        enhanced = enhanced.overlay(high_boost, gain_during_overlay=-6)
        
        return enhanced
    
    def _apply_equalization(self, audio: 'AudioSegment') -> 'AudioSegment':
        """Apply speech-optimized EQ"""
        # Cut low rumble
        audio = audio.high_pass_filter(80)
        
        # Presence boost at 3kHz
        presence = audio.low_pass_filter(4000).high_pass_filter(2000)
        audio = audio.overlay(presence, gain_during_overlay=-6)
        
        # Clarity boost at 5kHz
        clarity = audio.high_pass_filter(4000).low_pass_filter(6000)
        audio = audio.overlay(clarity, gain_during_overlay=-9)
        
        return audio


# ============================================================
# MULTI-TRACK MIXER
# ============================================================

class MultiTrackMixer:
    """Mix multiple BWC audio tracks for synchronized playback"""
    
    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "audio_mix"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def mix_tracks(
        self,
        tracks: List[Dict[str, Any]],
        output_path: Optional[str] = None,
        normalize_levels: bool = True,
        pan_spread: bool = True
    ) -> str:
        """
        Mix multiple audio tracks.
        
        Args:
            tracks: List of {"path": str, "offset_ms": int, "label": str}
            output_path: Output file path
            normalize_levels: Match volume levels
            pan_spread: Spread tracks in stereo field
        
        Returns:
            Path to mixed audio file
        """
        if not HAS_PYDUB:
            raise RuntimeError("pydub required for mixing")
        
        if not tracks:
            raise ValueError("No tracks provided")
        
        # Load and normalize tracks
        audio_tracks = []
        max_duration = 0
        
        for i, track in enumerate(tracks):
            audio = AudioSegment.from_file(track['path'])
            offset = track.get('offset_ms', 0)
            
            # Normalize if requested
            if normalize_levels:
                audio = normalize(audio)
            
            # Convert to stereo
            if audio.channels == 1:
                audio = audio.set_channels(2)
            
            # Pan spread
            if pan_spread and len(tracks) > 1:
                pan_position = (i / (len(tracks) - 1)) * 2 - 1  # -1 to 1
                audio = audio.pan(pan_position * 0.5)  # Limit to 50% pan
            
            audio_tracks.append({
                'audio': audio,
                'offset': offset,
                'label': track.get('label', f'Track {i+1}')
            })
            
            total_duration = offset + len(audio)
            max_duration = max(max_duration, total_duration)
        
        # Create base track
        mixed = AudioSegment.silent(duration=max_duration, frame_rate=44100)
        mixed = mixed.set_channels(2)
        
        # Overlay tracks
        for track in audio_tracks:
            mixed = mixed.overlay(track['audio'], position=track['offset'])
        
        # Export
        if output_path is None:
            output_path = self.temp_dir / f"mixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        
        mixed.export(str(output_path), format="wav")
        
        return str(output_path)
    
    def create_comparison_track(
        self,
        track_a: str,
        track_b: str,
        mode: str = "ab"  # "ab" = A/B split, "lr" = left/right
    ) -> str:
        """
        Create comparison track for side-by-side audio analysis.
        
        Modes:
        - "ab": Alternating A/B segments (5 seconds each)
        - "lr": Track A on left, Track B on right
        """
        if not HAS_PYDUB:
            raise RuntimeError("pydub required")
        
        audio_a = AudioSegment.from_file(track_a)
        audio_b = AudioSegment.from_file(track_b)
        
        # Match lengths
        max_len = max(len(audio_a), len(audio_b))
        audio_a = audio_a + AudioSegment.silent(duration=max_len - len(audio_a))
        audio_b = audio_b + AudioSegment.silent(duration=max_len - len(audio_b))
        
        if mode == "lr":
            # Left/Right comparison
            audio_a = audio_a.set_channels(1)
            audio_b = audio_b.set_channels(1)
            
            comparison = AudioSegment.from_mono_audiosegments(audio_a, audio_b)
        
        elif mode == "ab":
            # A/B alternating
            segment_length = 5000  # 5 seconds
            comparison = AudioSegment.empty()
            
            for i in range(0, max_len, segment_length):
                comparison += audio_a[i:i + segment_length]
                comparison += audio_b[i:i + segment_length]
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        output_path = self.temp_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        comparison.export(str(output_path), format="wav")
        
        return str(output_path)


# ============================================================
# SERVICE INSTANCES
# ============================================================

audio_analyzer = AdvancedAudioAnalyzer()
audio_enhancer = AdvancedAudioEnhancer()
multi_track_mixer = MultiTrackMixer()
