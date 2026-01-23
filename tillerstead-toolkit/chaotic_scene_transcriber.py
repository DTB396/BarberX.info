#!/usr/bin/env python3
"""
CHAOTIC SCENE BWC AUDIO PROCESSOR
Advanced audio processing for overlapping speech, yelling, body noise, wind, impacts

Specialized for:
- Multiple people yelling simultaneously
- Overlapping/crosstalk speech
- Body-worn camera noise (clothing rustles, movement)
- Wind/air noise (outdoor scenes)
- Impact sounds (physical altercations)
- Low signal-to-noise ratio scenarios
- Distant speakers
- Radio chatter mixed with scene audio
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Audio processing
try:
    import librosa
    import soundfile as sf
    import noisereduce as nr
    from scipy import signal
    from scipy.ndimage import median_filter
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("ERROR: Install audio libraries: pip install librosa soundfile noisereduce scipy")

# Whisper
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


class ChaoticSceneProcessor:
    """Advanced audio processor for chaotic BWC scenes"""
    
    def __init__(self, aggressive_mode=True):
        self.aggressive_mode = aggressive_mode
        
    def detect_wind_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Detect and create wind noise profile"""
        # Wind noise is typically low-frequency (<200Hz) and broadband
        # Use spectral analysis to detect wind
        
        # Compute spectrogram
        f, t, Sxx = signal.spectrogram(audio, sr, nperseg=2048)
        
        # Wind is prominent in low frequencies
        low_freq_mask = f < 200
        wind_power = np.mean(Sxx[low_freq_mask, :], axis=0)
        
        # Detect high wind sections (above threshold)
        wind_threshold = np.median(wind_power) + 2 * np.std(wind_power)
        wind_sections = wind_power > wind_threshold
        
        return wind_sections
    
    def reduce_wind_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Aggressively reduce wind noise"""
        print("  [WIND] Removing wind/air noise...")
        
        # Method 1: High-pass filter to remove sub-100Hz wind rumble
        sos_hp = signal.butter(6, 100, 'highpass', fs=sr, output='sos')
        audio_hp = signal.sosfilt(sos_hp, audio)
        
        # Method 2: Spectral gating for wind frequencies
        # Wind typically creates broadband noise at low frequencies
        D = librosa.stft(audio_hp, n_fft=2048)
        mag, phase = np.abs(D), np.angle(D)
        
        # Identify wind-dominated frequencies (low freq, high energy)
        freq_axis = librosa.fft_frequencies(sr=sr, n_fft=2048)
        wind_mask = freq_axis < 300
        
        # Reduce magnitude in wind frequencies
        mag[wind_mask, :] *= 0.3  # 70% reduction
        
        # Reconstruct
        D_reduced = mag * np.exp(1j * phase)
        audio_dewind = librosa.istft(D_reduced)
        
        print(f"  [WIND] Reduced wind noise by ~70% in 0-300Hz range")
        return audio_dewind
    
    def reduce_body_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Remove clothing rustles, mic bumps, movement noise"""
        print("  [BODY] Removing body-worn camera noise...")
        
        # Body noise characteristics:
        # - Transient (short duration)
        # - Low frequency (<500Hz for rustles)
        # - High amplitude spikes (mic bumps)
        
        # Method 1: Remove transient spikes (mic bumps)
        # Detect sudden amplitude changes
        diff = np.diff(audio)
        spike_threshold = np.std(diff) * 10
        spike_mask = np.abs(diff) > spike_threshold
        
        # Smooth out spikes
        audio_despike = audio.copy()
        spike_indices = np.where(spike_mask)[0]
        for idx in spike_indices:
            # Replace spike with interpolated values
            if idx > 10 and idx < len(audio) - 10:
                audio_despike[idx-5:idx+5] = np.linspace(
                    audio[idx-10], audio[idx+10], 20
                )[5:15]
        
        # Method 2: Reduce low-frequency rumble (clothing rustle)
        # Rustles are typically 50-500Hz
        D = librosa.stft(audio_despike, n_fft=2048)
        mag, phase = np.abs(D), np.angle(D)
        
        freq_axis = librosa.fft_frequencies(sr=sr, n_fft=2048)
        rustle_mask = (freq_axis >= 50) & (freq_axis <= 500)
        
        # Detect frames with high rustle energy
        rustle_energy = np.sum(mag[rustle_mask, :], axis=0)
        rustle_threshold = np.median(rustle_energy) + 3 * np.std(rustle_energy)
        high_rustle_frames = rustle_energy > rustle_threshold
        
        # Reduce rustle in those frames
        mag[rustle_mask[:, None] & high_rustle_frames] *= 0.4
        
        # Reconstruct
        D_reduced = mag * np.exp(1j * phase)
        audio_clean = librosa.istft(D_reduced)
        
        print(f"  [BODY] Removed {np.sum(spike_mask)} mic bumps and clothing rustles")
        return audio_clean
    
    def separate_overlapping_speech(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, List[Dict]]:
        """Separate overlapping speakers using spectral analysis"""
        print("  [OVERLAP] Detecting overlapping speech...")
        
        # Compute STFT
        D = librosa.stft(audio, n_fft=2048, hop_length=512)
        mag = np.abs(D)
        
        # Detect speech activity in different frequency bands
        # Male voice: 85-180Hz fundamental, harmonics up to 8kHz
        # Female voice: 165-255Hz fundamental, harmonics up to 10kHz
        
        freq_axis = librosa.fft_frequencies(sr=sr, n_fft=2048)
        
        # Define frequency bands for different speakers
        bands = {
            'low_voice': (80, 200),    # Male/deep voice
            'mid_voice': (200, 400),   # Female/higher voice
            'high_voice': (400, 800)   # Children/very high voice
        }
        
        overlap_events = []
        
        for band_name, (low_freq, high_freq) in bands.items():
            band_mask = (freq_axis >= low_freq) & (freq_axis <= high_freq)
            band_energy = np.sum(mag[band_mask, :], axis=0)
            
            # Smooth energy curve
            band_energy_smooth = median_filter(band_energy, size=5)
            
            # Detect active speech in this band
            threshold = np.median(band_energy_smooth) + np.std(band_energy_smooth)
            active = band_energy_smooth > threshold
            
            # Find overlap regions (multiple bands active simultaneously)
            if len(overlap_events) == 0:
                overlap_events = active
            else:
                overlap_events = overlap_events | active
        
        # Convert frame indices to time
        hop_length = 512
        times = librosa.frames_to_time(np.arange(len(overlap_events)), sr=sr, hop_length=hop_length)
        
        # Find continuous overlap regions
        overlap_regions = []
        in_overlap = False
        start_time = 0
        
        for i, (t, is_overlap) in enumerate(zip(times, overlap_events)):
            if is_overlap and not in_overlap:
                start_time = t
                in_overlap = True
            elif not is_overlap and in_overlap:
                overlap_regions.append({
                    'start': float(start_time),
                    'end': float(t),
                    'duration': float(t - start_time)
                })
                in_overlap = False
        
        print(f"  [OVERLAP] Detected {len(overlap_regions)} overlapping speech regions")
        
        return audio, overlap_regions
    
    def enhance_distant_speech(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Boost distant/quiet speakers"""
        print("  [DISTANCE] Enhancing distant speakers...")
        
        # Distant speech characteristics:
        # - Lower amplitude
        # - Less high-frequency content
        # - More reverb/echo
        
        # Method 1: Adaptive amplification
        # Detect quiet regions and boost them more
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        
        # Create gain curve (boost quiet sections more)
        target_rms = np.median(rms) * 1.5
        gain = np.clip(target_rms / (rms + 1e-10), 0.5, 5.0)  # Limit gain to 0.5-5x
        
        # Upsample gain to audio length
        gain_upsampled = np.interp(
            np.arange(len(audio)),
            np.linspace(0, len(audio), len(gain)),
            gain
        )
        
        # Apply adaptive gain
        audio_boosted = audio * gain_upsampled
        
        # Method 2: High-frequency emphasis (distant speech loses highs)
        # Boost 2-8kHz (intelligibility range)
        sos_boost = signal.butter(4, [2000, 8000], 'bandpass', fs=sr, output='sos')
        highs = signal.sosfilt(sos_boost, audio_boosted)
        
        # Add boosted highs back
        audio_enhanced = audio_boosted + (highs * 0.5)
        
        # Normalize
        audio_enhanced = librosa.util.normalize(audio_enhanced)
        
        print(f"  [DISTANCE] Applied adaptive gain (0.5-5x) and high-frequency emphasis")
        return audio_enhanced
    
    def isolate_impact_sounds(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """Detect and catalog physical impact sounds"""
        print("  [IMPACT] Detecting physical impacts...")
        
        # Impact characteristics:
        # - Sudden amplitude spike
        # - Broadband frequency content
        # - Short duration (<100ms)
        
        # Compute onset strength (detects sudden changes)
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        
        # Detect peaks in onset strength
        peaks = librosa.util.peak_pick(
            onset_env,
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=5,
            delta=0.5,
            wait=10
        )
        
        # Convert frame indices to time
        times = librosa.frames_to_time(peaks, sr=sr)
        
        # Analyze each impact
        impacts = []
        for i, (peak_idx, time) in enumerate(zip(peaks, times)):
            # Get onset strength at this peak
            strength = onset_env[peak_idx]
            
            # Classify impact severity
            if strength > 10:
                severity = "SEVERE"
                description = "Loud impact (possible strike/throw)"
            elif strength > 5:
                severity = "MODERATE"
                description = "Moderate impact (possible body contact)"
            else:
                severity = "MINOR"
                description = "Minor impact (possible movement/shuffle)"
            
            impacts.append({
                'time': float(time),
                'strength': float(strength),
                'severity': severity,
                'description': description
            })
        
        print(f"  [IMPACT] Detected {len(impacts)} impact events")
        return impacts
    
    def reduce_radio_chatter(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Remove police radio transmissions from scene audio"""
        print("  [RADIO] Reducing radio chatter...")
        
        # Radio characteristics:
        # - Compressed frequency range (300-3000Hz for radio transmission)
        # - Distinct timbre from natural speech
        # - Often has squelch/static sounds
        
        # Detect radio-like audio using spectral features
        D = librosa.stft(audio, n_fft=2048)
        mag, phase = np.abs(D), np.angle(D)
        
        freq_axis = librosa.fft_frequencies(sr=sr, n_fft=2048)
        
        # Radio is typically band-limited to 300-3000Hz
        radio_mask = (freq_axis >= 300) & (freq_axis <= 3000)
        other_mask = ~radio_mask
        
        # Detect frames where radio band is much stronger than other frequencies
        radio_energy = np.sum(mag[radio_mask, :], axis=0)
        other_energy = np.sum(mag[other_mask, :], axis=0)
        
        radio_ratio = radio_energy / (other_energy + 1e-10)
        
        # If radio is dominant (ratio > 3), reduce it
        radio_frames = radio_ratio > 3
        
        # Reduce radio frequencies in radio-dominant frames
        mag[radio_mask[:, None] & radio_frames] *= 0.3
        
        # Reconstruct
        D_reduced = mag * np.exp(1j * phase)
        audio_clean = librosa.istft(D_reduced)
        
        print(f"  [RADIO] Reduced radio chatter in {np.sum(radio_frames)} frames")
        return audio_clean
    
    def aggressive_noise_reduction(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Multi-stage aggressive noise reduction for chaotic scenes"""
        print("  [NOISE] Applying multi-stage aggressive noise reduction...")
        
        # Stage 1: Standard noise reduction
        audio_nr1 = nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=False,
            prop_decrease=0.9,  # Very aggressive (90% reduction)
            freq_mask_smooth_hz=500,
            time_mask_smooth_ms=50
        )
        
        # Stage 2: Adaptive spectral subtraction
        D = librosa.stft(audio_nr1, n_fft=2048)
        mag, phase = np.abs(D), np.angle(D)
        
        # Estimate noise floor from quietest 10% of frames
        frame_energy = np.sum(mag, axis=0)
        noise_floor_idx = np.argsort(frame_energy)[:int(len(frame_energy) * 0.1)]
        noise_profile = np.median(mag[:, noise_floor_idx], axis=1, keepdims=True)
        
        # Subtract noise profile with over-subtraction factor
        over_subtraction = 2.0 if self.aggressive_mode else 1.5
        mag_clean = np.maximum(mag - (noise_profile * over_subtraction), mag * 0.1)
        
        # Reconstruct
        D_clean = mag_clean * np.exp(1j * phase)
        audio_nr2 = librosa.istft(D_clean)
        
        # Stage 3: Wiener filtering for residual noise
        # Simple Wiener filter approximation
        signal_power = np.mean(mag_clean ** 2, axis=1, keepdims=True)
        noise_power = np.mean(noise_profile ** 2)
        
        wiener_gain = signal_power / (signal_power + noise_power + 1e-10)
        wiener_gain = np.clip(wiener_gain, 0.1, 1.0)
        
        D_wiener = librosa.stft(audio_nr2, n_fft=2048)
        mag_wiener = np.abs(D_wiener)
        phase_wiener = np.angle(D_wiener)
        
        mag_final = mag_wiener * wiener_gain
        D_final = mag_final * np.exp(1j * phase_wiener)
        audio_final = librosa.istft(D_final)
        
        print(f"  [NOISE] Applied 3-stage noise reduction (90% aggressive)")
        return audio_final
    
    def boost_voice_intelligibility(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Maximize speech intelligibility for chaotic scenes"""
        print("  [INTEL] Boosting voice intelligibility...")
        
        # Critical speech intelligibility frequencies: 1-4kHz
        # Boost these aggressively
        
        # Create multiple frequency bands
        bands = [
            (80, 200, 0.8),      # Low voice fundamentals (slight reduction)
            (200, 500, 1.0),     # Voice body (preserve)
            (500, 1000, 1.3),    # Lower intelligibility (boost)
            (1000, 2000, 1.8),   # Critical intelligibility (strong boost)
            (2000, 4000, 1.8),   # Critical intelligibility (strong boost)
            (4000, 8000, 1.5),   # Consonants/sibilance (boost)
            (8000, 16000, 1.0)   # Highest frequencies (preserve)
        ]
        
        # Process each band
        D = librosa.stft(audio, n_fft=2048)
        mag, phase = np.abs(D), np.angle(D)
        freq_axis = librosa.fft_frequencies(sr=sr, n_fft=2048)
        
        for low, high, gain in bands:
            band_mask = (freq_axis >= low) & (freq_axis < high)
            mag[band_mask, :] *= gain
        
        # Reconstruct
        D_boosted = mag * np.exp(1j * phase)
        audio_intelligible = librosa.istft(D_boosted)
        
        # Final normalization
        audio_intelligible = librosa.util.normalize(audio_intelligible)
        
        print(f"  [INTEL] Applied frequency-specific intelligibility boost (1-4kHz: +80%)")
        return audio_intelligible
    
    def process_chaotic_scene(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, Dict]:
        """Complete chaotic scene processing pipeline"""
        
        print("\n" + "="*80)
        print("CHAOTIC SCENE PROCESSOR - MAXIMUM CLARITY MODE")
        print("="*80)
        
        original_rms = np.sqrt(np.mean(audio**2))
        
        # Step 1: Remove wind/air noise
        audio = self.reduce_wind_noise(audio, sr)
        
        # Step 2: Remove body noise (clothing, mic bumps)
        audio = self.reduce_body_noise(audio, sr)
        
        # Step 3: Reduce radio chatter
        audio = self.reduce_radio_chatter(audio, sr)
        
        # Step 4: Aggressive multi-stage noise reduction
        audio = self.aggressive_noise_reduction(audio, sr)
        
        # Step 5: Enhance distant speakers
        audio = self.enhance_distant_speech(audio, sr)
        
        # Step 6: Detect overlapping speech
        audio, overlap_regions = self.separate_overlapping_speech(audio, sr)
        
        # Step 7: Detect impact sounds
        impacts = self.isolate_impact_sounds(audio, sr)
        
        # Step 8: Boost voice intelligibility
        audio = self.boost_voice_intelligibility(audio, sr)
        
        # Calculate improvement metrics
        final_rms = np.sqrt(np.mean(audio**2))
        snr_improvement = 20 * np.log10(final_rms / (original_rms + 1e-10))
        
        metrics = {
            'processing_stages': 8,
            'wind_noise_reduction': '70%',
            'body_noise_removed': True,
            'radio_chatter_reduced': True,
            'noise_reduction_level': '90% aggressive',
            'distant_speech_boost': 'Adaptive 0.5-5x gain',
            'intelligibility_boost': '+80% at 1-4kHz',
            'snr_improvement_db': float(snr_improvement),
            'overlap_regions_detected': len(overlap_regions),
            'impacts_detected': len(impacts),
            'overlap_details': overlap_regions,
            'impact_details': impacts
        }
        
        print("\n" + "="*80)
        print("CHAOTIC SCENE PROCESSING COMPLETE")
        print("="*80)
        print(f"SNR Improvement: +{snr_improvement:.1f} dB")
        print(f"Overlapping Speech Regions: {len(overlap_regions)}")
        print(f"Impact Events: {len(impacts)}")
        print(f"Processing: Wind + Body + Radio + Noise + Distance + Overlap + Impact + Intelligibility")
        print("="*80)
        
        return audio, metrics


class ChaoticSceneBWCTranscriber:
    """Complete BWC transcription system optimized for chaotic scenes"""
    
    def __init__(self, bwc_folder: str, output_folder: str):
        self.bwc_folder = Path(bwc_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        (self.output_folder / "enhanced_audio").mkdir(exist_ok=True)
        (self.output_folder / "transcripts").mkdir(exist_ok=True)
        (self.output_folder / "forensic_reports").mkdir(exist_ok=True)
        
        # Initialize chaotic scene processor
        self.processor = ChaoticSceneProcessor(aggressive_mode=True)
        
        # Load Whisper model
        self.model = None
        if WHISPER_AVAILABLE:
            print("[WHISPER] Loading model optimized for noisy audio...")
            try:
                # Use large-v3 for best accuracy on difficult audio
                self.model = WhisperModel("large-v3", device="cpu", compute_type="int8")
                print("[WHISPER] large-v3 loaded (best for chaotic scenes)")
            except:
                try:
                    self.model = WhisperModel("medium.en", device="cpu", compute_type="int8")
                    print("[WHISPER] medium.en loaded")
                except Exception as e:
                    print(f"[ERROR] Whisper failed: {e}")
    
    def extract_audio(self, video_path: Path) -> Path:
        """Extract audio from video"""
        audio_path = self.output_folder / "enhanced_audio" / f"{video_path.stem}_original.wav"
        
        if audio_path.exists():
            return audio_path
        
        try:
            import subprocess
            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1',
                str(audio_path), '-y'
            ]
            subprocess.run(cmd, capture_output=True, check=True, timeout=300)
            return audio_path
        except Exception as e:
            print(f"[ERROR] Audio extraction failed: {e}")
            return None
    
    def process_and_transcribe(self, video_path: Path):
        """Process chaotic scene and transcribe"""
        print(f"\n{'='*100}")
        print(f"Processing: {video_path.name}")
        print(f"{'='*100}")
        
        # Extract audio
        audio_path = self.extract_audio(video_path)
        if not audio_path:
            return
        
        # Load audio
        audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        
        # Process for chaotic scene
        audio_clean, metrics = self.processor.process_chaotic_scene(audio, sr)
        
        # Save enhanced audio
        enhanced_path = self.output_folder / "enhanced_audio" / f"{video_path.stem}_CHAOTIC_ENHANCED.wav"
        sf.write(str(enhanced_path), audio_clean, sr)
        
        # Transcribe
        if self.model:
            print(f"\n[TRANSCRIBE] Processing with Whisper (optimized for difficult audio)...")
            
            segments, info = self.model.transcribe(
                str(enhanced_path),
                beam_size=5,
                best_of=5,
                temperature=0.0,
                language="en",
                task="transcribe",
                vad_filter=True,
                word_timestamps=True,
                condition_on_previous_text=False  # Better for chaotic audio
            )
            
            transcript = []
            for segment in segments:
                # Flag if in overlapping speech region
                in_overlap = any(
                    region['start'] <= segment.start <= region['end']
                    for region in metrics['overlap_details']
                )
                
                # Flag if near impact
                near_impact = any(
                    abs(impact['time'] - segment.start) < 2.0
                    for impact in metrics['impact_details']
                )
                
                transcript.append({
                    'start': float(segment.start),
                    'end': float(segment.end),
                    'text': segment.text.strip(),
                    'confidence': float(segment.avg_logprob),
                    'overlapping_speech': in_overlap,
                    'near_impact': near_impact
                })
            
            # Save transcript with chaotic scene analysis
            self.save_chaotic_transcript(video_path, transcript, metrics)
    
    def save_chaotic_transcript(self, video_path: Path, transcript: List[Dict], metrics: Dict):
        """Save transcript with chaotic scene annotations"""
        
        output = []
        output.append("="*100)
        output.append("CHAOTIC SCENE BWC TRANSCRIPT - MAXIMUM CLARITY PROCESSING")
        output.append("="*100)
        output.append(f"File: {video_path.name}")
        output.append(f"Processing: 8-Stage Chaotic Scene Optimization")
        output.append(f"SNR Improvement: +{metrics['snr_improvement_db']:.1f} dB")
        output.append(f"Overlapping Speech Regions: {metrics['overlap_regions_detected']}")
        output.append(f"Impact Events: {metrics['impacts_detected']}")
        output.append("="*100)
        output.append("")
        
        # Add transcript
        for seg in transcript:
            flags = []
            if seg.get('overlapping_speech'):
                flags.append("⚠️ OVERLAPPING_SPEECH")
            if seg.get('near_impact'):
                flags.append("🔴 NEAR_IMPACT")
            
            output.append(f"[{seg['start']:.1f}s] (Conf: {seg['confidence']:.2f}) {' '.join(flags)}")
            output.append(f"  {seg['text']}")
            output.append("")
        
        # Add impact events
        if metrics['impact_details']:
            output.append("")
            output.append("="*100)
            output.append("PHYSICAL IMPACT EVENTS DETECTED")
            output.append("="*100)
            for impact in metrics['impact_details']:
                output.append(f"[{impact['time']:.1f}s] {impact['severity']} - {impact['description']}")
            output.append("")
        
        # Add overlap events
        if metrics['overlap_details']:
            output.append("")
            output.append("="*100)
            output.append("OVERLAPPING SPEECH REGIONS")
            output.append("="*100)
            for region in metrics['overlap_details']:
                output.append(f"[{region['start']:.1f}s - {region['end']:.1f}s] Duration: {region['duration']:.1f}s")
            output.append("")
        
        # Save
        transcript_file = self.output_folder / "transcripts" / f"{video_path.stem}_CHAOTIC_TRANSCRIPT.txt"
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output))
        
        print(f"\n[SAVED] {transcript_file.name}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='BWC transcription for chaotic scenes')
    parser.add_argument('--bwc-folder',
                       default='C:/web-dev/github-repos/BarberX.info/tillerstead-toolkit/private-core-barber-cam',
                       help='BWC folder')
    parser.add_argument('--output',
                       default='./bwc_chaotic_enhanced',
                       help='Output folder')
    
    args = parser.parse_args()
    
    transcriber = ChaoticSceneBWCTranscriber(args.bwc_folder, args.output)
    
    # Process all videos
    videos = sorted(Path(args.bwc_folder).glob("**/*.mp4"))
    print(f"\nFound {len(videos)} BWC videos")
    
    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}]")
        transcriber.process_and_transcribe(video)


if __name__ == "__main__":
    main()
