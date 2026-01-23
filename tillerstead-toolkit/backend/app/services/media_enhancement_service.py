"""
Media Enhancement Service - Non-Destructive Audio/Video Enhancement
====================================================================

Advanced media processing with:
- Audio isolation and enhancement (noise reduction, normalization, speaker isolation)
- Video enhancement (upscaling, stabilization, sharpening, denoising)
- Image enhancement (super-resolution, contrast, clarity)
- Non-destructive workflow (originals always preserved in vault)
- Enhancement metadata tracking (settings, versions, quality metrics)
- Court-defensible processing logs

Author: BarberX Legal Case Management
Version: 1.0.0
"""

import asyncio
import json
import os
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

# Would require for production:
# pip install librosa noisereduce soundfile pydub opencv-python pillow
# pip install tensorflow torch (for AI upscaling models)


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class EnhancementType(str, Enum):
    """Types of media enhancement"""
    AUDIO_NOISE_REDUCTION = "Audio Noise Reduction"
    AUDIO_NORMALIZATION = "Audio Normalization"
    AUDIO_SPEAKER_ISOLATION = "Audio Speaker Isolation"
    AUDIO_VOICE_ENHANCEMENT = "Audio Voice Enhancement"
    VIDEO_UPSCALING = "Video Upscaling"
    VIDEO_STABILIZATION = "Video Stabilization"
    VIDEO_DENOISING = "Video Denoising"
    VIDEO_SHARPENING = "Video Sharpening"
    VIDEO_CONTRAST_ENHANCEMENT = "Video Contrast Enhancement"
    IMAGE_SUPER_RESOLUTION = "Image Super-Resolution"
    IMAGE_CLARITY_ENHANCEMENT = "Image Clarity Enhancement"
    IMAGE_CONTRAST_ADJUSTMENT = "Image Contrast Adjustment"


class EnhancementQuality(str, Enum):
    """Enhancement quality/aggressiveness levels"""
    MINIMAL = "Minimal (Preserve Maximum Authenticity)"
    MODERATE = "Moderate (Balance Enhancement/Authenticity)"
    AGGRESSIVE = "Aggressive (Maximum Enhancement)"


class AudioNoiseProfile(str, Enum):
    """Common noise profiles for BWC/audio evidence"""
    TRAFFIC_NOISE = "Traffic/Road Noise"
    WIND_NOISE = "Wind Noise"
    RADIO_STATIC = "Radio Static/Interference"
    BACKGROUND_CROWD = "Background Crowd/Multiple Speakers"
    ENGINE_HUM = "Vehicle Engine Hum"
    HVAC_NOISE = "HVAC/Air Conditioning"
    GENERAL = "General Noise Reduction"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class EnhancementSettings:
    """Settings used for enhancement processing"""
    enhancement_type: EnhancementType
    quality_level: EnhancementQuality
    
    # Audio settings
    noise_reduction_db: Optional[float] = None  # dB reduction
    normalization_target_db: Optional[float] = None  # Target loudness
    noise_profile: Optional[AudioNoiseProfile] = None
    preserve_frequency_range: Optional[Tuple[int, int]] = None  # Hz range to preserve
    
    # Video/Image settings
    target_resolution: Optional[str] = None  # e.g., "1920x1080", "3840x2160"
    upscaling_algorithm: Optional[str] = None  # "bicubic", "lanczos", "AI_ESRGAN"
    stabilization_strength: Optional[float] = None  # 0.0 to 1.0
    sharpening_amount: Optional[float] = None  # 0.0 to 1.0
    contrast_adjustment: Optional[float] = None  # -1.0 to 1.0
    
    # Processing metadata
    processing_date: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    processing_software: str = "BarberX Media Enhancer v1.0"
    processing_notes: str = ""


@dataclass
class QualityMetrics:
    """Quality assessment metrics for enhanced media"""
    # Audio metrics
    signal_to_noise_ratio_db: Optional[float] = None
    dynamic_range_db: Optional[float] = None
    clarity_score: Optional[float] = None  # 0.0 to 1.0
    
    # Video/Image metrics
    sharpness_score: Optional[float] = None  # 0.0 to 1.0
    contrast_ratio: Optional[float] = None
    brightness_average: Optional[float] = None
    resolution: Optional[str] = None
    
    # Comparison to original
    enhancement_improvement_score: Optional[float] = None  # 0.0 to 1.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EnhancementResult:
    """Complete enhancement processing result"""
    enhancement_id: str
    source_evidence_id: str
    source_file: str
    enhanced_file: str
    enhancement_type: EnhancementType
    settings: EnhancementSettings
    
    # Quality metrics
    original_metrics: QualityMetrics
    enhanced_metrics: QualityMetrics
    
    # Metadata
    original_hash: str  # SHA-256 of original (vault file)
    enhanced_hash: str  # SHA-256 of enhanced version
    processing_date: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    processing_duration_seconds: float = 0.0
    court_admissible: bool = True  # Flag if enhancement is court-acceptable
    enhancement_notes: str = ""
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['enhancement_type'] = self.enhancement_type.value
        data['settings']['enhancement_type'] = self.settings.enhancement_type.value
        data['settings']['quality_level'] = self.settings.quality_level.value
        if self.settings.noise_profile:
            data['settings']['noise_profile'] = self.settings.noise_profile.value
        return data


# ============================================================================
# MEDIA ENHANCEMENT SERVICE
# ============================================================================

class MediaEnhancementService:
    """
    Non-destructive media enhancement with court-defensible processing
    
    CRITICAL: Original files are NEVER modified. All enhancements create new versions.
    """
    
    def __init__(self, output_dir: str = "./enhanced_media"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.audio_enhanced_dir = self.output_dir / "audio_enhanced"
        self.video_enhanced_dir = self.output_dir / "video_enhanced"
        self.image_enhanced_dir = self.output_dir / "image_enhanced"
        self.enhancement_metadata_dir = self.output_dir / "enhancement_metadata"
        
        for directory in [self.audio_enhanced_dir, self.video_enhanced_dir, 
                          self.image_enhanced_dir, self.enhancement_metadata_dir]:
            directory.mkdir(exist_ok=True)
    
    async def enhance_audio(
        self,
        source_file: Path,
        evidence_id: str,
        quality_level: EnhancementQuality = EnhancementQuality.MODERATE,
        noise_profile: AudioNoiseProfile = AudioNoiseProfile.GENERAL,
        apply_noise_reduction: bool = True,
        apply_normalization: bool = True,
        apply_voice_enhancement: bool = True,
        isolate_speaker: Optional[str] = None  # "SPEAKER_01", etc.
    ) -> EnhancementResult:
        """
        Enhance audio quality with noise reduction and voice clarity
        
        Non-destructive: Original file is never modified
        
        Args:
            source_file: Path to original audio/video file
            evidence_id: Evidence ID from vault
            quality_level: How aggressive to enhance
            noise_profile: Type of noise to reduce
            apply_noise_reduction: Enable noise reduction
            apply_normalization: Normalize volume levels
            apply_voice_enhancement: Enhance voice frequencies
            isolate_speaker: Extract/enhance specific speaker
        
        Returns:
            EnhancementResult with enhanced file path and metrics
        """
        import hashlib
        
        start_time = datetime.utcnow()
        enhancement_id = f"ENH-AUD-{evidence_id}-{start_time.strftime('%Y%m%d%H%M%S')}"
        
        # Compute original hash
        original_hash = self._compute_file_hash(source_file)
        
        # Configure settings
        settings = EnhancementSettings(
            enhancement_type=EnhancementType.AUDIO_NOISE_REDUCTION,
            quality_level=quality_level,
            noise_profile=noise_profile,
            noise_reduction_db=self._get_noise_reduction_level(quality_level),
            normalization_target_db=-20.0,  # Target -20 dBFS
            preserve_frequency_range=(85, 8000),  # Preserve voice range
            processing_notes=f"Profile: {noise_profile.value}, Quality: {quality_level.value}"
        )
        
        # Measure original quality
        original_metrics = await self._analyze_audio_quality(source_file)
        
        # Process audio
        enhanced_file = await self._process_audio_enhancement(
            source_file,
            enhancement_id,
            settings,
            apply_noise_reduction,
            apply_normalization,
            apply_voice_enhancement,
            isolate_speaker
        )
        
        # Measure enhanced quality
        enhanced_metrics = await self._analyze_audio_quality(enhanced_file)
        
        # Compute enhanced hash
        enhanced_hash = self._compute_file_hash(enhanced_file)
        
        # Calculate processing duration
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Build result
        result = EnhancementResult(
            enhancement_id=enhancement_id,
            source_evidence_id=evidence_id,
            source_file=str(source_file),
            enhanced_file=str(enhanced_file),
            enhancement_type=EnhancementType.AUDIO_NOISE_REDUCTION,
            settings=settings,
            original_metrics=original_metrics,
            enhanced_metrics=enhanced_metrics,
            original_hash=original_hash,
            enhanced_hash=enhanced_hash,
            processing_duration_seconds=duration,
            court_admissible=True,  # Moderate enhancement is court-acceptable
            enhancement_notes=f"Audio enhanced using {quality_level.value} settings"
        )
        
        # Save metadata
        await self._save_enhancement_metadata(result)
        
        return result
    
    async def enhance_video(
        self,
        source_file: Path,
        evidence_id: str,
        quality_level: EnhancementQuality = EnhancementQuality.MODERATE,
        apply_upscaling: bool = True,
        apply_stabilization: bool = True,
        apply_sharpening: bool = True,
        apply_denoising: bool = True,
        target_resolution: Optional[str] = "1920x1080"
    ) -> EnhancementResult:
        """
        Enhance video quality (upscaling, stabilization, sharpening)
        
        Non-destructive: Original file is never modified
        """
        import hashlib
        
        start_time = datetime.utcnow()
        enhancement_id = f"ENH-VID-{evidence_id}-{start_time.strftime('%Y%m%d%H%M%S')}"
        
        # Compute original hash
        original_hash = self._compute_file_hash(source_file)
        
        # Configure settings
        settings = EnhancementSettings(
            enhancement_type=EnhancementType.VIDEO_UPSCALING,
            quality_level=quality_level,
            target_resolution=target_resolution,
            upscaling_algorithm="lanczos",  # High-quality, court-defensible
            stabilization_strength=0.5 if quality_level == EnhancementQuality.MODERATE else 0.7,
            sharpening_amount=0.3 if quality_level == EnhancementQuality.MODERATE else 0.5,
            processing_notes=f"Video enhancement: {quality_level.value}"
        )
        
        # Measure original quality
        original_metrics = await self._analyze_video_quality(source_file)
        
        # Process video
        enhanced_file = await self._process_video_enhancement(
            source_file,
            enhancement_id,
            settings,
            apply_upscaling,
            apply_stabilization,
            apply_sharpening,
            apply_denoising
        )
        
        # Measure enhanced quality
        enhanced_metrics = await self._analyze_video_quality(enhanced_file)
        
        # Compute enhanced hash
        enhanced_hash = self._compute_file_hash(enhanced_file)
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        result = EnhancementResult(
            enhancement_id=enhancement_id,
            source_evidence_id=evidence_id,
            source_file=str(source_file),
            enhanced_file=str(enhanced_file),
            enhancement_type=EnhancementType.VIDEO_UPSCALING,
            settings=settings,
            original_metrics=original_metrics,
            enhanced_metrics=enhanced_metrics,
            original_hash=original_hash,
            enhanced_hash=enhanced_hash,
            processing_duration_seconds=duration,
            court_admissible=quality_level != EnhancementQuality.AGGRESSIVE,
            enhancement_notes=f"Video enhanced to {target_resolution}"
        )
        
        await self._save_enhancement_metadata(result)
        
        return result
    
    async def enhance_image(
        self,
        source_file: Path,
        evidence_id: str,
        quality_level: EnhancementQuality = EnhancementQuality.MODERATE,
        apply_super_resolution: bool = True,
        apply_clarity: bool = True,
        apply_contrast: bool = True,
        target_scale: float = 2.0  # 2x upscale
    ) -> EnhancementResult:
        """
        Enhance image quality (super-resolution, clarity, contrast)
        
        Non-destructive: Original file is never modified
        """
        import hashlib
        
        start_time = datetime.utcnow()
        enhancement_id = f"ENH-IMG-{evidence_id}-{start_time.strftime('%Y%m%d%H%M%S')}"
        
        original_hash = self._compute_file_hash(source_file)
        
        settings = EnhancementSettings(
            enhancement_type=EnhancementType.IMAGE_SUPER_RESOLUTION,
            quality_level=quality_level,
            upscaling_algorithm="lanczos" if quality_level != EnhancementQuality.AGGRESSIVE else "AI_ESRGAN",
            sharpening_amount=0.3,
            contrast_adjustment=0.1,
            processing_notes=f"Image enhancement: {target_scale}x upscale"
        )
        
        original_metrics = await self._analyze_image_quality(source_file)
        
        enhanced_file = await self._process_image_enhancement(
            source_file,
            enhancement_id,
            settings,
            apply_super_resolution,
            apply_clarity,
            apply_contrast,
            target_scale
        )
        
        enhanced_metrics = await self._analyze_image_quality(enhanced_file)
        enhanced_hash = self._compute_file_hash(enhanced_file)
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        result = EnhancementResult(
            enhancement_id=enhancement_id,
            source_evidence_id=evidence_id,
            source_file=str(source_file),
            enhanced_file=str(enhanced_file),
            enhancement_type=EnhancementType.IMAGE_SUPER_RESOLUTION,
            settings=settings,
            original_metrics=original_metrics,
            enhanced_metrics=enhanced_metrics,
            original_hash=original_hash,
            enhanced_hash=enhanced_hash,
            processing_duration_seconds=duration,
            court_admissible=quality_level != EnhancementQuality.AGGRESSIVE,
            enhancement_notes=f"Image enhanced {target_scale}x"
        )
        
        await self._save_enhancement_metadata(result)
        
        return result
    
    def _get_noise_reduction_level(self, quality: EnhancementQuality) -> float:
        """Get noise reduction amount based on quality level"""
        levels = {
            EnhancementQuality.MINIMAL: 6.0,      # 6 dB reduction (subtle)
            EnhancementQuality.MODERATE: 12.0,    # 12 dB reduction (balanced)
            EnhancementQuality.AGGRESSIVE: 20.0   # 20 dB reduction (aggressive)
        }
        return levels.get(quality, 12.0)
    
    async def _process_audio_enhancement(
        self,
        source_file: Path,
        enhancement_id: str,
        settings: EnhancementSettings,
        apply_noise_reduction: bool,
        apply_normalization: bool,
        apply_voice_enhancement: bool,
        isolate_speaker: Optional[str]
    ) -> Path:
        """
        Process audio enhancement using ffmpeg and audio processing libraries
        
        Production would use:
        - librosa for spectral analysis
        - noisereduce for noise reduction
        - pydub for audio manipulation
        - ffmpeg for format conversion
        """
        output_file = self.audio_enhanced_dir / f"{enhancement_id}.wav"
        
        # Simulated enhancement pipeline
        # In production, would use actual audio processing:
        
        # Step 1: Extract audio from video if needed
        # ffmpeg -i input.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 output.wav
        
        # Step 2: Apply noise reduction
        if apply_noise_reduction:
            # librosa/noisereduce:
            # y, sr = librosa.load(audio_file)
            # y_reduced = noisereduce.reduce_noise(y=y, sr=sr, stationary=True)
            pass
        
        # Step 3: Apply normalization
        if apply_normalization:
            # ffmpeg -i input.wav -af loudnorm=I=-20:TP=-1.5:LRA=11 output.wav
            pass
        
        # Step 4: Voice enhancement (boost speech frequencies)
        if apply_voice_enhancement:
            # ffmpeg -i input.wav -af "equalizer=f=3000:width_type=h:width=2000:g=6" output.wav
            # Boost 1-4 kHz range (human voice fundamental + harmonics)
            pass
        
        # Step 5: Speaker isolation (if requested)
        if isolate_speaker:
            # Use diarization results to extract specific speaker's segments
            # and suppress other speakers
            pass
        
        # For demo, just copy file
        if source_file.exists():
            import shutil
            shutil.copy2(source_file, output_file)
        
        return output_file
    
    async def _process_video_enhancement(
        self,
        source_file: Path,
        enhancement_id: str,
        settings: EnhancementSettings,
        apply_upscaling: bool,
        apply_stabilization: bool,
        apply_sharpening: bool,
        apply_denoising: bool
    ) -> Path:
        """
        Process video enhancement using ffmpeg and video processing libraries
        
        Production would use:
        - ffmpeg for video manipulation
        - OpenCV for stabilization and filtering
        - AI models (Real-ESRGAN, GFPGAN) for super-resolution
        """
        output_file = self.video_enhanced_dir / f"{enhancement_id}.mp4"
        
        # Simulated enhancement pipeline
        # In production:
        
        # Step 1: Upscaling
        if apply_upscaling:
            # ffmpeg -i input.mp4 -vf scale=1920:1080:flags=lanczos output.mp4
            # Or AI upscaling:
            # realesrgan-ncnn-vulkan -i input.mp4 -o output.mp4 -n realesrgan-x4plus
            pass
        
        # Step 2: Stabilization
        if apply_stabilization:
            # ffmpeg two-pass stabilization:
            # Pass 1: ffmpeg -i input.mp4 -vf vidstabdetect=shakiness=5 -f null -
            # Pass 2: ffmpeg -i input.mp4 -vf vidstabtransform=smoothing=30 output.mp4
            pass
        
        # Step 3: Sharpening
        if apply_sharpening:
            # ffmpeg -i input.mp4 -vf unsharp=5:5:1.0:5:5:0.0 output.mp4
            pass
        
        # Step 4: Denoising
        if apply_denoising:
            # ffmpeg -i input.mp4 -vf hqdn3d=4:3:6:4.5 output.mp4
            pass
        
        # For demo
        if source_file.exists():
            import shutil
            shutil.copy2(source_file, output_file)
        
        return output_file
    
    async def _process_image_enhancement(
        self,
        source_file: Path,
        enhancement_id: str,
        settings: EnhancementSettings,
        apply_super_resolution: bool,
        apply_clarity: bool,
        apply_contrast: bool,
        target_scale: float
    ) -> Path:
        """
        Process image enhancement using PIL/OpenCV and AI models
        
        Production would use:
        - PIL/Pillow for basic image processing
        - OpenCV for advanced filtering
        - Real-ESRGAN for AI super-resolution
        """
        output_file = self.image_enhanced_dir / f"{enhancement_id}.png"
        
        # In production:
        
        # Step 1: Super-resolution
        if apply_super_resolution:
            # PIL (Lanczos):
            # from PIL import Image
            # img = Image.open(source_file)
            # new_size = (int(img.width * target_scale), int(img.height * target_scale))
            # img_upscaled = img.resize(new_size, Image.LANCZOS)
            
            # Or AI (Real-ESRGAN):
            # realesrgan-ncnn-vulkan -i input.jpg -o output.png -s 4
            pass
        
        # Step 2: Clarity enhancement
        if apply_clarity:
            # OpenCV unsharp mask:
            # import cv2
            # img = cv2.imread(str(source_file))
            # gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
            # img_sharp = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
            pass
        
        # Step 3: Contrast adjustment
        if apply_contrast:
            # PIL:
            # from PIL import ImageEnhance
            # enhancer = ImageEnhance.Contrast(img)
            # img_contrast = enhancer.enhance(1.2)
            pass
        
        # For demo
        if source_file.exists():
            import shutil
            shutil.copy2(source_file, output_file)
        
        return output_file
    
    async def _analyze_audio_quality(self, audio_file: Path) -> QualityMetrics:
        """Analyze audio quality metrics"""
        # In production, would use librosa/soundfile to calculate:
        # - Signal-to-noise ratio
        # - Dynamic range
        # - Clarity score (spectral analysis)
        
        return QualityMetrics(
            signal_to_noise_ratio_db=15.5,  # Example
            dynamic_range_db=45.0,
            clarity_score=0.72
        )
    
    async def _analyze_video_quality(self, video_file: Path) -> QualityMetrics:
        """Analyze video quality metrics"""
        # In production, would use OpenCV/ffmpeg to calculate:
        # - Sharpness (Laplacian variance)
        # - Contrast ratio
        # - Average brightness
        # - Resolution
        
        return QualityMetrics(
            sharpness_score=0.68,
            contrast_ratio=1.5,
            brightness_average=128.0,
            resolution="1920x1080"
        )
    
    async def _analyze_image_quality(self, image_file: Path) -> QualityMetrics:
        """Analyze image quality metrics"""
        return QualityMetrics(
            sharpness_score=0.65,
            contrast_ratio=1.4,
            brightness_average=125.0,
            resolution="1280x720"
        )
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file"""
        import hashlib
        sha256 = hashlib.sha256()
        if file_path.exists():
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)
        return sha256.hexdigest()
    
    async def _save_enhancement_metadata(self, result: EnhancementResult):
        """Save enhancement metadata as JSON"""
        metadata_path = self.enhancement_metadata_dir / f"{result.enhancement_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    async def compare_versions(
        self,
        original_evidence_id: str,
        enhancement_id: str
    ) -> Dict:
        """
        Generate side-by-side comparison report
        
        Returns:
            Comparison metrics and recommendations
        """
        metadata_path = self.enhancement_metadata_dir / f"{enhancement_id}_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Enhancement metadata not found: {enhancement_id}")
        
        with open(metadata_path, 'r') as f:
            enhancement_data = json.load(f)
        
        original_metrics = enhancement_data['original_metrics']
        enhanced_metrics = enhancement_data['enhanced_metrics']
        
        # Calculate improvements
        improvements = {}
        if original_metrics.get('signal_to_noise_ratio_db') and enhanced_metrics.get('signal_to_noise_ratio_db'):
            snr_improvement = enhanced_metrics['signal_to_noise_ratio_db'] - original_metrics['signal_to_noise_ratio_db']
            improvements['snr_improvement_db'] = round(snr_improvement, 2)
        
        if original_metrics.get('sharpness_score') and enhanced_metrics.get('sharpness_score'):
            sharpness_improvement = enhanced_metrics['sharpness_score'] - original_metrics['sharpness_score']
            improvements['sharpness_improvement_percent'] = round(sharpness_improvement * 100, 2)
        
        return {
            'enhancement_id': enhancement_id,
            'original_evidence_id': original_evidence_id,
            'enhancement_type': enhancement_data['enhancement_type'],
            'quality_level': enhancement_data['settings']['quality_level'],
            'original_metrics': original_metrics,
            'enhanced_metrics': enhanced_metrics,
            'improvements': improvements,
            'court_admissible': enhancement_data['court_admissible'],
            'recommendation': self._generate_recommendation(improvements, enhancement_data['court_admissible'])
        }
    
    def _generate_recommendation(self, improvements: Dict, court_admissible: bool) -> str:
        """Generate recommendation for court use"""
        if not court_admissible:
            return "CAUTION: Aggressive enhancement may not be court-admissible. Use original or moderate enhancement for exhibits."
        
        if improvements.get('snr_improvement_db', 0) > 10:
            return "EXCELLENT: Significant audio clarity improvement. Recommended for court exhibits."
        elif improvements.get('sharpness_improvement_percent', 0) > 20:
            return "GOOD: Notable visual improvement. Suitable for court exhibits with disclosure of enhancement."
        else:
            return "MINIMAL: Limited improvement. Consider using original for court exhibits."


# ============================================================================
# GLOBAL SERVICE INSTANCE
# ============================================================================

media_enhancer = MediaEnhancementService()
