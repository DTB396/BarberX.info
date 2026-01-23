"""
Local AI Tools Setup - Download and Configure Open-Source AI Models
====================================================================

Downloads and configures open-source AI tools for local processing:
- OpenAI Whisper (audio transcription)
- pyannote.audio (speaker diarization)
- Tesseract OCR (document text extraction)
- Real-ESRGAN (image/video super-resolution)
- YOLO (object/scene detection)
- spaCy (entity extraction)
- sentence-transformers (semantic search)

NO CLOUD APIs - ALL LOCAL PROCESSING
100% Open Source - Court-Defensible - Free

Author: BarberX Legal Case Management
Version: 1.0.0
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import List, Dict, Optional
import urllib.request
import zipfile
import tarfile
import shutil


class LocalAISetup:
    """Setup and configure local open-source AI tools"""
    
    def __init__(self, install_dir: str = "./ai_models"):
        self.install_dir = Path(install_dir)
        self.install_dir.mkdir(parents=True, exist_ok=True)
        
        self.whisper_dir = self.install_dir / "whisper"
        self.pyannote_dir = self.install_dir / "pyannote"
        self.tesseract_dir = self.install_dir / "tesseract"
        self.realesrgan_dir = self.install_dir / "realesrgan"
        self.yolo_dir = self.install_dir / "yolo"
        self.spacy_dir = self.install_dir / "spacy"
        self.sentence_transformers_dir = self.install_dir / "sentence_transformers"
        
        self.system = platform.system()
        self.python_exe = sys.executable
    
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> bool:
        """Run shell command and return success status"""
        try:
            print(f"  Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"  ✅ Success")
            return True
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Error: {e.stderr}")
            return False
    
    def install_python_package(self, package: str, version: Optional[str] = None) -> bool:
        """Install Python package via pip"""
        if version:
            package_spec = f"{package}=={version}"
        else:
            package_spec = package
        
        cmd = [self.python_exe, "-m", "pip", "install", package_spec]
        return self.run_command(cmd)
    
    def download_file(self, url: str, destination: Path) -> bool:
        """Download file from URL"""
        try:
            print(f"  Downloading: {url}")
            urllib.request.urlretrieve(url, destination)
            print(f"  ✅ Downloaded to {destination}")
            return True
        except Exception as e:
            print(f"  ❌ Download failed: {e}")
            return False
    
    def setup_whisper(self) -> bool:
        """
        Install OpenAI Whisper for audio transcription
        
        Models:
        - tiny: Fastest, lowest accuracy (~1GB RAM)
        - base: Fast, good accuracy (~1GB RAM)
        - small: Balanced (~2GB RAM) ⭐ RECOMMENDED
        - medium: High accuracy (~5GB RAM)
        - large: Best accuracy (~10GB RAM)
        """
        print("\n[WHISPER] Installing OpenAI Whisper (Audio Transcription)...")
        
        # Install whisper package
        if not self.install_python_package("openai-whisper"):
            return False
        
        # Install dependencies
        deps = ["torch", "torchaudio", "ffmpeg-python", "numpy", "soundfile"]
        for dep in deps:
            self.install_python_package(dep)
        
        # Download recommended model (small)
        print("  Downloading Whisper 'small' model (optimal balance)...")
        try:
            import whisper
            model = whisper.load_model("small", download_root=str(self.whisper_dir))
            print("  ✅ Whisper 'small' model ready")
            
            # Optionally download base model for faster processing
            print("  Downloading Whisper 'base' model (faster processing)...")
            model_base = whisper.load_model("base", download_root=str(self.whisper_dir))
            print("  ✅ Whisper 'base' model ready")
            
            return True
        except Exception as e:
            print(f"  ⚠️  Model download will happen on first use: {e}")
            return True
    
    def setup_pyannote(self) -> bool:
        """
        Install pyannote.audio for speaker diarization
        
        Requires Hugging Face account (free):
        1. Create account at https://huggingface.co
        2. Accept model license at https://huggingface.co/pyannote/speaker-diarization
        3. Create access token at https://huggingface.co/settings/tokens
        4. Set env var: HUGGINGFACE_TOKEN=your_token
        """
        print("\n[PYANNOTE] Installing pyannote.audio (Speaker Diarization)...")
        
        # Install pyannote
        if not self.install_python_package("pyannote.audio"):
            return False
        
        # Install dependencies
        deps = ["torch", "torchaudio", "onnxruntime"]
        for dep in deps:
            self.install_python_package(dep)
        
        print("""
  ℹ️  To use pyannote.audio, you need a Hugging Face token:
     1. Create free account: https://huggingface.co
     2. Accept license: https://huggingface.co/pyannote/speaker-diarization
     3. Get token: https://huggingface.co/settings/tokens
     4. Set environment variable: HUGGINGFACE_TOKEN=your_token
        """)
        
        return True
    
    def setup_tesseract(self) -> bool:
        """
        Install Tesseract OCR for document text extraction
        
        Note: Tesseract is a system binary, not a Python package
        """
        print("\n[TESSERACT] Installing Tesseract OCR (Document Text Extraction)...")
        
        # Install Python wrapper
        if not self.install_python_package("pytesseract"):
            return False
        
        # Install Pillow for image processing
        self.install_python_package("Pillow")
        self.install_python_package("pdf2image")
        
        if self.system == "Windows":
            print("""
  ℹ️  Tesseract OCR Installation (Windows):
     1. Download installer: https://github.com/UB-Mannheim/tesseract/wiki
     2. Run installer (default path: C:\\Program Files\\Tesseract-OCR)
     3. Add to PATH or set: TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe
            """)
        elif self.system == "Linux":
            print("  Installing Tesseract via apt...")
            self.run_command(["sudo", "apt-get", "update"])
            self.run_command(["sudo", "apt-get", "install", "-y", "tesseract-ocr"])
        elif self.system == "Darwin":  # macOS
            print("  Installing Tesseract via Homebrew...")
            self.run_command(["brew", "install", "tesseract"])
        
        return True
    
    def setup_realesrgan(self) -> bool:
        """
        Install Real-ESRGAN for AI super-resolution (images/video)
        
        Models:
        - RealESRGAN_x4plus: General 4x upscaling
        - RealESRGAN_x4plus_anime: Anime/cartoon 4x
        - RealESRNet_x4plus: For real photos
        """
        print("\n[REALESRGAN] Installing Real-ESRGAN (AI Super-Resolution)...")
        
        # Install realesrgan
        if not self.install_python_package("realesrgan"):
            return False
        
        # Install dependencies
        deps = ["torch", "torchvision", "opencv-python", "numpy", "Pillow"]
        for dep in deps:
            self.install_python_package(dep)
        
        # Download pre-trained models
        self.realesrgan_dir.mkdir(exist_ok=True)
        
        models = {
            "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "RealESRNet_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
        }
        
        print("  Downloading Real-ESRGAN models...")
        for model_name, url in models.items():
            model_path = self.realesrgan_dir / model_name
            if not model_path.exists():
                self.download_file(url, model_path)
        
        return True
    
    def setup_yolo(self) -> bool:
        """
        Install YOLOv8 for object/scene detection
        
        Models:
        - yolov8n: Nano (fastest)
        - yolov8s: Small
        - yolov8m: Medium ⭐ RECOMMENDED
        - yolov8l: Large
        - yolov8x: Extra Large (most accurate)
        """
        print("\n[YOLO] Installing YOLOv8 (Object/Scene Detection)...")
        
        # Install ultralytics (YOLOv8)
        if not self.install_python_package("ultralytics"):
            return False
        
        # Install dependencies
        deps = ["torch", "torchvision", "opencv-python", "numpy", "Pillow"]
        for dep in deps:
            self.install_python_package(dep)
        
        # Download YOLO models
        print("  Downloading YOLOv8 models...")
        try:
            from ultralytics import YOLO
            
            # Download medium model (balanced)
            model = YOLO("yolov8m.pt")
            print("  ✅ YOLOv8 medium model ready")
            
            # Optionally download nano for faster processing
            model_nano = YOLO("yolov8n.pt")
            print("  ✅ YOLOv8 nano model ready")
            
            return True
        except Exception as e:
            print(f"  ⚠️  Model download will happen on first use: {e}")
            return True
    
    def setup_spacy(self) -> bool:
        """
        Install spaCy for entity extraction (names, locations, organizations)
        
        Models:
        - en_core_web_sm: Small, fast
        - en_core_web_md: Medium, balanced ⭐ RECOMMENDED
        - en_core_web_lg: Large, most accurate
        """
        print("\n[SPACY] Installing spaCy (Entity Extraction)...")
        
        # Install spacy
        if not self.install_python_package("spacy"):
            return False
        
        # Download medium English model
        print("  Downloading spaCy 'en_core_web_md' model...")
        cmd = [self.python_exe, "-m", "spacy", "download", "en_core_web_md"]
        self.run_command(cmd)
        
        return True
    
    def setup_sentence_transformers(self) -> bool:
        """
        Install sentence-transformers for semantic search
        
        Models:
        - all-MiniLM-L6-v2: Fast, lightweight ⭐ RECOMMENDED
        - all-mpnet-base-v2: Higher quality, slower
        """
        print("\n[SENTENCE-TRANSFORMERS] Installing sentence-transformers (Semantic Search)...")
        
        # Install sentence-transformers
        if not self.install_python_package("sentence-transformers"):
            return False
        
        # Install dependencies
        deps = ["torch", "transformers", "numpy", "scikit-learn"]
        for dep in deps:
            self.install_python_package(dep)
        
        # Download model
        print("  Downloading 'all-MiniLM-L6-v2' model...")
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            print("  ✅ Sentence transformer model ready")
            return True
        except Exception as e:
            print(f"  ⚠️  Model download will happen on first use: {e}")
            return True
    
    def setup_audio_processing(self) -> bool:
        """Install audio processing libraries"""
        print("\n[AUDIO] Installing Audio Processing Libraries...")
        
        libs = [
            "librosa",      # Audio analysis
            "noisereduce",  # Noise reduction
            "soundfile",    # Audio I/O
            "pydub",        # Audio manipulation
            "scipy",        # Signal processing
            "audioread"     # Audio file reading
        ]
        
        for lib in libs:
            self.install_python_package(lib)
        
        return True
    
    def setup_video_processing(self) -> bool:
        """Install video processing libraries"""
        print("\n[VIDEO] Installing Video Processing Libraries...")
        
        libs = [
            "opencv-python",  # Video processing
            "moviepy",        # Video editing
            "imageio",        # Image/video I/O
            "imageio-ffmpeg"  # FFmpeg backend
        ]
        
        for lib in libs:
            self.install_python_package(lib)
        
        return True
    
    def setup_document_processing(self) -> bool:
        """Install document processing libraries"""
        print("\n[DOCS] Installing Document Processing Libraries...")
        
        libs = [
            "PyPDF2",       # PDF reading
            "pikepdf",      # PDF manipulation
            "python-docx",  # Word documents
            "openpyxl",     # Excel files
            "pandas",       # Data processing
            "tabula-py"     # PDF table extraction
        ]
        
        for lib in libs:
            self.install_python_package(lib)
        
        return True
    
    def verify_installation(self) -> Dict[str, bool]:
        """Verify all tools are properly installed"""
        print("\n[VERIFY] Verifying Installation...\n")
        
        results = {}
        
        # Test Whisper
        try:
            import whisper
            results['whisper'] = True
            print("  ✅ Whisper: OK")
        except ImportError:
            results['whisper'] = False
            print("  ❌ Whisper: FAILED")
        
        # Test pyannote
        try:
            import pyannote.audio
            results['pyannote'] = True
            print("  ✅ pyannote.audio: OK")
        except ImportError:
            results['pyannote'] = False
            print("  ❌ pyannote.audio: FAILED")
        
        # Test Tesseract
        try:
            import pytesseract
            results['tesseract'] = True
            print("  ✅ Tesseract: OK")
        except ImportError:
            results['tesseract'] = False
            print("  ❌ Tesseract: FAILED")
        
        # Test Real-ESRGAN
        try:
            import realesrgan
            results['realesrgan'] = True
            print("  ✅ Real-ESRGAN: OK")
        except ImportError:
            results['realesrgan'] = False
            print("  ❌ Real-ESRGAN: FAILED")
        
        # Test YOLO
        try:
            from ultralytics import YOLO
            results['yolo'] = True
            print("  ✅ YOLO: OK")
        except ImportError:
            results['yolo'] = False
            print("  ❌ YOLO: FAILED")
        
        # Test spaCy
        try:
            import spacy
            results['spacy'] = True
            print("  ✅ spaCy: OK")
        except ImportError:
            results['spacy'] = False
            print("  ❌ spaCy: FAILED")
        
        # Test sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            results['sentence_transformers'] = True
            print("  ✅ sentence-transformers: OK")
        except ImportError:
            results['sentence_transformers'] = False
            print("  ❌ sentence-transformers: FAILED")
        
        # Test audio libraries
        try:
            import librosa
            import noisereduce
            results['audio'] = True
            print("  ✅ Audio libraries: OK")
        except ImportError:
            results['audio'] = False
            print("  ❌ Audio libraries: FAILED")
        
        # Test video libraries
        try:
            import cv2
            import moviepy.editor
            results['video'] = True
            print("  ✅ Video libraries: OK")
        except ImportError:
            results['video'] = False
            print("  ❌ Video libraries: FAILED")
        
        return results
    
    def setup_all(self) -> bool:
        """Setup all AI tools"""
        print("=" * 80)
        print("LOCAL AI TOOLS SETUP - Open Source Evidence Processing")
        print("=" * 80)
        
        print(f"\n📍 Installation Directory: {self.install_dir.absolute()}")
        print(f"🐍 Python: {self.python_exe}")
        print(f"💻 System: {self.system}")
        
        # Setup each component
        self.setup_whisper()
        self.setup_pyannote()
        self.setup_tesseract()
        self.setup_realesrgan()
        self.setup_yolo()
        self.setup_spacy()
        self.setup_sentence_transformers()
        self.setup_audio_processing()
        self.setup_video_processing()
        self.setup_document_processing()
        
        # Verify installation
        results = self.verify_installation()
        
        # Summary
        success_count = sum(results.values())
        total_count = len(results)
        
        print("\n" + "=" * 80)
        print(f"✅ Setup Complete: {success_count}/{total_count} components installed")
        print("=" * 80)
        
        if success_count == total_count:
            print("\nSUCCESS: ALL TOOLS READY FOR LOCAL AI PROCESSING!")
            print("\n📖 Next Steps:")
            print("  1. Set environment variables (if needed):")
            print("     - HUGGINGFACE_TOKEN=your_token (for pyannote)")
            print("     - TESSERACT_CMD=path_to_tesseract (if not in PATH)")
            print("  2. Test the services with sample files")
            print("  3. Review docs/LOCAL-AI-GUIDE.md for usage examples")
        else:
            print("\n⚠️  Some components failed. Review errors above.")
        
        return success_count == total_count


def main():
    """Main setup script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup local open-source AI tools")
    parser.add_argument(
        "--install-dir",
        type=str,
        default="./ai_models",
        help="Directory to install AI models (default: ./ai_models)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing installation"
    )
    
    args = parser.parse_args()
    
    setup = LocalAISetup(install_dir=args.install_dir)
    
    if args.verify_only:
        setup.verify_installation()
    else:
        setup.setup_all()


if __name__ == "__main__":
    main()
