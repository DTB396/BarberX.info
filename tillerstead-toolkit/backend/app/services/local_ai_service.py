"""
Local AI Service Manager - Unified Interface for Open-Source AI Tools
======================================================================

Provides a unified interface to local AI processing:
- Audio transcription (Whisper)
- Speaker diarization (pyannote.audio)
- Document OCR (Tesseract)
- Image super-resolution (Real-ESRGAN)
- Object detection (YOLO)
- Entity extraction (spaCy)
- Semantic search (sentence-transformers)

NO CLOUD APIs - 100% Local - Free - Court-Defensible

Author: BarberX Legal Case Management
Version: 1.0.0
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LocalAIConfig:
    """Configuration for local AI processing"""
    # Hugging Face
    huggingface_token: Optional[str] = None
    
    # Model directories
    ai_models_dir: Path = Path("./ai_models")
    whisper_model_dir: Optional[Path] = None
    pyannote_model_dir: Optional[Path] = None
    realesrgan_model_dir: Optional[Path] = None
    yolo_model_dir: Optional[Path] = None
    
    # Whisper settings
    whisper_model_size: str = "small"
    whisper_language: str = "en"
    whisper_device: str = "cpu"
    
    # pyannote settings
    pyannote_model: str = "pyannote/speaker-diarization@2.1"
    pyannote_device: str = "cpu"
    pyannote_min_speakers: int = 2
    pyannote_max_speakers: int = 5
    
    # Tesseract settings
    tesseract_cmd: Optional[str] = None
    tesseract_lang: str = "eng"
    tesseract_oem: int = 3
    tesseract_psm: int = 3
    
    # Real-ESRGAN settings
    realesrgan_model: str = "RealESRGAN_x4plus"
    realesrgan_device: str = "cpu"
    realesrgan_tile_size: int = 512
    
    # YOLO settings
    yolo_model: str = "yolov8m"
    yolo_device: str = "cpu"
    yolo_confidence: float = 0.5
    
    # spaCy settings
    spacy_model: str = "en_core_web_md"
    
    # Sentence transformers
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    sentence_transformer_device: str = "cpu"
    
    # Processing settings
    use_gpu_if_available: bool = True
    num_cpu_threads: int = 4
    processing_batch_size: int = 8
    
    # Fallback settings
    enable_cloud_fallback: bool = False
    
    @classmethod
    def from_env(cls) -> "LocalAIConfig":
        """Load configuration from environment variables"""
        config = cls()
        
        # Hugging Face
        config.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        
        # Model directories
        if models_dir := os.getenv("AI_MODELS_DIR"):
            config.ai_models_dir = Path(models_dir)
        
        # Whisper
        config.whisper_model_size = os.getenv("WHISPER_MODEL_SIZE", "small")
        config.whisper_language = os.getenv("WHISPER_LANGUAGE", "en")
        config.whisper_device = os.getenv("WHISPER_DEVICE", "cpu")
        
        # pyannote
        config.pyannote_model = os.getenv("PYANNOTE_MODEL", "pyannote/speaker-diarization@2.1")
        config.pyannote_device = os.getenv("PYANNOTE_DEVICE", "cpu")
        config.pyannote_min_speakers = int(os.getenv("PYANNOTE_MIN_SPEAKERS", "2"))
        config.pyannote_max_speakers = int(os.getenv("PYANNOTE_MAX_SPEAKERS", "5"))
        
        # Tesseract
        config.tesseract_cmd = os.getenv("TESSERACT_CMD")
        config.tesseract_lang = os.getenv("TESSERACT_LANG", "eng")
        config.tesseract_oem = int(os.getenv("TESSERACT_OEM", "3"))
        config.tesseract_psm = int(os.getenv("TESSERACT_PSM", "3"))
        
        # Real-ESRGAN
        config.realesrgan_model = os.getenv("REALESRGAN_MODEL", "RealESRGAN_x4plus")
        config.realesrgan_device = os.getenv("REALESRGAN_DEVICE", "cpu")
        config.realesrgan_tile_size = int(os.getenv("REALESRGAN_TILE_SIZE", "512"))
        
        # YOLO
        config.yolo_model = os.getenv("YOLO_MODEL", "yolov8m")
        config.yolo_device = os.getenv("YOLO_DEVICE", "cpu")
        config.yolo_confidence = float(os.getenv("YOLO_CONFIDENCE", "0.5"))
        
        # spaCy
        config.spacy_model = os.getenv("SPACY_MODEL", "en_core_web_md")
        
        # Sentence transformers
        config.sentence_transformer_model = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
        config.sentence_transformer_device = os.getenv("SENTENCE_TRANSFORMER_DEVICE", "cpu")
        
        # Processing
        config.use_gpu_if_available = os.getenv("USE_GPU_IF_AVAILABLE", "true").lower() == "true"
        config.num_cpu_threads = int(os.getenv("NUM_CPU_THREADS", "4"))
        config.processing_batch_size = int(os.getenv("PROCESSING_BATCH_SIZE", "8"))
        
        # Fallback
        config.enable_cloud_fallback = os.getenv("ENABLE_CLOUD_FALLBACK", "false").lower() == "true"
        
        # Auto-detect GPU
        if config.use_gpu_if_available:
            if torch.cuda.is_available():
                logger.info("✅ CUDA GPU detected - enabling GPU acceleration")
                config.whisper_device = "cuda"
                config.pyannote_device = "cuda"
                config.realesrgan_device = "cuda"
                config.yolo_device = "cuda"
                config.sentence_transformer_device = "cuda"
            elif torch.backends.mps.is_available():
                logger.info("✅ Apple Silicon GPU detected - enabling MPS acceleration")
                config.whisper_device = "mps"
                config.sentence_transformer_device = "mps"
        
        return config


class LocalAIService:
    """Unified service for local AI processing"""
    
    def __init__(self, config: Optional[LocalAIConfig] = None):
        self.config = config or LocalAIConfig.from_env()
        
        # Lazy-loaded models (initialized on first use)
        self._whisper_model = None
        self._pyannote_pipeline = None
        self._tesseract = None
        self._realesrgan_model = None
        self._yolo_model = None
        self._spacy_nlp = None
        self._sentence_transformer = None
        
        logger.info("🤖 Local AI Service initialized")
        logger.info(f"   Whisper: {self.config.whisper_model_size} on {self.config.whisper_device}")
        logger.info(f"   pyannote: {self.config.pyannote_model} on {self.config.pyannote_device}")
        logger.info(f"   YOLO: {self.config.yolo_model} on {self.config.yolo_device}")
    
    @property
    def whisper_model(self):
        """Lazy-load Whisper model"""
        if self._whisper_model is None:
            try:
                import whisper
                logger.info(f"Loading Whisper model: {self.config.whisper_model_size}...")
                self._whisper_model = whisper.load_model(
                    self.config.whisper_model_size,
                    device=self.config.whisper_device,
                    download_root=str(self.config.whisper_model_dir) if self.config.whisper_model_dir else None
                )
                logger.info("✅ Whisper model loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load Whisper: {e}")
                raise
        return self._whisper_model
    
    @property
    def pyannote_pipeline(self):
        """Lazy-load pyannote pipeline"""
        if self._pyannote_pipeline is None:
            try:
                from pyannote.audio import Pipeline
                logger.info(f"Loading pyannote pipeline: {self.config.pyannote_model}...")
                
                if not self.config.huggingface_token:
                    logger.warning("⚠️  HUGGINGFACE_TOKEN not set - pyannote may not work")
                
                self._pyannote_pipeline = Pipeline.from_pretrained(
                    self.config.pyannote_model,
                    use_auth_token=self.config.huggingface_token
                )
                
                if self.config.pyannote_device != "cpu":
                    self._pyannote_pipeline.to(torch.device(self.config.pyannote_device))
                
                logger.info("✅ pyannote pipeline loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load pyannote: {e}")
                raise
        return self._pyannote_pipeline
    
    @property
    def yolo_model(self):
        """Lazy-load YOLO model"""
        if self._yolo_model is None:
            try:
                from ultralytics import YOLO
                logger.info(f"Loading YOLO model: {self.config.yolo_model}...")
                self._yolo_model = YOLO(f"{self.config.yolo_model}.pt")
                logger.info("✅ YOLO model loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load YOLO: {e}")
                raise
        return self._yolo_model
    
    @property
    def spacy_nlp(self):
        """Lazy-load spaCy model"""
        if self._spacy_nlp is None:
            try:
                import spacy
                logger.info(f"Loading spaCy model: {self.config.spacy_model}...")
                self._spacy_nlp = spacy.load(self.config.spacy_model)
                logger.info("✅ spaCy model loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load spaCy: {e}")
                raise
        return self._spacy_nlp
    
    @property
    def sentence_transformer(self):
        """Lazy-load sentence transformer"""
        if self._sentence_transformer is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading sentence transformer: {self.config.sentence_transformer_model}...")
                self._sentence_transformer = SentenceTransformer(
                    self.config.sentence_transformer_model,
                    device=self.config.sentence_transformer_device
                )
                logger.info("✅ Sentence transformer loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load sentence transformer: {e}")
                raise
        return self._sentence_transformer
    
    def transcribe_audio(
        self,
        audio_file: Path,
        language: Optional[str] = None,
        word_timestamps: bool = True
    ) -> Dict:
        """
        Transcribe audio using Whisper
        
        Returns:
        {
            "text": "full transcript",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 5.2,
                    "text": "segment text",
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.5, "probability": 0.98}
                    ]
                }
            ],
            "language": "en"
        }
        """
        logger.info(f"Transcribing audio: {audio_file}")
        
        result = self.whisper_model.transcribe(
            str(audio_file),
            language=language or self.config.whisper_language,
            word_timestamps=word_timestamps,
            verbose=False
        )
        
        logger.info(f"✅ Transcription complete: {len(result.get('segments', []))} segments")
        return result
    
    def diarize_audio(
        self,
        audio_file: Path,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> List[Dict]:
        """
        Perform speaker diarization
        
        Returns:
        [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.2},
            {"speaker": "SPEAKER_01", "start": 5.3, "end": 10.1}
        ]
        """
        logger.info(f"Diarizing audio: {audio_file}")
        
        diarization = self.pyannote_pipeline(
            str(audio_file),
            min_speakers=min_speakers or self.config.pyannote_min_speakers,
            max_speakers=max_speakers or self.config.pyannote_max_speakers
        )
        
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end
            })
        
        logger.info(f"✅ Diarization complete: {len(segments)} segments")
        return segments
    
    def detect_objects(
        self,
        image_or_video: Path,
        confidence: Optional[float] = None
    ) -> List[Dict]:
        """
        Detect objects using YOLO
        
        Returns:
        [
            {
                "class": "person",
                "confidence": 0.95,
                "bbox": [x, y, width, height]
            }
        ]
        """
        logger.info(f"Detecting objects: {image_or_video}")
        
        results = self.yolo_model(
            str(image_or_video),
            conf=confidence or self.config.yolo_confidence,
            device=self.config.yolo_device
        )
        
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    "class": result.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xywh.tolist()[0]
                })
        
        logger.info(f"✅ Object detection complete: {len(detections)} objects")
        return detections
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract entities using spaCy
        
        Returns:
        [
            {"text": "John Smith", "label": "PERSON", "start": 0, "end": 10},
            {"text": "Atlantic County", "label": "GPE", "start": 20, "end": 35}
        ]
        """
        logger.info("Extracting entities from text")
        
        doc = self.spacy_nlp(text)
        
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        logger.info(f"✅ Entity extraction complete: {len(entities)} entities")
        return entities
    
    def semantic_search(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Semantic search using sentence transformers
        
        Returns:
        [(doc_index, similarity_score)]
        """
        logger.info(f"Semantic search: '{query}' across {len(documents)} documents")
        
        # Encode query and documents
        query_embedding = self.sentence_transformer.encode(query, convert_to_tensor=True)
        doc_embeddings = self.sentence_transformer.encode(documents, convert_to_tensor=True)
        
        # Compute cosine similarity
        from torch.nn.functional import cosine_similarity
        similarities = cosine_similarity(query_embedding.unsqueeze(0), doc_embeddings)
        
        # Get top-k results
        top_results = torch.topk(similarities, min(top_k, len(documents)))
        
        results = [
            (int(idx), float(score))
            for score, idx in zip(top_results.values[0], top_results.indices[0])
        ]
        
        logger.info(f"✅ Semantic search complete: {len(results)} results")
        return results


# Global service instance
local_ai = LocalAIService()
