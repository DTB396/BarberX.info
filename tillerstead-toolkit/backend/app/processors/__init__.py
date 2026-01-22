"""
BarberX Legal Case Management Pro Suite
Processors Package
"""
from .pdf_processor import PDFProcessor, DocumentClassifier, extract_pdf_text, classify_document
from .bwc_processor import (
    MotorolaBWCParser, BWCVideoProcessor, MultiPOVSynchronizer, AudioHarmonizer,
    parse_bwc_filename, get_video_duration
)

__all__ = [
    # PDF Processing
    'PDFProcessor',
    'DocumentClassifier', 
    'extract_pdf_text',
    'classify_document',
    
    # BWC Processing
    'MotorolaBWCParser',
    'BWCVideoProcessor',
    'MultiPOVSynchronizer',
    'AudioHarmonizer',
    'parse_bwc_filename',
    'get_video_duration',
]
