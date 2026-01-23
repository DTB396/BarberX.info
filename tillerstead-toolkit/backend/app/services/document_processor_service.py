"""
Document Discovery Processor - PDF/Document Evidence Processing
================================================================

Handles document discovery with:
- OCR and text-layer repair for scanned PDFs
- Metadata extraction (author, dates, embedded data)
- Near-duplicate detection
- Bates stamping and exhibit numbering
- Redaction management with audit trails
- Email processing and threading

Author: BarberX Legal Case Management
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum
from collections import defaultdict

import PyPDF2
# Would require: pip install PyPDF2 pypdf pikepdf pdf2image pytesseract pillow python-magic


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class DocumentType(str, Enum):
    """Document types"""
    PDF_NATIVE = "PDF (Native)"
    PDF_SCANNED = "PDF (Scanned/Image)"
    EMAIL = "Email Message"
    SPREADSHEET = "Spreadsheet"
    WORD_DOCUMENT = "Word Document"
    TEXT_FILE = "Text File"
    IMAGE = "Image/Photograph"
    PRESENTATION = "Presentation"


class BatesPrefix(str, Enum):
    """Bates numbering prefixes for different doc types"""
    DEFENDANT = "DEF"
    PLAINTIFF = "PLT"
    EXHIBIT = "EX"
    CONFIDENTIAL = "CONF"
    ATTORNEYS_EYES_ONLY = "AEO"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class DocumentMetadata:
    """Extracted document metadata"""
    author: Optional[str] = None
    created_date: Optional[str] = None
    modified_date: Optional[str] = None
    title: Optional[str] = None
    subject: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    producer: Optional[str] = None  # PDF producer
    application: Optional[str] = None
    page_count: int = 0
    file_size_bytes: int = 0
    has_text_layer: bool = False
    is_searchable: bool = False
    embedded_attachments: List[str] = field(default_factory=list)


@dataclass
class BatesStamp:
    """Bates stamping information"""
    prefix: str
    start_number: int
    end_number: int
    format: str  # e.g., "DEF-00001"
    applied_date: str
    applied_by: str
    page_range: str  # e.g., "1-5"


@dataclass
class Redaction:
    """Redaction record"""
    redaction_id: str
    page_number: int
    coordinates: Dict  # {x, y, width, height}
    reason: str  # "Privacy", "Privileged", "Irrelevant", etc.
    applied_by: str
    applied_date: str
    review_required: bool = False
    notes: str = ""


@dataclass
class NearDuplicate:
    """Near-duplicate document match"""
    doc_id_1: str
    doc_id_2: str
    similarity_score: float  # 0.0 to 1.0
    match_type: str  # "Exact", "Near_Duplicate", "Version"
    differences_summary: str = ""


@dataclass
class DocumentProcessingResult:
    """Complete document processing result"""
    evidence_id: str
    case_id: str
    document_type: DocumentType
    original_file: str
    processed_file: str
    
    # Extracted data
    metadata: DocumentMetadata
    extracted_text: str = ""
    ocr_performed: bool = False
    ocr_confidence: float = 0.0
    
    # Bates/numbering
    bates_stamp: Optional[BatesStamp] = None
    exhibit_number: Optional[str] = None
    
    # Redactions
    redactions: List[Redaction] = field(default_factory=list)
    
    # Duplicates
    near_duplicates: List[NearDuplicate] = field(default_factory=list)
    
    # Processing metadata
    processing_date: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    processing_notes: str = ""
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['document_type'] = self.document_type.value
        return data


# ============================================================================
# DOCUMENT PROCESSOR SERVICE
# ============================================================================

class DocumentProcessorService:
    """
    Process document discovery with OCR, Bates stamping, and redaction management
    """
    
    def __init__(self, output_dir: str = "./document_processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.ocr_dir = self.output_dir / "ocr"
        self.bates_dir = self.output_dir / "bates_stamped"
        self.redacted_dir = self.output_dir / "redacted"
        self.metadata_dir = self.output_dir / "metadata"
        
        for directory in [self.ocr_dir, self.bates_dir, self.redacted_dir, self.metadata_dir]:
            directory.mkdir(exist_ok=True)
        
        # Track Bates numbering
        self.bates_counters: Dict[str, int] = defaultdict(int)
        
        # Store document hashes for deduplication
        self.document_hashes: Dict[str, str] = {}  # hash -> evidence_id
    
    async def process_document(
        self,
        file_path: Path,
        evidence_id: str,
        case_id: str,
        perform_ocr: bool = True,
        apply_bates: bool = False,
        bates_prefix: str = "DEF"
    ) -> DocumentProcessingResult:
        """
        Process document with OCR, metadata extraction, and optional Bates stamping
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        doc_type = self._detect_document_type(file_path)
        
        # Extract metadata
        metadata = await self._extract_metadata(file_path, doc_type)
        
        # Extract text
        extracted_text = ""
        ocr_performed = False
        ocr_confidence = 0.0
        
        if doc_type == DocumentType.PDF_NATIVE and metadata.has_text_layer:
            extracted_text = await self._extract_pdf_text(file_path)
        elif perform_ocr and doc_type in [DocumentType.PDF_SCANNED, DocumentType.IMAGE]:
            extracted_text, ocr_confidence = await self._perform_ocr(file_path, evidence_id)
            ocr_performed = True
        
        # Check for duplicates
        near_dupes = await self._detect_near_duplicates(evidence_id, extracted_text, file_path)
        
        # Bates stamping
        bates_stamp = None
        processed_file = str(file_path)
        if apply_bates:
            bates_stamp, processed_file = await self._apply_bates_stamp(
                file_path, evidence_id, bates_prefix, metadata.page_count
            )
        
        # Build result
        result = DocumentProcessingResult(
            evidence_id=evidence_id,
            case_id=case_id,
            document_type=doc_type,
            original_file=str(file_path),
            processed_file=processed_file,
            metadata=metadata,
            extracted_text=extracted_text,
            ocr_performed=ocr_performed,
            ocr_confidence=ocr_confidence,
            bates_stamp=bates_stamp,
            near_duplicates=near_dupes
        )
        
        # Save metadata
        await self._save_document_metadata(result)
        
        return result
    
    def _detect_document_type(self, file_path: Path) -> DocumentType:
        """Detect document type"""
        ext = file_path.suffix.lower()
        
        if ext == '.pdf':
            # Check if PDF has text layer
            try:
                with open(file_path, 'rb') as f:
                    pdf = PyPDF2.PdfReader(f)
                    if len(pdf.pages) > 0:
                        text = pdf.pages[0].extract_text()
                        return DocumentType.PDF_NATIVE if text.strip() else DocumentType.PDF_SCANNED
            except:
                pass
            return DocumentType.PDF_SCANNED
        
        elif ext in ['.doc', '.docx']:
            return DocumentType.WORD_DOCUMENT
        elif ext in ['.xls', '.xlsx', '.csv']:
            return DocumentType.SPREADSHEET
        elif ext in ['.eml', '.msg']:
            return DocumentType.EMAIL
        elif ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            return DocumentType.IMAGE
        elif ext in ['.ppt', '.pptx']:
            return DocumentType.PRESENTATION
        else:
            return DocumentType.TEXT_FILE
    
    async def _extract_metadata(self, file_path: Path, doc_type: DocumentType) -> DocumentMetadata:
        """Extract document metadata"""
        metadata = DocumentMetadata(
            file_size_bytes=file_path.stat().st_size
        )
        
        if doc_type in [DocumentType.PDF_NATIVE, DocumentType.PDF_SCANNED]:
            try:
                with open(file_path, 'rb') as f:
                    pdf = PyPDF2.PdfReader(f)
                    metadata.page_count = len(pdf.pages)
                    
                    if pdf.metadata:
                        metadata.author = pdf.metadata.get('/Author')
                        metadata.title = pdf.metadata.get('/Title')
                        metadata.subject = pdf.metadata.get('/Subject')
                        metadata.producer = pdf.metadata.get('/Producer')
                        
                        created = pdf.metadata.get('/CreationDate')
                        modified = pdf.metadata.get('/ModDate')
                        
                        if created:
                            metadata.created_date = self._parse_pdf_date(created)
                        if modified:
                            metadata.modified_date = self._parse_pdf_date(modified)
                    
                    # Check if has text layer
                    if len(pdf.pages) > 0:
                        text = pdf.pages[0].extract_text()
                        metadata.has_text_layer = bool(text.strip())
                        metadata.is_searchable = metadata.has_text_layer
            except Exception as e:
                print(f"Error extracting PDF metadata: {e}")
        
        return metadata
    
    def _parse_pdf_date(self, pdf_date: str) -> str:
        """Parse PDF date format to ISO"""
        # PDF dates: D:20250122143022-05'00'
        match = re.match(r"D:(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", str(pdf_date))
        if match:
            year, month, day, hour, minute, second = match.groups()
            return f"{year}-{month}-{day}T{hour}:{minute}:{second}"
        return str(pdf_date)
    
    async def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from searchable PDF"""
        text_parts = []
        try:
            with open(file_path, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                for page in pdf.pages:
                    text_parts.append(page.extract_text())
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
        
        return "\n\n".join(text_parts)
    
    async def _perform_ocr(self, file_path: Path, evidence_id: str) -> Tuple[str, float]:
        """
        Perform OCR on scanned PDF or image
        
        Production would use: Tesseract OCR, AWS Textract, Google Cloud Vision,  Azure OCR
        """
        # Simulated OCR result
        ocr_text = """
        [OCR SIMULATED TEXT]
        
        This is a simulated OCR extraction from a scanned document.
        In production, this would use Tesseract or a cloud OCR service.
        
        OCR would detect:
        - Rotated pages
        - Handwritten text
        - Stamps and signatures
        - Form fields
        - Tables
        """
        
        ocr_confidence = 0.89  # Average word-level confidence
        
        # Save OCR output
        ocr_output_path = self.ocr_dir / f"{evidence_id}_ocr.txt"
        with open(ocr_output_path, 'w', encoding='utf-8') as f:
            f.write(ocr_text)
        
        return ocr_text, ocr_confidence
    
    async def _apply_bates_stamp(
        self,
        file_path: Path,
        evidence_id: str,
        prefix: str,
        page_count: int
    ) -> Tuple[BatesStamp, str]:
        """
        Apply Bates stamping to PDF
        
        Production would use: pikepdf, PyMuPDF, or dedicated Bates stamping tools
        """
        # Get next Bates number for this prefix
        start_number = self.bates_counters[prefix] + 1
        end_number = start_number + page_count - 1
        
        # Update counter
        self.bates_counters[prefix] = end_number
        
        # Generate Bates format
        bates_format = f"{prefix}-{{:05d}}"
        start_bates = bates_format.format(start_number)
        end_bates = bates_format.format(end_number)
        
        # In production, would stamp each page
        # For now, just copy file
        output_path = self.bates_dir / f"{evidence_id}_bates_{start_bates}-{end_bates}.pdf"
        
        bates_stamp = BatesStamp(
            prefix=prefix,
            start_number=start_number,
            end_number=end_number,
            format=f"{start_bates} - {end_bates}",
            applied_date=datetime.utcnow().isoformat(),
            applied_by="System",
            page_range=f"1-{page_count}"
        )
        
        return bates_stamp, str(output_path)
    
    async def _detect_near_duplicates(
        self,
        evidence_id: str,
        text_content: str,
        file_path: Path
    ) -> List[NearDuplicate]:
        """
        Detect near-duplicate documents using text similarity
        
        Production would use: MinHash, SimHash, fuzzy hashing (ssdeep)
        """
        duplicates = []
        
        # Compute document hash
        doc_hash = hashlib.md5(text_content.encode()).hexdigest()
        
        # Check if exact duplicate exists
        if doc_hash in self.document_hashes:
            existing_id = self.document_hashes[doc_hash]
            duplicates.append(NearDuplicate(
                doc_id_1=evidence_id,
                doc_id_2=existing_id,
                similarity_score=1.0,
                match_type="Exact",
                differences_summary="Exact duplicate (same content hash)"
            ))
        else:
            self.document_hashes[doc_hash] = evidence_id
        
        # Near-duplicate detection would use fuzzy matching here
        
        return duplicates
    
    async def apply_redaction(
        self,
        evidence_id: str,
        page_number: int,
        coordinates: Dict,
        reason: str,
        applied_by: str
    ) -> Redaction:
        """
        Apply redaction to document page
        
        Production would use: pikepdf, PyMuPDF with permanent redaction
        """
        redaction = Redaction(
            redaction_id=f"RED-{evidence_id}-{page_number}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            page_number=page_number,
            coordinates=coordinates,
            reason=reason,
            applied_by=applied_by,
            applied_date=datetime.utcnow().isoformat()
        )
        
        # In production, would apply permanent redaction to PDF
        # Save redacted version to redacted_dir
        
        return redaction
    
    async def generate_redaction_log(self, evidence_id: str, redactions: List[Redaction]) -> Path:
        """Generate audit log of all redactions"""
        log_path = self.redacted_dir / f"{evidence_id}_redaction_log.json"
        
        log_data = {
            "evidence_id": evidence_id,
            "total_redactions": len(redactions),
            "redactions": [asdict(r) for r in redactions],
            "log_generated": datetime.utcnow().isoformat()
        }
        
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        return log_path
    
    async def _save_document_metadata(self, result: DocumentProcessingResult):
        """Save document processing metadata"""
        metadata_path = self.metadata_dir / f"{result.evidence_id}_document_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)


# ============================================================================
# GLOBAL SERVICE INSTANCE
# ============================================================================

document_processor = DocumentProcessorService()
