"""
BarberX Legal Case Management Pro Suite
PDF Processor - Document OCR, Text Extraction, Analysis
"""
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

# PDF processing imports (graceful fallback if not installed)
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import pytesseract
    from PIL import Image
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False


@dataclass
class PDFPage:
    """Extracted page content"""
    page_num: int
    text: str
    images: List[Dict[str, Any]]
    links: List[str]


@dataclass
class PDFMetadata:
    """PDF document metadata"""
    title: Optional[str]
    author: Optional[str]
    subject: Optional[str]
    creator: Optional[str]
    producer: Optional[str]
    creation_date: Optional[datetime]
    modification_date: Optional[datetime]
    page_count: int
    file_size: int


@dataclass
class PDFExtractionResult:
    """Complete PDF extraction result"""
    metadata: PDFMetadata
    full_text: str
    pages: List[PDFPage]
    ocr_applied: bool
    ocr_confidence: Optional[float]
    extracted_dates: List[datetime]
    extracted_entities: Dict[str, List[str]]


class PDFProcessor:
    """
    PDF Processing Engine for Legal Documents
    
    Features:
    - Text extraction (native + OCR fallback)
    - Metadata extraction
    - Date/entity recognition
    - Bates numbering
    - Page analysis
    """
    
    def __init__(self, tesseract_path: Optional[str] = None):
        """Initialize processor with optional Tesseract path"""
        if tesseract_path and HAS_TESSERACT:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    def extract_text(self, file_path: str) -> PDFExtractionResult:
        """
        Extract all text and metadata from a PDF.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            PDFExtractionResult with text, metadata, and analysis
        """
        if not HAS_PYMUPDF:
            raise RuntimeError("PyMuPDF not installed. Run: pip install PyMuPDF")
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")
        
        doc = fitz.open(file_path)
        
        # Extract metadata
        meta = doc.metadata
        metadata = PDFMetadata(
            title=meta.get("title"),
            author=meta.get("author"),
            subject=meta.get("subject"),
            creator=meta.get("creator"),
            producer=meta.get("producer"),
            creation_date=self._parse_pdf_date(meta.get("creationDate")),
            modification_date=self._parse_pdf_date(meta.get("modDate")),
            page_count=len(doc),
            file_size=path.stat().st_size
        )
        
        # Extract pages
        pages = []
        full_text_parts = []
        ocr_applied = False
        ocr_confidences = []
        
        for page_num, page in enumerate(doc, 1):
            # Try native text extraction first
            text = page.get_text()
            
            # If minimal text, try OCR
            if len(text.strip()) < 50 and HAS_TESSERACT:
                ocr_text, confidence = self._ocr_page(page)
                if ocr_text:
                    text = ocr_text
                    ocr_applied = True
                    ocr_confidences.append(confidence)
            
            # Extract images info
            images = []
            for img_index, img in enumerate(page.get_images()):
                images.append({
                    "index": img_index,
                    "xref": img[0],
                    "width": img[2],
                    "height": img[3]
                })
            
            # Extract links
            links = []
            for link in page.get_links():
                if "uri" in link:
                    links.append(link["uri"])
            
            pages.append(PDFPage(
                page_num=page_num,
                text=text,
                images=images,
                links=links
            ))
            full_text_parts.append(text)
        
        doc.close()
        
        full_text = "\n\n".join(full_text_parts)
        
        # Extract dates and entities
        extracted_dates = self._extract_dates(full_text)
        extracted_entities = self._extract_entities(full_text)
        
        # Calculate average OCR confidence
        avg_confidence = sum(ocr_confidences) / len(ocr_confidences) if ocr_confidences else None
        
        return PDFExtractionResult(
            metadata=metadata,
            full_text=full_text,
            pages=pages,
            ocr_applied=ocr_applied,
            ocr_confidence=avg_confidence,
            extracted_dates=extracted_dates,
            extracted_entities=extracted_entities
        )
    
    def _ocr_page(self, page) -> Tuple[Optional[str], float]:
        """
        OCR a PDF page using Tesseract.
        
        Returns:
            Tuple of (text, confidence)
        """
        if not HAS_TESSERACT:
            return None, 0.0
        
        try:
            # Render page to image
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Run OCR
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            
            # Extract text and calculate confidence
            texts = []
            confidences = []
            
            for i, text in enumerate(data['text']):
                if text.strip():
                    texts.append(text)
                    conf = data['conf'][i]
                    if conf > 0:
                        confidences.append(conf)
            
            full_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return full_text, avg_confidence / 100  # Normalize to 0-1
            
        except Exception as e:
            print(f"OCR error: {e}")
            return None, 0.0
    
    def _parse_pdf_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse PDF date format (D:YYYYMMDDHHmmSS)"""
        if not date_str:
            return None
        
        try:
            # Remove 'D:' prefix if present
            if date_str.startswith("D:"):
                date_str = date_str[2:]
            
            # Parse YYYYMMDDHHMMSS format
            return datetime.strptime(date_str[:14], "%Y%m%d%H%M%S")
        except:
            return None
    
    def _extract_dates(self, text: str) -> List[datetime]:
        """Extract dates from text"""
        dates = []
        
        # Common date patterns
        patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY or M/D/YY
            r'\d{1,2}-\d{1,2}-\d{2,4}',  # MM-DD-YYYY
            r'\w+ \d{1,2}, \d{4}',  # Month DD, YYYY
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD (ISO)
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                parsed = self._try_parse_date(match)
                if parsed:
                    dates.append(parsed)
        
        return sorted(set(dates))
    
    def _try_parse_date(self, date_str: str) -> Optional[datetime]:
        """Try to parse a date string with multiple formats"""
        formats = [
            "%m/%d/%Y", "%m/%d/%y",
            "%m-%d-%Y", "%m-%d-%y",
            "%B %d, %Y", "%b %d, %Y",
            "%Y-%m-%d"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
        return None
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        entities = {
            "case_numbers": [],
            "names": [],
            "badge_numbers": [],
            "addresses": [],
            "phone_numbers": []
        }
        
        # Case number patterns
        case_patterns = [
            r'\b\d{2}-\d{4,6}\b',  # YY-NNNNN
            r'\b[A-Z]{2,3}-[A-Z]-\d{5,7}-\d{2}\b',  # ATL-L-001234-25
            r'\b\d:\d{2}-[a-z]{2}-\d{5}\b',  # Federal: 1:22-cv-06206
        ]
        for pattern in case_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["case_numbers"].extend(matches)
        
        # Badge numbers (various formats)
        badge_patterns = [
            r'\bBadge\s*#?\s*(\d{3,6})\b',
            r'\bOfficer\s*#?\s*(\d{3,6})\b',
        ]
        for pattern in badge_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["badge_numbers"].extend(matches)
        
        # Phone numbers
        phone_pattern = r'\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
        entities["phone_numbers"] = re.findall(phone_pattern, text)
        
        # Deduplicate
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def add_bates_numbers(
        self,
        input_path: str,
        output_path: str,
        prefix: str = "EX",
        start_number: int = 1,
        position: str = "bottom-right"
    ) -> int:
        """
        Add Bates numbers to a PDF document.
        
        Args:
            input_path: Source PDF path
            output_path: Destination PDF path
            prefix: Bates number prefix (e.g., "EX", "DEF")
            start_number: Starting number
            position: Where to place stamp (bottom-right, bottom-center, etc.)
            
        Returns:
            Next available Bates number
        """
        if not HAS_PYMUPDF:
            raise RuntimeError("PyMuPDF not installed")
        
        doc = fitz.open(input_path)
        
        for page_num, page in enumerate(doc):
            bates_num = f"{prefix}{start_number + page_num:05d}"
            
            # Calculate position
            rect = page.rect
            if position == "bottom-right":
                point = fitz.Point(rect.width - 70, rect.height - 20)
            elif position == "bottom-center":
                point = fitz.Point(rect.width / 2 - 30, rect.height - 20)
            else:  # bottom-left
                point = fitz.Point(20, rect.height - 20)
            
            # Add text
            page.insert_text(
                point,
                bates_num,
                fontsize=10,
                color=(0, 0, 0)
            )
        
        doc.save(output_path)
        doc.close()
        
        return start_number + len(doc)
    
    def calculate_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()


class DocumentClassifier:
    """
    Legal Document Classifier
    
    Classifies documents based on content analysis.
    """
    
    # Keywords for classification
    CLASSIFICATION_RULES = {
        "complaint": [
            "plaintiff", "defendant", "comes now", "cause of action",
            "wherefore", "prayer for relief", "jurisdiction"
        ],
        "motion": [
            "motion to", "movant", "moves this court", "memorandum in support",
            "good cause", "hereby moves"
        ],
        "order": [
            "it is hereby ordered", "so ordered", "the court orders",
            "it is ordered that"
        ],
        "police_report": [
            "incident report", "police report", "badge number",
            "officer name", "subject description", "narrative"
        ],
        "deposition": [
            "deponent", "deposition of", "examination by",
            "q:", "a:", "direct examination", "cross examination"
        ],
        "discovery_request": [
            "interrogatory", "request for production", "request for admission",
            "demand for", "please produce"
        ],
        "affidavit": [
            "affidavit of", "sworn statement", "being duly sworn",
            "under penalty of perjury", "notary public"
        ]
    }
    
    def classify(self, text: str) -> Tuple[str, float]:
        """
        Classify document type based on text content.
        
        Returns:
            Tuple of (document_type, confidence)
        """
        text_lower = text.lower()
        scores = {}
        
        for doc_type, keywords in self.CLASSIFICATION_RULES.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            
            if score > 0:
                scores[doc_type] = score / len(keywords)
        
        if not scores:
            return "other", 0.5
        
        best_type = max(scores.items(), key=lambda x: x[1])
        return best_type[0], min(0.95, best_type[1] + 0.3)


# Convenience functions
def extract_pdf_text(file_path: str) -> str:
    """Quick text extraction from PDF"""
    processor = PDFProcessor()
    result = processor.extract_text(file_path)
    return result.full_text


def classify_document(text: str) -> str:
    """Quick document classification"""
    classifier = DocumentClassifier()
    doc_type, _ = classifier.classify(text)
    return doc_type
