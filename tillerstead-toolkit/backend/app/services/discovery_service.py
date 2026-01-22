"""
BarberX Legal Case Management Pro Suite
E-Discovery & Document Review Service - Premium Legal Analysis
"""
import os
import re
import hashlib
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import uuid

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class PrivilegeType(str, Enum):
    """Types of legal privilege"""
    ATTORNEY_CLIENT = "attorney_client"
    WORK_PRODUCT = "work_product"
    ATTORNEY_CLIENT_WORK_PRODUCT = "attorney_client_work_product"
    JOINT_DEFENSE = "joint_defense"
    COMMON_INTEREST = "common_interest"
    DELIBERATIVE_PROCESS = "deliberative_process"
    LAW_ENFORCEMENT = "law_enforcement"
    NONE = "none"


class ReviewStatus(str, Enum):
    """Document review status"""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    RESPONSIVE = "responsive"
    NON_RESPONSIVE = "non_responsive"
    PRIVILEGED = "privileged"
    PARTIALLY_PRIVILEGED = "partially_privileged"
    NEEDS_REDACTION = "needs_redaction"
    HOT_DOC = "hot_doc"  # Key evidence
    DUPLICATE = "duplicate"


class RedactionType(str, Enum):
    """Types of redactions"""
    PII = "pii"  # Personal identifying information
    SSN = "ssn"
    FINANCIAL = "financial"
    MEDICAL = "medical"
    PRIVILEGED = "privileged"
    TRADE_SECRET = "trade_secret"
    SECURITY = "security"
    MINOR = "minor"  # Protecting minor's identity


@dataclass
class BatesRange:
    """Bates numbering range"""
    prefix: str
    start_number: int
    end_number: int
    document_id: str
    assigned_at: str = ""
    
    @property
    def start_label(self) -> str:
        return f"{self.prefix}{self.start_number:08d}"
    
    @property
    def end_label(self) -> str:
        return f"{self.prefix}{self.end_number:08d}"


@dataclass
class PrivilegeLogEntry:
    """Entry in privilege log"""
    entry_id: str
    document_id: str
    bates_range: Optional[BatesRange]
    date: str
    author: str
    recipients: List[str]
    subject: str
    privilege_type: PrivilegeType
    privilege_description: str
    document_type: str  # email, memo, letter, etc.
    page_count: int = 1
    attachments: List[str] = field(default_factory=list)


@dataclass
class DocumentReview:
    """Document review record"""
    review_id: str
    document_id: str
    reviewer: str
    review_date: str
    status: ReviewStatus
    privilege_type: Optional[PrivilegeType] = None
    privilege_basis: str = ""
    responsiveness_issues: List[str] = field(default_factory=list)
    key_concepts: List[str] = field(default_factory=list)
    custodian: str = ""
    date_range: Tuple[str, str] = ("", "")
    confidentiality_level: str = "standard"  # standard, confidential, highly_confidential, attorneys_eyes_only
    redactions_needed: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    time_spent_seconds: int = 0
    is_hot_doc: bool = False
    linked_documents: List[str] = field(default_factory=list)


@dataclass
class ProductionSet:
    """Document production set"""
    production_id: str
    production_name: str
    bates_prefix: str
    bates_start: int
    bates_current: int
    documents: List[str] = field(default_factory=list)
    total_pages: int = 0
    produced_to: str = ""  # Opposing party name
    production_date: str = ""
    confidentiality_designation: str = "standard"
    load_file_format: str = "concordance"  # concordance, relativity, summation
    image_format: str = "pdf"  # pdf, tiff
    native_production: bool = False
    ocr_included: bool = True
    metadata_fields: List[str] = field(default_factory=list)


@dataclass
class SearchQuery:
    """Discovery search query"""
    query_id: str
    query_text: str
    search_type: str  # keyword, boolean, concept, dtSearch
    date_range: Optional[Tuple[str, str]] = None
    custodians: List[str] = field(default_factory=list)
    file_types: List[str] = field(default_factory=list)
    hit_count: int = 0
    executed_at: str = ""
    results: List[str] = field(default_factory=list)


class EDiscoveryService:
    """
    Premium E-Discovery & Document Review Service
    
    Features:
    - Bates numbering with prefix management
    - Privilege log generation
    - Document review workflow
    - Production set management
    - Advanced search (Boolean, concept, proximity)
    - Duplicate detection (hash + near-duplicate)
    - Redaction tracking
    - Load file generation (Concordance, Relativity)
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.client = AsyncOpenAI(api_key=self.api_key) if OPENAI_AVAILABLE and self.api_key else None
        
        # Bates tracking
        self.bates_counters: Dict[str, int] = {}
        
        # Review tracking
        self.document_reviews: Dict[str, DocumentReview] = {}
        self.privilege_log: List[PrivilegeLogEntry] = []
        self.production_sets: Dict[str, ProductionSet] = {}
        self.search_history: List[SearchQuery] = []
        
        # Duplicate detection
        self.hash_index: Dict[str, List[str]] = {}  # hash -> document_ids
        
    # ============================================================
    # BATES NUMBERING
    # ============================================================
    
    def assign_bates_numbers(
        self,
        document_id: str,
        page_count: int,
        prefix: str = "DEF"
    ) -> BatesRange:
        """
        Assign Bates numbers to document.
        
        Args:
            document_id: Document identifier
            page_count: Number of pages
            prefix: Bates prefix (e.g., "DEF", "PLAINTIFF", "JONES")
        
        Returns:
            BatesRange with assigned numbers
        """
        if prefix not in self.bates_counters:
            self.bates_counters[prefix] = 1
        
        start = self.bates_counters[prefix]
        end = start + page_count - 1
        self.bates_counters[prefix] = end + 1
        
        return BatesRange(
            prefix=prefix,
            start_number=start,
            end_number=end,
            document_id=document_id,
            assigned_at=datetime.utcnow().isoformat()
        )
    
    def get_bates_label(self, prefix: str, number: int, padding: int = 8) -> str:
        """Generate formatted Bates label"""
        return f"{prefix}{number:0{padding}d}"
    
    def parse_bates_label(self, label: str) -> Tuple[str, int]:
        """Parse Bates label into prefix and number"""
        match = re.match(r'^([A-Za-z]+)(\d+)$', label)
        if match:
            return match.group(1), int(match.group(2))
        raise ValueError(f"Invalid Bates label: {label}")
    
    # ============================================================
    # PRIVILEGE REVIEW
    # ============================================================
    
    async def analyze_privilege(
        self,
        document_text: str,
        document_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        AI-powered privilege analysis.
        
        Detects:
        - Attorney-client communications
        - Work product
        - Joint defense materials
        - Common interest doctrine applicability
        """
        if not self.client:
            return self._rule_based_privilege_check(document_text, document_metadata)
        
        prompt = f"""Analyze this document for legal privilege. Consider:

1. ATTORNEY-CLIENT PRIVILEGE:
   - Communication between attorney and client
   - For purpose of legal advice
   - Intended to be confidential
   - Not waived by disclosure

2. WORK PRODUCT DOCTRINE:
   - Prepared in anticipation of litigation
   - By attorney or at attorney's direction
   - Contains mental impressions, conclusions, opinions

3. JOINT DEFENSE PRIVILEGE:
   - Shared between co-defendants/co-parties
   - Common legal interest
   - Reasonable expectation of confidentiality

Document Metadata:
- From: {document_metadata.get('author', 'Unknown')}
- To: {document_metadata.get('recipients', [])}
- Date: {document_metadata.get('date', 'Unknown')}
- Subject: {document_metadata.get('subject', 'Unknown')}

Document Text (first 2000 chars):
{document_text[:2000]}

Provide analysis in JSON format:
{{
    "privilege_detected": true/false,
    "privilege_type": "attorney_client|work_product|joint_defense|common_interest|none",
    "confidence": 0.0-1.0,
    "privilege_basis": "Brief explanation",
    "attorney_involved": "Name if identified",
    "legal_advice_requested": true/false,
    "anticipation_of_litigation": true/false,
    "work_product_indicators": ["list of indicators"],
    "potential_waiver_concerns": ["list of concerns"],
    "recommended_designation": "privileged|partially_privileged|non_privileged",
    "redaction_recommendations": ["specific sections to redact"]
}}"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a senior litigation paralegal expert in privilege review and document classification."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            import json
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            return {
                "error": str(e),
                "privilege_detected": False,
                "fallback": self._rule_based_privilege_check(document_text, document_metadata)
            }
    
    def _rule_based_privilege_check(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Rule-based privilege detection fallback"""
        text_lower = text.lower()
        
        # Attorney-client indicators
        ac_indicators = [
            "privileged and confidential",
            "attorney-client",
            "legal advice",
            "seeking your counsel",
            "attorney work product",
            "prepared at direction of counsel",
            "do not forward",
            "confidential communication"
        ]
        
        # Work product indicators
        wp_indicators = [
            "litigation strategy",
            "case assessment",
            "witness preparation",
            "deposition outline",
            "trial preparation",
            "legal analysis",
            "memo to file",
            "case theory"
        ]
        
        ac_score = sum(1 for ind in ac_indicators if ind in text_lower)
        wp_score = sum(1 for ind in wp_indicators if ind in text_lower)
        
        # Check if attorney in author/recipients
        author = metadata.get('author', '').lower()
        recipients = ' '.join(metadata.get('recipients', [])).lower()
        attorney_keywords = ['esq', 'attorney', 'counsel', 'lawyer', 'jd', 'law firm', 'llp', 'pllc']
        
        attorney_involved = any(kw in author or kw in recipients for kw in attorney_keywords)
        
        privilege_detected = (ac_score >= 2 or wp_score >= 2) and attorney_involved
        
        if ac_score > wp_score:
            privilege_type = PrivilegeType.ATTORNEY_CLIENT
        elif wp_score > ac_score:
            privilege_type = PrivilegeType.WORK_PRODUCT
        elif ac_score > 0 and wp_score > 0:
            privilege_type = PrivilegeType.ATTORNEY_CLIENT_WORK_PRODUCT
        else:
            privilege_type = PrivilegeType.NONE
        
        return {
            "privilege_detected": privilege_detected,
            "privilege_type": privilege_type.value,
            "confidence": min(0.9, (ac_score + wp_score) * 0.15 + (0.3 if attorney_involved else 0)),
            "attorney_involved": attorney_involved,
            "ac_indicators_found": ac_score,
            "wp_indicators_found": wp_score
        }
    
    def create_privilege_log_entry(
        self,
        document_id: str,
        bates_range: Optional[BatesRange],
        metadata: Dict[str, Any],
        privilege_analysis: Dict[str, Any]
    ) -> PrivilegeLogEntry:
        """Create privilege log entry"""
        entry = PrivilegeLogEntry(
            entry_id=f"priv_{uuid.uuid4().hex[:8]}",
            document_id=document_id,
            bates_range=bates_range,
            date=metadata.get('date', ''),
            author=metadata.get('author', ''),
            recipients=metadata.get('recipients', []),
            subject=metadata.get('subject', ''),
            privilege_type=PrivilegeType(privilege_analysis.get('privilege_type', 'none')),
            privilege_description=privilege_analysis.get('privilege_basis', ''),
            document_type=metadata.get('document_type', 'document'),
            page_count=metadata.get('page_count', 1),
            attachments=metadata.get('attachments', [])
        )
        
        self.privilege_log.append(entry)
        return entry
    
    def export_privilege_log(self, format: str = "csv") -> str:
        """Export privilege log"""
        if format == "csv":
            lines = [
                "Bates Range,Date,Author,Recipients,Subject,Privilege Type,Privilege Description,Document Type,Page Count"
            ]
            for entry in self.privilege_log:
                bates = f"{entry.bates_range.start_label}-{entry.bates_range.end_label}" if entry.bates_range else "N/A"
                lines.append(
                    f'"{bates}","{entry.date}","{entry.author}","{"; ".join(entry.recipients)}",'
                    f'"{entry.subject}","{entry.privilege_type.value}","{entry.privilege_description}",'
                    f'"{entry.document_type}",{entry.page_count}'
                )
            return "\n".join(lines)
        
        return ""
    
    # ============================================================
    # DOCUMENT REVIEW
    # ============================================================
    
    def create_review(
        self,
        document_id: str,
        reviewer: str,
        status: ReviewStatus,
        **kwargs
    ) -> DocumentReview:
        """Create document review record"""
        review = DocumentReview(
            review_id=f"rev_{uuid.uuid4().hex[:8]}",
            document_id=document_id,
            reviewer=reviewer,
            review_date=datetime.utcnow().isoformat(),
            status=status,
            **kwargs
        )
        self.document_reviews[review.review_id] = review
        return review
    
    async def ai_assisted_review(
        self,
        document_text: str,
        case_issues: List[str],
        search_terms: List[str]
    ) -> Dict[str, Any]:
        """
        AI-assisted document review.
        
        Analyzes for:
        - Responsiveness to case issues
        - Key concepts and entities
        - Recommended coding
        - Hot document indicators
        """
        if not self.client:
            return self._basic_responsiveness_check(document_text, case_issues, search_terms)
        
        issues_text = "\n".join(f"- {issue}" for issue in case_issues)
        terms_text = ", ".join(search_terms)
        
        prompt = f"""Review this document for litigation relevance.

CASE ISSUES:
{issues_text}

KEY SEARCH TERMS: {terms_text}

DOCUMENT TEXT:
{document_text[:4000]}

Analyze and provide JSON response:
{{
    "responsive": true/false,
    "responsiveness_score": 0.0-1.0,
    "relevant_issues": ["list of case issues this document relates to"],
    "key_concepts": ["important concepts/entities found"],
    "key_dates": ["significant dates mentioned"],
    "key_people": ["people mentioned"],
    "key_organizations": ["organizations mentioned"],
    "hot_document_indicators": ["reasons this might be a key document"],
    "is_hot_document": true/false,
    "recommended_status": "responsive|non_responsive|needs_further_review",
    "confidentiality_concerns": ["any sensitivity issues"],
    "summary": "2-3 sentence summary of document",
    "recommended_tags": ["suggested classification tags"]
}}"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert litigation document reviewer. Analyze documents for relevance to case issues."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            import json
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            return {"error": str(e)}
    
    def _basic_responsiveness_check(
        self,
        text: str,
        issues: List[str],
        terms: List[str]
    ) -> Dict[str, Any]:
        """Basic responsiveness check without AI"""
        text_lower = text.lower()
        
        # Check for search terms
        term_hits = sum(1 for term in terms if term.lower() in text_lower)
        
        # Check for issue keywords
        issue_hits = 0
        relevant_issues = []
        for issue in issues:
            issue_words = issue.lower().split()
            if any(word in text_lower for word in issue_words if len(word) > 3):
                issue_hits += 1
                relevant_issues.append(issue)
        
        score = min(1.0, (term_hits * 0.1) + (issue_hits * 0.2))
        
        return {
            "responsive": score > 0.3,
            "responsiveness_score": score,
            "relevant_issues": relevant_issues,
            "term_hits": term_hits,
            "issue_hits": issue_hits
        }
    
    # ============================================================
    # PRODUCTION MANAGEMENT
    # ============================================================
    
    def create_production_set(
        self,
        name: str,
        bates_prefix: str,
        produced_to: str,
        **kwargs
    ) -> ProductionSet:
        """Create new production set"""
        production = ProductionSet(
            production_id=f"prod_{uuid.uuid4().hex[:8]}",
            production_name=name,
            bates_prefix=bates_prefix,
            bates_start=self.bates_counters.get(bates_prefix, 1),
            bates_current=self.bates_counters.get(bates_prefix, 1),
            produced_to=produced_to,
            production_date=datetime.utcnow().isoformat(),
            **kwargs
        )
        self.production_sets[production.production_id] = production
        return production
    
    def add_to_production(
        self,
        production_id: str,
        document_ids: List[str],
        page_counts: Dict[str, int]
    ) -> Dict[str, BatesRange]:
        """Add documents to production set with Bates numbering"""
        production = self.production_sets.get(production_id)
        if not production:
            raise ValueError(f"Production set not found: {production_id}")
        
        bates_assignments = {}
        
        for doc_id in document_ids:
            pages = page_counts.get(doc_id, 1)
            bates = self.assign_bates_numbers(doc_id, pages, production.bates_prefix)
            bates_assignments[doc_id] = bates
            production.documents.append(doc_id)
            production.total_pages += pages
            production.bates_current = bates.end_number + 1
        
        return bates_assignments
    
    def generate_load_file(
        self,
        production_id: str,
        format: str = "concordance"
    ) -> str:
        """
        Generate load file for production.
        
        Formats:
        - concordance: .dat file format
        - relativity: .csv with Relativity fields
        - summation: .dii format
        """
        production = self.production_sets.get(production_id)
        if not production:
            raise ValueError(f"Production set not found: {production_id}")
        
        if format == "concordance":
            return self._generate_concordance_dat(production)
        elif format == "relativity":
            return self._generate_relativity_csv(production)
        else:
            return self._generate_concordance_dat(production)
    
    def _generate_concordance_dat(self, production: ProductionSet) -> str:
        """Generate Concordance .dat load file"""
        delimiter = "\x14"  # Concordance delimiter
        quote = "\xfe"  # Concordance quote character
        newline = "\x0a"
        
        # Header
        fields = [
            "BEGBATES", "ENDBATES", "BEGATTACH", "ENDATTACH",
            "CUSTODIAN", "FROM", "TO", "CC", "BCC",
            "SUBJECT", "DATESENT", "DATERECEIVED",
            "FILENAME", "FILEPATH", "PAGECOUNT",
            "CONFIDENTIALITY", "DOCTYPE"
        ]
        
        lines = [delimiter.join(f"{quote}{f}{quote}" for f in fields)]
        
        # Data rows would be populated from actual document metadata
        # This is a template structure
        
        return newline.join(lines)
    
    def _generate_relativity_csv(self, production: ProductionSet) -> str:
        """Generate Relativity-compatible CSV"""
        fields = [
            "Control Number", "Beg Bates", "End Bates",
            "Beg Attach", "End Attach", "Custodian",
            "From", "To", "CC", "Subject",
            "Date Sent", "File Name", "Native File Path",
            "Text Path", "Page Count", "Confidentiality"
        ]
        
        lines = [",".join(f'"{f}"' for f in fields)]
        return "\n".join(lines)
    
    # ============================================================
    # DUPLICATE DETECTION
    # ============================================================
    
    def calculate_document_hash(self, content: bytes) -> str:
        """Calculate document hash for deduplication"""
        return hashlib.sha256(content).hexdigest()
    
    def check_duplicate(self, content: bytes, document_id: str) -> Dict[str, Any]:
        """Check if document is duplicate"""
        doc_hash = self.calculate_document_hash(content)
        
        if doc_hash in self.hash_index:
            existing_docs = self.hash_index[doc_hash]
            return {
                "is_duplicate": True,
                "duplicate_of": existing_docs,
                "hash": doc_hash
            }
        
        # Add to index
        self.hash_index[doc_hash] = self.hash_index.get(doc_hash, []) + [document_id]
        
        return {
            "is_duplicate": False,
            "hash": doc_hash
        }
    
    def find_near_duplicates(
        self,
        document_text: str,
        similarity_threshold: float = 0.85
    ) -> List[Dict[str, Any]]:
        """Find near-duplicate documents using text similarity"""
        # This would use MinHash or SimHash for efficient near-duplicate detection
        # Placeholder for implementation
        return []
    
    # ============================================================
    # SEARCH
    # ============================================================
    
    def execute_search(
        self,
        query: str,
        search_type: str = "boolean",
        **filters
    ) -> SearchQuery:
        """Execute discovery search"""
        search = SearchQuery(
            query_id=f"search_{uuid.uuid4().hex[:8]}",
            query_text=query,
            search_type=search_type,
            date_range=filters.get('date_range'),
            custodians=filters.get('custodians', []),
            file_types=filters.get('file_types', []),
            executed_at=datetime.utcnow().isoformat()
        )
        
        self.search_history.append(search)
        return search
    
    def parse_boolean_query(self, query: str) -> Dict[str, Any]:
        """Parse Boolean search query"""
        # Parse AND, OR, NOT, proximity operators
        tokens = []
        
        # Handle phrases in quotes
        phrase_pattern = r'"([^"]+)"'
        phrases = re.findall(phrase_pattern, query)
        
        # Handle proximity searches w/n
        proximity_pattern = r'(\w+)\s+w/(\d+)\s+(\w+)'
        proximity_matches = re.findall(proximity_pattern, query)
        
        return {
            "phrases": phrases,
            "proximity_searches": proximity_matches,
            "operators": re.findall(r'\b(AND|OR|NOT)\b', query, re.IGNORECASE)
        }


# Service instance
discovery_service = EDiscoveryService()
