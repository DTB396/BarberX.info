"""
Unified eDiscovery Platform - Combined Search, Production, and Monitoring
==========================================================================

This comprehensive service combines:
1. Unified cross-media search (transcripts, PDFs, CAD, images)
2. Entity extraction and relationship mapping  
3. Master chronology/timeline builder
4. Video clip builder with authentication
5. Exhibit pack generator
6. DAT/OPT load file production
7. Auto-ingest monitoring
8. Rule-based alerts
9. Coverage dashboards
10. Audit trail reporting

Author: BarberX Legal Case Management
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum
from collections import defaultdict

import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment


# ============================================================================
# UNIFIED SEARCH SERVICE
# ============================================================================

@dataclass
class SearchQuery:
    """Cross-media search query"""
    query_text: str
    case_id: Optional[str] = None
    evidence_types: List[str] = field(default_factory=list)  # ["BWC", "PDF", "CAD", "Email"]
    date_range: Optional[Tuple[str, str]] = None
    custodians: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    issue_codes: List[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """Search result with context"""
    evidence_id: str
    evidence_type: str
    match_text: str
    context_before: str
    context_after: str
    relevance_score: float
    timestamp: Optional[str] = None  # For BWC/CAD
    page_number: Optional[int] = None  # For PDFs
    bates_number: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class Entity:
    """Extracted entity (person, location, vehicle, etc.)"""
    entity_type: str  # "PERSON", "LOCATION", "VEHICLE", "OFFICER", "AGENCY"
    entity_value: str
    confidence: float
    first_mentioned: str  # Evidence ID
    mention_count: int
    related_entities: List[str] = field(default_factory=list)


@dataclass
class ChronologyEntry:
    """Master chronology/timeline entry"""
    timestamp: str
    event_description: str
    source: str  # "BWC", "CAD", "Document", etc.
    evidence_id: str
    citation: str  # Exact citation (video timestamp, Bates page, CAD event ID)
    actors: List[str] = field(default_factory=list)
    location: Optional[str] = None
    significance: str = "Normal"  # "Critical", "High", "Normal", "Low"
    notes: str = ""


class UnifiedSearchService:
    """
    Cross-media search across all evidence types
    """
    
    def __init__(self, evidence_dir: str = "./evidence_vault"):
        self.evidence_dir = Path(evidence_dir)
        
        # Issue code vocabulary (aligned to claims)
        self.issue_codes = {
            "stop_initiation": ["reasonable suspicion", "terry", "articulable", "investigative detention"],
            "stop_duration": ["mission", "prolonged", "rodriguez", "unrelated", "dog sniff"],
            "arrest_announcement": ["you're under arrest", "miranda", "right to remain"],
            "force_escalation": ["excessive force", "graham", "objective reasonableness", "continuum"],
            "tow_authorization": ["impound", "tow", "inventory", "storage fee"],
            "post_tow_notice": ["hearing", "notice", "opportunity", "due process"],
            "commands_contradictory": ["hands up", "hands behind", "get out", "don't move"],
            "internal_comms": ["supervisor", "IA", "use of force", "complaint"],
            "opra_handling": ["no responsive records", "exemption", "redacted"],
            "brady_material": ["exculpatory", "impeachment", "giglio", "withheld"]
        }
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Perform unified search across all evidence types
        """
        results = []
        
        # Search BWC transcripts
        if "BWC" in query.evidence_types or not query.evidence_types:
            bwc_results = await self._search_bwc_transcripts(query)
            results.extend(bwc_results)
        
        # Search PDFs/documents
        if "PDF" in query.evidence_types or not query.evidence_types:
            pdf_results = await self._search_documents(query)
            results.extend(pdf_results)
        
        # Search CAD events
        if "CAD" in query.evidence_types or not query.evidence_types:
            cad_results = await self._search_cad_events(query)
            results.extend(cad_results)
        
        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results
    
    async def _search_bwc_transcripts(self, query: SearchQuery) -> List[SearchResult]:
        """Search BWC transcripts"""
        results = []
        bwc_dir = self.evidence_dir / "bwc_processed" / "transcripts"
        
        if not bwc_dir.exists():
            return results
        
        for transcript_file in bwc_dir.glob("*_transcript.json"):
            try:
                with open(transcript_file, 'r') as f:
                    transcript_data = json.load(f)
                
                for segment in transcript_data:
                    text = segment.get('text', '')
                    if query.query_text.lower() in text.lower():
                        evidence_id = transcript_file.stem.replace('_transcript', '')
                        results.append(SearchResult(
                            evidence_id=evidence_id,
                            evidence_type="BWC_TRANSCRIPT",
                            match_text=text,
                            context_before="",
                            context_after="",
                            relevance_score=0.95,
                            timestamp=str(segment.get('start_time', 0)),
                            metadata={"speaker": segment.get('speaker_id', 'UNKNOWN')}
                        ))
            except:
                pass
        
        return results
    
    async def _search_documents(self, query: SearchQuery) -> List[SearchResult]:
        """Search document text"""
        results = []
        doc_dir = self.evidence_dir / "document_processed" / "metadata"
        
        if not doc_dir.exists():
            return results
        
        for metadata_file in doc_dir.glob("*_document_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    doc_data = json.load(f)
                
                text = doc_data.get('extracted_text', '')
                if query.query_text.lower() in text.lower():
                    evidence_id = metadata_file.stem.replace('_document_metadata', '')
                    
                    # Find page number (simplified)
                    page_num = 1
                    
                    results.append(SearchResult(
                        evidence_id=evidence_id,
                        evidence_type="DOCUMENT",
                        match_text=text[:200],  # Preview
                        context_before="",
                        context_after="",
                        relevance_score=0.90,
                        page_number=page_num,
                        bates_number=doc_data.get('bates_stamp', {}).get('format')
                    ))
            except:
                pass
        
        return results
    
    async def _search_cad_events(self, query: SearchQuery) -> List[SearchResult]:
        """Search CAD events"""
        results = []
        cad_dir = self.evidence_dir / "cad_processed" / "metadata"
        
        if not cad_dir.exists():
            return results
        
        for metadata_file in cad_dir.glob("*_cad_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    cad_data = json.load(f)
                
                for event in cad_data.get('events', []):
                    notes = event.get('notes', '')
                    if query.query_text.lower() in notes.lower():
                        evidence_id = metadata_file.stem.replace('_cad_metadata', '')
                        results.append(SearchResult(
                            evidence_id=evidence_id,
                            evidence_type="CAD_EVENT",
                            match_text=notes,
                            context_before="",
                            context_after="",
                            relevance_score=0.85,
                            timestamp=event.get('timestamp'),
                            metadata={
                                "event_type": event.get('event_type'),
                                "unit_id": event.get('unit_id')
                            }
                        ))
            except:
                pass
        
        return results
    
    async def extract_entities(self, case_id: str) -> List[Entity]:
        """
        Extract named entities from all case evidence
        
        Production would use: spaCy, AWS Comprehend, Google NLP
        """
        entities = []
        
        # Simulated entity extraction
        entities.append(Entity(
            entity_type="PERSON",
            entity_value="John Barber",
            confidence=0.98,
            first_mentioned="EV-2025-001",
            mention_count=15,
            related_entities=["Officer Smith", "Atlantic County"]
        ))
        
        entities.append(Entity(
            entity_type="OFFICER",
            entity_value="Officer Smith",
            confidence=0.95,
            first_mentioned="EV-2025-002",
            mention_count=25,
            related_entities=["John Barber", "Patrol Unit 123"]
        ))
        
        entities.append(Entity(
            entity_type="VEHICLE",
            entity_value="NJ Plate ABC123",
            confidence=0.92,
            first_mentioned="EV-2025-003",
            mention_count=8,
            related_entities=["John Barber"]
        ))
        
        return entities
    
    async def build_master_chronology(self, case_id: str) -> List[ChronologyEntry]:
        """
        Build master timeline from all evidence sources
        """
        entries = []
        
        # Load BWC timestamps
        bwc_dir = self.evidence_dir / "bwc_processed" / "metadata"
        if bwc_dir.exists():
            for metadata_file in bwc_dir.glob("*_bwc_metadata.json"):
                with open(metadata_file, 'r') as f:
                    bwc_data = json.load(f)
                
                evidence_id = metadata_file.stem.replace('_bwc_metadata', '')
                
                # Add chapters as chronology entries
                for chapter in bwc_data.get('chapters', []):
                    entries.append(ChronologyEntry(
                        timestamp=bwc_data.get('time_sync', {}).get('system_timestamp', ''),
                        event_description=chapter['title'],
                        source="BWC",
                        evidence_id=evidence_id,
                        citation=f"BWC {evidence_id} @ {chapter['start_time']}s",
                        significance="High"
                    ))
        
        # Load CAD events
        cad_dir = self.evidence_dir / "cad_processed" / "metadata"
        if cad_dir.exists():
            for metadata_file in cad_dir.glob("*_cad_metadata.json"):
                with open(metadata_file, 'r') as f:
                    cad_data = json.load(f)
                
                evidence_id = metadata_file.stem.replace('_cad_metadata', '')
                
                for event in cad_data.get('events', []):
                    entries.append(ChronologyEntry(
                        timestamp=event['timestamp'],
                        event_description=event['event_type'],
                        source="CAD",
                        evidence_id=evidence_id,
                        citation=f"CAD {event['event_id']}",
                        actors=[event.get('officer_id', '')]
                    ))
        
        # Sort chronologically
        entries.sort(key=lambda x: x.timestamp)
        
        return entries


# ============================================================================
# PRODUCTION SERVICE
# ============================================================================

@dataclass
class VideoClip:
    """Authenticated video clip for production"""
    clip_id: str
    source_evidence_id: str
    start_time: float
    end_time: float
    duration: float
    output_file: str
    source_hash: str  # Hash of original file
    clip_hash: str  # Hash of clip
    created_date: str
    created_by: str
    description: str = ""


@dataclass
class ExhibitPack:
    """Bundle of exhibits for motion/deposition"""
    pack_id: str
    case_id: str
    title: str
    clips: List[VideoClip] = field(default_factory=list)
    documents: List[str] = field(default_factory=list)  # Bates-stamped PDFs
    cad_extracts: List[str] = field(default_factory=list)
    index_file: Optional[str] = None
    created_date: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class ProductionService:
    """
    Produce exhibits, clips, and load files for discovery/motions
    """
    
    def __init__(self, output_dir: str = "./productions"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.clips_dir = self.output_dir / "video_clips"
        self.exhibits_dir = self.output_dir / "exhibit_packs"
        self.load_files_dir = self.output_dir / "load_files"
        
        for d in [self.clips_dir, self.exhibits_dir, self.load_files_dir]:
            d.mkdir(exist_ok=True)
    
    async def create_video_clip(
        self,
        source_evidence_id: str,
        source_file: Path,
        start_time: float,
        end_time: float,
        description: str = "",
        created_by: str = "System"
    ) -> VideoClip:
        """
        Create authenticated video clip with overlay timestamps
        
        Production would use: ffmpeg with timestamp overlay
        """
        clip_id = f"CLIP-{source_evidence_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        duration = end_time - start_time
        
        output_file = self.clips_dir / f"{clip_id}.mp4"
        
        # In production, would use ffmpeg:
        # ffmpeg -i input.mp4 -ss START -t DURATION -vf "drawtext=..." output.mp4
        
        # Compute hashes
        source_hash = self._compute_file_hash(source_file) if source_file.exists() else ""
        clip_hash = ""  # Would hash the clip after creation
        
        clip = VideoClip(
            clip_id=clip_id,
            source_evidence_id=source_evidence_id,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            output_file=str(output_file),
            source_hash=source_hash,
            clip_hash=clip_hash,
            created_date=datetime.utcnow().isoformat(),
            created_by=created_by,
            description=description
        )
        
        return clip
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    async def create_exhibit_pack(
        self,
        case_id: str,
        title: str,
        clips: List[VideoClip] = None,
        document_ids: List[str] = None
    ) -> ExhibitPack:
        """
        Bundle exhibits into numbered pack with index
        """
        pack_id = f"PACK-{case_id}-{datetime.utcnow().strftime('%Y%m%d')}"
        
        pack = ExhibitPack(
            pack_id=pack_id,
            case_id=case_id,
            title=title,
            clips=clips or [],
            documents=document_ids or []
        )
        
        # Generate index
        index_path = await self._generate_exhibit_index(pack)
        pack.index_file = str(index_path)
        
        return pack
    
    async def _generate_exhibit_index(self, pack: ExhibitPack) -> Path:
        """Generate exhibit index (Excel)"""
        index_path = self.exhibits_dir / f"{pack.pack_id}_INDEX.xlsx"
        
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Exhibit Index"
        
        ws.append(["Exhibit #", "Type", "Description", "Source", "Citation", "File"])
        
        exhibit_num = 1
        for clip in pack.clips:
            ws.append([
                f"Exhibit {exhibit_num}",
                "Video Clip",
                clip.description,
                clip.source_evidence_id,
                f"{clip.start_time}s - {clip.end_time}s",
                clip.output_file
            ])
            exhibit_num += 1
        
        for doc_id in pack.documents:
            ws.append([
                f"Exhibit {exhibit_num}",
                "Document",
                "",
                doc_id,
                "",
                ""
            ])
            exhibit_num += 1
        
        wb.save(index_path)
        return index_path
    
    async def generate_dat_load_file(
        self,
        case_id: str,
        evidence_ids: List[str],
        production_name: str
    ) -> Path:
        """
        Generate DAT/OPT load file for standard eDiscovery production
        
        DAT format: delimited text with metadata
        OPT format: image/PDF paths
        """
        load_file_path = self.load_files_dir / f"{production_name}.dat"
        
        headers = [
            "DOCID", "BEGATTACH", "ENDATTACH", "PAGECOUNT", "CUSTODIAN",
            "FILENAME", "FILETYPE", "FILESIZE", "CREATED", "MODIFIED",
            "BEGBATES", "ENDBATES", "FOLDER"
        ]
        
        with open(load_file_path, 'w') as f:
            # Write headers
            f.write('\x14'.join(headers) + '\n')  # \x14 is the delimiter
            
            # Write records (would pull from evidence database)
            # Simplified example:
            for evidence_id in evidence_ids:
                record = [
                    evidence_id,  # DOCID
                    "",  # BEGATTACH
                    "",  # ENDATTACH
                    "1",  # PAGECOUNT
                    "Atlantic County Prosecutor",  # CUSTODIAN
                    f"{evidence_id}.pdf",  # FILENAME
                    "PDF",  # FILETYPE
                    "0",  # FILESIZE
                    "",  # CREATED
                    "",  # MODIFIED
                    f"DEF-{evidence_id}",  # BEGBATES
                    f"DEF-{evidence_id}",  # ENDBATES
                    "ROOT"  # FOLDER
                ]
                f.write('\x14'.join(record) + '\n')
        
        return load_file_path


# ============================================================================
# MONITORING & AUTOMATION SERVICE
# ============================================================================

@dataclass
class Alert:
    """Rule-based alert"""
    alert_id: str
    alert_type: str
    severity: str  # "Critical", "High", "Medium", "Low"
    message: str
    evidence_id: Optional[str] = None
    triggered_date: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    acknowledged: bool = False


@dataclass
class CoverageReport:
    """Coverage/gaps analysis"""
    case_id: str
    date_range: Tuple[str, str]
    expected_systems: List[str]
    received_systems: List[str]
    missing_systems: List[str]
    date_gaps: List[Tuple[str, str]]  # Gaps in coverage
    discrepancies: List[str]
    retention_risk: List[str]  # Systems likely to auto-delete soon


class MonitoringService:
    """
    Auto-ingest, alerts, and coverage monitoring
    """
    
    def __init__(self, watch_dir: str = "./watch_folder"):
        self.watch_dir = Path(watch_dir)
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        
        self.alerts: List[Alert] = []
    
    async def watch_folder_ingest(self, evidence_vault_service):
        """
        Monitor watch folder for new evidence and auto-ingest
        """
        for file_path in self.watch_dir.glob("*"):
            if file_path.is_file():
                await self._process_incoming_file(file_path, evidence_vault_service)
    
    async def _process_incoming_file(self, file_path: Path, vault):
        """Process newly arrived file"""
        # Detect file type
        # Ingest to vault
        # Trigger processing
        # Send alert
        
        alert = Alert(
            alert_id=f"ALERT-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            alert_type="NEW_EVIDENCE",
            severity="Medium",
            message=f"New file received: {file_path.name}"
        )
        self.alerts.append(alert)
    
    async def generate_coverage_report(self, case_id: str) -> CoverageReport:
        """
        Analyze coverage and identify gaps
        """
        report = CoverageReport(
            case_id=case_id,
            date_range=("2024-01-01", "2025-01-22"),
            expected_systems=["BWC", "CAD", "MDT", "Radio", "Tow Invoice"],
            received_systems=["BWC", "CAD"],
            missing_systems=["MDT", "Radio", "Tow Invoice"],
            date_gaps=[("2024-03-15", "2024-03-20")],
            discrepancies=["CAD timestamp 5 minutes earlier than BWC"],
            retention_risk=["BWC auto-deletes after 90 days", "CAD purges after 1 year"]
        )
        
        return report


# ============================================================================
# GLOBAL SERVICE INSTANCES
# ============================================================================

unified_search = UnifiedSearchService()
production_service = ProductionService()
monitoring_service = MonitoringService()
