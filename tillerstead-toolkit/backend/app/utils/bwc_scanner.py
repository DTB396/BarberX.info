"""
BarberX Legal Case Management Pro Suite
BWC Folder Scanner & Importer
Scans Motorola BWC folder structure and imports footage metadata

Supports two folder structures:
1. Flat: .bwc/25-41706 Barber, Devon/OfficerName_timestamp_device.mp4
2. Officer subfolders: .bwc/atl-l-003252-25/officer_Name/*.mp4
"""
import re
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class OfficerFootage:
    """Footage from a single officer"""
    officer_name: str
    badge_number: Optional[str] = None
    files: List[Dict[str, Any]] = field(default_factory=list)
    total_files: int = 0
    total_size_bytes: int = 0
    earliest_timestamp: Optional[datetime] = None
    latest_timestamp: Optional[datetime] = None
    device_ids: List[str] = field(default_factory=list)
    folder_path: Optional[str] = None


@dataclass
class IncidentFootage:
    """All footage for an incident"""
    case_number: str
    docket_number: Optional[str] = None
    folder_path: str = ""
    officers: Dict[str, OfficerFootage] = field(default_factory=dict)
    total_files: int = 0
    total_officers: int = 0
    total_size_bytes: int = 0
    documents: List[Dict[str, Any]] = field(default_factory=list)
    date_range: Optional[Tuple[datetime, datetime]] = None
    has_officer_subfolders: bool = False


class BWCFolderScanner:
    """
    Scan Motorola BWC folder structures.
    
    Supports two patterns:
    1. Flat structure:
       .bwc/25-41706 Barber, Devon/
       ├── BryanMerritt_202511292256_311-0.mp4
       ├── EdwardRuiz_202511292250_312-0.mp4
       └── Barber, Devon T. (2025-41706) Pro Se.pdf
    
    2. Officer subfolder structure:
       .bwc/atl-l-003252-25/
       ├── officer_Hare/
       │   └── RachelHare_202511292258_BWL7139108-0.mp4
       ├── officer_Ruiz/
       │   └── EdwardRuiz_*.mp4
       └── officer_Merritt/
           └── BryanMerritt_*.mp4
    """
    
    # Motorola filename pattern
    MOTOROLA_PATTERN = re.compile(
        r'^([A-Za-z]+)_(\d{12})_([A-Z0-9]+-?\d*)-?(\d+)?\.mp4$',
        re.IGNORECASE
    )
    
    # Alternative pattern with spaces in officer name
    ALT_PATTERN = re.compile(
        r'^([A-Za-z\s]+)_(\d{12})_([A-Z0-9]+-?\d*)\.mp4$',
        re.IGNORECASE
    )
    
    # Officer subfolder pattern (officer_Name or officer_name)
    OFFICER_FOLDER_PATTERN = re.compile(r'^officer_([A-Za-z]+)$', re.IGNORECASE)
    
    # Document patterns
    DOCUMENT_PATTERNS = ['.pdf', '.doc', '.docx', '.txt']
    VIDEO_PATTERNS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    # NJ Docket pattern
    NJ_DOCKET_PATTERN = re.compile(r'(atl|ber|bur|cam|cap|cum|ess|glo|hud|hun|mer|mid|mon|mor|ocn|pas|sal|som|sus|uni|war)-[a-z]+-\d+-\d+', re.IGNORECASE)
    
    def __init__(self, base_path: Optional[Path] = None):
        """Initialize scanner with optional base path"""
        self.base_path = Path(base_path) if base_path else None
    
    def scan_folder(self, folder_path: Path) -> IncidentFootage:
        """
        Scan a single incident folder for BWC footage.
        
        Args:
            folder_path: Path to incident folder
            
        Returns:
            IncidentFootage with all discovered files
        """
        folder_path = Path(folder_path)
        
        # Extract case/docket number from folder name
        case_number = self._extract_case_number(folder_path.name)
        docket_number = self._extract_docket_number(folder_path.name)
        
        incident = IncidentFootage(
            case_number=case_number,
            docket_number=docket_number,
            folder_path=str(folder_path)
        )
        
        # Check for officer subfolder structure
        officer_folders = self._find_officer_folders(folder_path)
        
        if officer_folders:
            incident.has_officer_subfolders = True
            for officer_folder in officer_folders:
                self._scan_officer_folder(officer_folder, incident)
        
        # Also scan root folder for any files
        for item in folder_path.iterdir():
            if item.is_file():
                self._process_file(item, incident)
        
        # Calculate totals and date range
        self._finalize_incident(incident)
        
        return incident
    
    def _find_officer_folders(self, folder_path: Path) -> List[Path]:
        """Find officer_* subfolders"""
        officer_folders = []
        for item in folder_path.iterdir():
            if item.is_dir():
                match = self.OFFICER_FOLDER_PATTERN.match(item.name)
                if match:
                    officer_folders.append(item)
        return officer_folders
    
    def _scan_officer_folder(self, officer_folder: Path, incident: IncidentFootage):
        """Scan an officer-specific subfolder"""
        # Extract officer name from folder
        match = self.OFFICER_FOLDER_PATTERN.match(officer_folder.name)
        if match:
            folder_officer_name = match.group(1).title()
        else:
            folder_officer_name = officer_folder.name
        
        # Scan all files in this officer's folder
        for file_path in officer_folder.rglob("*"):
            if file_path.is_file():
                self._process_file(file_path, incident, folder_officer_hint=folder_officer_name)
    
    def _process_file(
        self, 
        file_path: Path, 
        incident: IncidentFootage,
        folder_officer_hint: Optional[str] = None
    ):
        """Process a single file"""
        suffix = file_path.suffix.lower()
        
        if suffix in self.VIDEO_PATTERNS:
            self._process_video(file_path, incident, folder_officer_hint)
        elif suffix in self.DOCUMENT_PATTERNS:
            self._process_document(file_path, incident)
    
    def _process_video(
        self, 
        file_path: Path, 
        incident: IncidentFootage,
        folder_officer_hint: Optional[str] = None
    ):
        """Process a BWC video file"""
        parsed = self._parse_motorola_filename(file_path.name)
        
        if not parsed:
            # Unknown format - use folder hint or Unknown
            parsed = {
                'officer_name': folder_officer_hint or 'Unknown',
                'timestamp': None,
                'device_id': 'Unknown',
                'segment': 0
            }
        
        officer_name = parsed['officer_name']
        
        # Get or create officer record
        if officer_name not in incident.officers:
            incident.officers[officer_name] = OfficerFootage(
                officer_name=officer_name
            )
        
        officer = incident.officers[officer_name]
        
        # Get file stats
        stat = file_path.stat()
        
        file_info = {
            'path': str(file_path),
            'filename': file_path.name,
            'size_bytes': stat.st_size,
            'timestamp': parsed['timestamp'],
            'device_id': parsed['device_id'],
            'segment': parsed['segment'],
            'modified': datetime.fromtimestamp(stat.st_mtime)
        }
        
        officer.files.append(file_info)
        officer.total_files += 1
        officer.total_size_bytes += stat.st_size
        
        # Track device IDs
        if parsed['device_id'] and parsed['device_id'] not in officer.device_ids:
            officer.device_ids.append(parsed['device_id'])
        
        # Track timestamps
        if parsed['timestamp']:
            ts = parsed['timestamp']
            if officer.earliest_timestamp is None or ts < officer.earliest_timestamp:
                officer.earliest_timestamp = ts
            if officer.latest_timestamp is None or ts > officer.latest_timestamp:
                officer.latest_timestamp = ts
    
    def _process_document(self, file_path: Path, incident: IncidentFootage):
        """Process a document file"""
        stat = file_path.stat()
        
        incident.documents.append({
            'path': str(file_path),
            'filename': file_path.name,
            'type': file_path.suffix.lower(),
            'size_bytes': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime)
        })
    
    def _parse_motorola_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Parse Motorola BWC filename"""
        match = self.MOTOROLA_PATTERN.match(filename)
        if not match:
            match = self.ALT_PATTERN.match(filename)
            if not match:
                return None
        
        groups = match.groups()
        officer_name = groups[0]
        timestamp_str = groups[1]
        device_id = groups[2]
        segment = int(groups[3]) if len(groups) > 3 and groups[3] else 0
        
        # Parse timestamp (YYYYMMDDHHMI)
        try:
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d%H%M')
        except ValueError:
            timestamp = None
        
        # Format officer name (CamelCase to "First Last")
        officer_name = self._format_officer_name(officer_name)
        
        return {
            'officer_name': officer_name,
            'timestamp': timestamp,
            'device_id': device_id,
            'segment': segment,
            'raw_timestamp': timestamp_str
        }
    
    def _format_officer_name(self, name: str) -> str:
        """Format officer name from CamelCase"""
        # Insert space before capital letters
        formatted = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        return formatted.title()
    
    def _finalize_incident(self, incident: IncidentFootage):
        """Calculate totals and finalize incident data"""
        incident.total_officers = len(incident.officers)
        incident.total_files = sum(o.total_files for o in incident.officers.values())
        incident.total_size_bytes = sum(o.total_size_bytes for o in incident.officers.values())
        
        # Calculate overall date range
        all_earliest = []
        all_latest = []
        
        for officer in incident.officers.values():
            if officer.earliest_timestamp:
                all_earliest.append(officer.earliest_timestamp)
            if officer.latest_timestamp:
                all_latest.append(officer.latest_timestamp)
        
        if all_earliest and all_latest:
            incident.date_range = (min(all_earliest), max(all_latest))
    
    def scan_bwc_root(self, bwc_root: Path) -> List[IncidentFootage]:
        """
        Scan a .bwc root folder for all incidents.
        
        Args:
            bwc_root: Path to .bwc folder
            
        Returns:
            List of IncidentFootage for all found incidents
        """
        bwc_root = Path(bwc_root)
        incidents = []
        
        for item in bwc_root.iterdir():
            if item.is_dir():
                incident = self.scan_folder(item)
                incidents.append(incident)
        
        return incidents
    
    def generate_report(self, incident: IncidentFootage) -> Dict[str, Any]:
        """Generate a summary report for an incident"""
        return {
            'case_number': incident.case_number,
            'docket_number': incident.docket_number,
            'folder_path': incident.folder_path,
            'has_officer_subfolders': incident.has_officer_subfolders,
            'total_officers': incident.total_officers,
            'total_video_files': incident.total_files,
            'total_size_mb': round(incident.total_size_bytes / (1024 * 1024), 2),
            'total_documents': len(incident.documents),
            'date_range': {
                'start': incident.date_range[0].isoformat() if incident.date_range else None,
                'end': incident.date_range[1].isoformat() if incident.date_range else None
            },
            'officers': [
                {
                    'name': officer.officer_name,
                    'files': officer.total_files,
                    'size_mb': round(officer.total_size_bytes / (1024 * 1024), 2),
                    'devices': officer.device_ids,
                    'time_range': {
                        'start': officer.earliest_timestamp.isoformat() if officer.earliest_timestamp else None,
                        'end': officer.latest_timestamp.isoformat() if officer.latest_timestamp else None
                    }
                }
                for officer in incident.officers.values()
            ],
            'documents': [
                {
                    'filename': doc['filename'],
                    'type': doc['type']
                }
                for doc in incident.documents
            ]
        }


class BWCTimelineBuilder:
    """Build synchronized timeline from multi-officer footage"""
    
    def __init__(self, incident: IncidentFootage):
        self.incident = incident
    
    def build_timeline(self) -> List[Dict[str, Any]]:
        """
        Build chronological timeline of all footage.
        
        Returns:
            List of timeline events sorted by timestamp
        """
        events = []
        
        for officer in self.incident.officers.values():
            for file_info in officer.files:
                if file_info['timestamp']:
                    events.append({
                        'timestamp': file_info['timestamp'],
                        'officer': officer.officer_name,
                        'filename': file_info['filename'],
                        'device_id': file_info['device_id'],
                        'segment': file_info['segment'],
                        'path': file_info['path']
                    })
        
        # Sort by timestamp
        events.sort(key=lambda e: e['timestamp'])
        
        return events
    
    def find_overlapping_footage(
        self,
        tolerance_minutes: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find footage from multiple officers at same time.
        
        Args:
            tolerance_minutes: Time window to consider overlapping
            
        Returns:
            List of overlapping footage groups
        """
        from datetime import timedelta
        
        timeline = self.build_timeline()
        overlaps = []
        
        for i, event in enumerate(timeline):
            group = [event]
            
            for j in range(i + 1, len(timeline)):
                other = timeline[j]
                
                # Check if within tolerance
                delta = abs((other['timestamp'] - event['timestamp']).total_seconds())
                if delta <= tolerance_minutes * 60:
                    if other['officer'] != event['officer']:
                        group.append(other)
                else:
                    break
            
            if len(group) > 1:
                overlaps.append({
                    'timestamp': event['timestamp'].isoformat(),
                    'officers': list(set(e['officer'] for e in group)),
                    'files': [e['filename'] for e in group]
                })
        
        # Deduplicate overlaps
        seen = set()
        unique = []
        for overlap in overlaps:
            key = tuple(sorted(overlap['officers'])) + (overlap['timestamp'][:16],)
            if key not in seen:
                seen.add(key)
                unique.append(overlap)
        
        return unique


# Convenience function
def scan_bwc_folder(folder_path: str) -> Dict[str, Any]:
    """Quick scan of a BWC folder"""
    scanner = BWCFolderScanner()
    incident = scanner.scan_folder(Path(folder_path))
    return scanner.generate_report(incident)


def find_all_incidents(bwc_root: str) -> List[Dict[str, Any]]:
    """Scan all incidents in a .bwc folder"""
    scanner = BWCFolderScanner()
    incidents = scanner.scan_bwc_root(Path(bwc_root))
    return [scanner.generate_report(i) for i in incidents]
