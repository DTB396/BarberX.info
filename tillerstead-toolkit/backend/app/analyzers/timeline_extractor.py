"""
BarberX Legal Case Management Pro Suite
Timeline Extractor
Automatically extracts and organizes chronological events from documents
"""
import re
from datetime import datetime, date, time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class EventType(Enum):
    """Types of timeline events"""
    INCIDENT = "incident"
    ARREST = "arrest"
    BOOKING = "booking"
    COURT_APPEARANCE = "court_appearance"
    FILING = "filing"
    MOTION = "motion"
    ORDER = "order"
    HEARING = "hearing"
    DEPOSITION = "deposition"
    DISCOVERY = "discovery"
    MEDICAL = "medical"
    WITNESS_INTERVIEW = "witness_interview"
    EVIDENCE_COLLECTED = "evidence_collected"
    COMMUNICATION = "communication"
    OTHER = "other"


@dataclass
class TimelineEvent:
    """Represents a single timeline event"""
    event_date: date
    event_time: Optional[time]
    event_type: EventType
    title: str
    description: str
    actors: List[str]
    source_document: str
    source_reference: str
    confidence: float
    keywords: List[str]
    metadata: Dict[str, Any]


class TimelineExtractor:
    """
    Extract chronological events from text documents.
    
    Uses pattern matching and NLP to identify dates, times,
    and associated events.
    """
    
    # Date patterns
    DATE_PATTERNS = [
        # MM/DD/YYYY or MM-DD-YYYY
        (r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})', 'mdy'),
        (r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{2})', 'mdy_short'),
        # Month DD, YYYY
        (r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})', 'month_name'),
        # DD Month YYYY
        (r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})', 'day_month'),
        # YYYY-MM-DD
        (r'(\d{4})-(\d{2})-(\d{2})', 'iso'),
    ]
    
    # Time patterns
    TIME_PATTERNS = [
        # HH:MM:SS AM/PM
        (r'(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(AM|PM|am|pm|a\.m\.|p\.m\.)', 'ampm'),
        # HH:MM:SS (24hr)
        (r'(\d{2}):(\d{2})(?::(\d{2}))?(?!\s*(?:AM|PM))', '24hr'),
        # Military time references
        (r'(\d{4})\s*(?:hours?|hrs?)', 'military'),
    ]
    
    # Event type keywords
    EVENT_KEYWORDS = {
        EventType.ARREST: [
            'arrested', 'taken into custody', 'apprehended',
            'placed under arrest', 'handcuffed', 'detained'
        ],
        EventType.INCIDENT: [
            'incident', 'occurred', 'happened', 'took place',
            'assault', 'attack', 'confrontation', 'encounter'
        ],
        EventType.FILING: [
            'filed', 'submitted', 'lodged', 'complaint filed',
            'petition filed', 'motion filed'
        ],
        EventType.COURT_APPEARANCE: [
            'appeared in court', 'arraignment', 'initial appearance',
            'sentencing', 'plea', 'trial date'
        ],
        EventType.HEARING: [
            'hearing', 'conference', 'oral argument',
            'status conference', 'pretrial'
        ],
        EventType.MOTION: [
            'motion to', 'motion for', 'moved to', 'moved for',
            'suppression motion', 'dismiss'
        ],
        EventType.ORDER: [
            'court ordered', 'order entered', 'judge ordered',
            'ruling', 'decision', 'judgment'
        ],
        EventType.MEDICAL: [
            'medical', 'hospital', 'emergency room', 'ER',
            'treated', 'diagnosed', 'injury', 'injuries'
        ],
        EventType.DISCOVERY: [
            'discovery', 'produced', 'interrogatories',
            'request for production', 'subpoena'
        ],
        EventType.EVIDENCE_COLLECTED: [
            'evidence collected', 'seized', 'obtained',
            'recovered', 'bwc footage', 'body camera'
        ]
    }
    
    # Actor patterns (who was involved)
    ACTOR_PATTERNS = [
        r'Officer\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'Detective\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'Sgt\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'Lieutenant\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'Judge\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'(?:the\s+)?(?:defendant|plaintiff|petitioner|respondent)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
    ]
    
    MONTH_MAP = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    
    def __init__(self):
        """Initialize the extractor"""
        self.events = []
    
    def extract_from_text(
        self,
        text: str,
        source_document: str = "",
        base_page: int = 1
    ) -> List[TimelineEvent]:
        """
        Extract timeline events from text.
        
        Args:
            text: Document text to analyze
            source_document: Name/path of source document
            base_page: Starting page number
            
        Returns:
            List of extracted timeline events
        """
        events = []
        
        # Split into paragraphs/sentences for context
        paragraphs = self._split_paragraphs(text)
        
        for para_idx, paragraph in enumerate(paragraphs):
            # Find dates in paragraph
            dates = self._extract_dates(paragraph)
            
            if not dates:
                continue
            
            # For each date, extract event details
            for event_date, date_span in dates:
                # Get time if present
                event_time = self._extract_time_near(paragraph, date_span)
                
                # Determine event type
                event_type = self._classify_event(paragraph)
                
                # Extract actors
                actors = self._extract_actors(paragraph)
                
                # Generate title and description
                title = self._generate_title(paragraph, event_type)
                description = self._clean_description(paragraph)
                
                # Extract keywords
                keywords = self._extract_keywords(paragraph)
                
                # Calculate confidence
                confidence = self._calculate_confidence(
                    paragraph, event_date, event_type
                )
                
                event = TimelineEvent(
                    event_date=event_date,
                    event_time=event_time,
                    event_type=event_type,
                    title=title,
                    description=description,
                    actors=actors,
                    source_document=source_document,
                    source_reference=f"Paragraph {base_page + para_idx}",
                    confidence=confidence,
                    keywords=keywords,
                    metadata={}
                )
                
                events.append(event)
        
        # Remove duplicates and sort
        events = self._deduplicate_events(events)
        events.sort(key=lambda e: (e.event_date, e.event_time or time(0, 0)))
        
        return events
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into meaningful paragraphs"""
        # Split on double newlines or numbered lists
        paragraphs = re.split(r'\n\s*\n|\d+\.\s+', text)
        
        # Filter empty and very short paragraphs
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 30]
        
        return paragraphs
    
    def _extract_dates(self, text: str) -> List[Tuple[date, Tuple[int, int]]]:
        """Extract dates from text with their positions"""
        dates = []
        
        for pattern, format_type in self.DATE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    parsed_date = self._parse_date_match(match, format_type)
                    if parsed_date:
                        dates.append((parsed_date, match.span()))
                except ValueError:
                    continue
        
        return dates
    
    def _parse_date_match(self, match: re.Match, format_type: str) -> Optional[date]:
        """Parse a regex match into a date"""
        groups = match.groups()
        
        try:
            if format_type == 'mdy':
                return date(int(groups[2]), int(groups[0]), int(groups[1]))
            
            elif format_type == 'mdy_short':
                year = int(groups[2])
                year = year + 2000 if year < 50 else year + 1900
                return date(year, int(groups[0]), int(groups[1]))
            
            elif format_type == 'month_name':
                month = self.MONTH_MAP.get(groups[0].lower(), 1)
                return date(int(groups[2]), month, int(groups[1]))
            
            elif format_type == 'day_month':
                month = self.MONTH_MAP.get(groups[1].lower(), 1)
                return date(int(groups[2]), month, int(groups[0]))
            
            elif format_type == 'iso':
                return date(int(groups[0]), int(groups[1]), int(groups[2]))
        
        except (ValueError, TypeError):
            return None
        
        return None
    
    def _extract_time_near(
        self,
        text: str,
        date_span: Tuple[int, int]
    ) -> Optional[time]:
        """Extract time near a date reference"""
        # Look in a window around the date
        start = max(0, date_span[0] - 50)
        end = min(len(text), date_span[1] + 50)
        window = text[start:end]
        
        for pattern, format_type in self.TIME_PATTERNS:
            match = re.search(pattern, window, re.IGNORECASE)
            if match:
                return self._parse_time_match(match, format_type)
        
        return None
    
    def _parse_time_match(self, match: re.Match, format_type: str) -> Optional[time]:
        """Parse a time match"""
        groups = match.groups()
        
        try:
            if format_type == 'ampm':
                hour = int(groups[0])
                minute = int(groups[1])
                second = int(groups[2]) if groups[2] else 0
                period = groups[3].upper().replace('.', '')
                
                if period.startswith('P') and hour != 12:
                    hour += 12
                elif period.startswith('A') and hour == 12:
                    hour = 0
                
                return time(hour, minute, second)
            
            elif format_type == '24hr':
                return time(int(groups[0]), int(groups[1]), int(groups[2] or 0))
            
            elif format_type == 'military':
                military = int(groups[0])
                return time(military // 100, military % 100)
        
        except ValueError:
            return None
        
        return None
    
    def _classify_event(self, text: str) -> EventType:
        """Classify event type based on keywords"""
        text_lower = text.lower()
        
        best_type = EventType.OTHER
        best_score = 0
        
        for event_type, keywords in self.EVENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_score:
                best_score = score
                best_type = event_type
        
        return best_type
    
    def _extract_actors(self, text: str) -> List[str]:
        """Extract actors (people involved) from text"""
        actors = []
        
        for pattern in self.ACTOR_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            actors.extend(matches)
        
        # Deduplicate
        return list(dict.fromkeys(actors))
    
    def _generate_title(self, text: str, event_type: EventType) -> str:
        """Generate a concise title for the event"""
        # Get first sentence or first 100 chars
        sentences = re.split(r'[.!?]', text)
        first = sentences[0].strip() if sentences else text[:100]
        
        # Truncate if needed
        if len(first) > 80:
            first = first[:77] + '...'
        
        # Prepend event type
        type_prefix = event_type.value.replace('_', ' ').title()
        
        return f"{type_prefix}: {first}"
    
    def _clean_description(self, text: str) -> str:
        """Clean and format description text"""
        # Remove excess whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Truncate if very long
        if len(text) > 500:
            text = text[:497] + '...'
        
        return text
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        keywords = []
        
        # Legal keywords
        legal_terms = [
            'arrest', 'detained', 'search', 'seizure', 'warrant',
            'custody', 'miranda', 'force', 'injury', 'complaint',
            'motion', 'hearing', 'order', 'dismiss', 'suppress'
        ]
        
        text_lower = text.lower()
        for term in legal_terms:
            if term in text_lower:
                keywords.append(term)
        
        return keywords[:10]
    
    def _calculate_confidence(
        self,
        text: str,
        event_date: date,
        event_type: EventType
    ) -> float:
        """Calculate confidence score for event extraction"""
        confidence = 0.5
        
        # Boost for specific event type
        if event_type != EventType.OTHER:
            confidence += 0.15
        
        # Boost for more context
        if len(text) > 100:
            confidence += 0.1
        
        # Boost for reasonable date (past, not too far)
        today = date.today()
        if event_date <= today and (today - event_date).days < 3650:
            confidence += 0.1
        
        # Boost for legal terminology
        legal_count = len(self._extract_keywords(text))
        confidence += min(0.15, legal_count * 0.03)
        
        return min(0.95, confidence)
    
    def _deduplicate_events(
        self,
        events: List[TimelineEvent]
    ) -> List[TimelineEvent]:
        """Remove duplicate events"""
        seen = set()
        unique = []
        
        for event in events:
            key = (
                event.event_date,
                event.event_time,
                event.event_type,
                event.description[:50] if event.description else ''
            )
            
            if key not in seen:
                seen.add(key)
                unique.append(event)
        
        return unique
    
    def merge_timelines(
        self,
        timelines: List[List[TimelineEvent]]
    ) -> List[TimelineEvent]:
        """Merge multiple timelines into one"""
        all_events = []
        for timeline in timelines:
            all_events.extend(timeline)
        
        # Deduplicate and sort
        events = self._deduplicate_events(all_events)
        events.sort(key=lambda e: (e.event_date, e.event_time or time(0, 0)))
        
        return events


def extract_timeline(
    text: str,
    source: str = ""
) -> List[Dict[str, Any]]:
    """
    Convenience function to extract timeline.
    
    Returns list of event dictionaries.
    """
    extractor = TimelineExtractor()
    events = extractor.extract_from_text(text, source)
    
    return [
        {
            'date': e.event_date.isoformat(),
            'time': e.event_time.isoformat() if e.event_time else None,
            'type': e.event_type.value,
            'title': e.title,
            'description': e.description,
            'actors': e.actors,
            'source': e.source_document,
            'confidence': e.confidence
        }
        for e in events
    ]
