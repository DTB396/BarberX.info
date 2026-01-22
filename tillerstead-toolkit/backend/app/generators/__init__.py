"""
BarberX Legal Case Management Pro Suite
Generators Module
Document Generation, Pleadings, Exports
"""

from .pleadings import (
    NJCivilPleadingsGenerator,
    PleadingType,
    PleadingResult,
    FilingsSearchEngine,
    SearchResult,
    ExhibitMerger,
    generate_complaint,
    generate_motion,
    generate_certification,
    search_filings
)

__all__ = [
    # Pleadings Generator
    'NJCivilPleadingsGenerator',
    'PleadingType',
    'PleadingResult',
    
    # Search Engine
    'FilingsSearchEngine',
    'SearchResult',
    
    # Exhibit Tools
    'ExhibitMerger',
    
    # Convenience Functions
    'generate_complaint',
    'generate_motion',
    'generate_certification',
    'search_filings',
]
