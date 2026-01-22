# BarberX Legal Case Management Pro Suite

**Constitutional Rights Defense & Evidence Management Platform**

A comprehensive legal case management system designed for civil rights litigation, police misconduct cases, and constitutional violation analysis. Features batch PDF processing, body-worn camera (BWC) footage integration, multi-POV video synchronization, NJ Civil pleading generation, and AI-powered violation detection.

## 🎯 Mission

Built for BarberX to organize, analyze, and strengthen civil rights cases through systematic evidence management and constitutional violation identification.

## ⚖️ Core Features

### 📄 Document Management
- **Batch PDF Upload**: Drag-and-drop batch upload for court filings, police reports, discovery documents
- **OCR Processing**: Extract text from scanned documents using Tesseract/PyMuPDF
- **Smart Classification**: Auto-categorize documents (motions, orders, depositions, evidence)
- **Timeline Extraction**: Parse dates and events from documents to build case chronologies
- **Cross-Reference Engine**: Link related documents across cases
- **Full-Text Search**: SQLite FTS5 indexing for fast keyword search with NEAR/AND/OR support

### 🎥 Body-Worn Camera (BWC) Integration
- **Motorola Solutions Footprint**: Native support for Motorola BWC export formats
  - Parses `OfficerName_YYYYMMDDHHMI_DeviceID-Segment.mp4` filename format
  - Auto-extracts officer name, timestamp, device ID, segment number
- **Folder Scanner**: Scan `.bwc/` folders to discover all incident footage
- **Batch MP4 Upload**: Process multiple video files simultaneously
- **Multi-POV Sync**: Align and synchronize footage from multiple officers/angles
- **Audio Harmonization**: Normalize audio levels, reduce noise, enhance clarity
- **Timestamp Alignment**: Sync video timestamps across different devices
- **Frame-by-Frame Analysis**: Mark key moments for legal review
- **Overlap Detection**: Find footage where multiple officers recorded same moment

### ⚖️ NJ Civil Pleading Generator
- **Verified Complaint**: Auto-generate NJ Superior Court complaints from YAML facts
- **Motion for Interim Relief**: Emergency motion templates
- **Certification in Support**: Supporting certification documents
- **DOCX/TXT/HTML Output**: Court-ready formatted documents
- **Exhibit Merger**: Combine PDFs into exhibit packages
- **Jinja2 Templates**: Customizable pleading templates

### 🔍 Constitutional Analysis Engine
- **4th Amendment Scanner**: Unlawful search/seizure pattern detection
- **5th Amendment Analyzer**: Self-incrimination, due process violations
- **6th Amendment Review**: Right to counsel, speedy trial issues
- **8th Amendment Detection**: Excessive force, cruel punishment indicators
- **14th Amendment Analysis**: Equal protection, procedural due process
- **Brady Violation Detection**: Identify potential withheld exculpatory evidence
- **Giglio Material Flagging**: Police credibility issues and prior bad acts

### 📊 Case Intelligence
- **Liability Scoring**: Risk assessment for defendants
- **Pattern Recognition**: Cross-case analysis for systemic issues
- **Officer History Tracking**: Build profiles from multiple incidents
- **Department Analytics**: Identify training deficiencies, policy failures
- **Settlement Calculator**: Comparable case analysis for valuations

### 📁 Case Organization
- **Hierarchical Structure**: Case → Subcases → Documents → Evidence
- **Smart Tagging**: Auto-tag with parties, dates, violation types
- **Search & Discovery**: Full-text search across all documents
- **Docket Integration**: Link to court docket systems
- **OPRA Request Tracking**: NJ Open Public Records Act management

## 🏗️ Architecture

```
barberx-legal-suite/
├── backend/                 # FastAPI + Python
│   └── app/
│       ├── api/             # REST endpoints
│       │   ├── cases.py     # Case CRUD operations
│       │   ├── documents.py # PDF upload/processing
│       │   ├── evidence.py  # BWC footage management
│       │   ├── analysis.py  # Constitutional analysis
│       │   ├── exports.py   # Reports and exports
│       │   └── pleadings.py # NJ Civil document generation
│       ├── analyzers/       # Analysis engines
│       │   ├── constitutional.py
│       │   ├── timeline_extractor.py
│       │   └── __init__.py
│       ├── processors/      # Media processing
│       │   ├── pdf_processor.py
│       │   ├── bwc_processor.py
│       │   └── __init__.py
│       ├── generators/      # Document generation
│       │   ├── pleadings.py
│       │   └── __init__.py
│       ├── utils/           # Utilities
│       │   ├── bwc_scanner.py
│       │   └── __init__.py
│       ├── db/              # SQLAlchemy models
│       └── schemas/         # Pydantic validation
├── njcivil_terminal_tool/   # CLI for pleading generation
│   ├── njcivil.py           # Typer CLI tool
│   ├── facts.example.yml    # Example facts file
│   └── templates/           # Jinja2 templates
├── private-core-barber-cam/ # BWC footage storage
│   └── .bwc/                # Motorola export folders
└── docs/                    # Documentation
```

## 🚀 Quick Start

### Backend Setup
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### API Endpoints
- **API Base**: `http://localhost:8000`
- **Interactive Docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🔌 API Overview

### Cases
```
POST   /api/cases              # Create new case
GET    /api/cases              # List all cases
GET    /api/cases/{id}         # Get case details
PUT    /api/cases/{id}         # Update case
DELETE /api/cases/{id}         # Archive case
GET    /api/cases/{id}/timeline # Get case timeline
```

### Documents
```
POST   /api/documents/upload        # Batch PDF upload
POST   /api/documents/scan          # OCR processing
GET    /api/documents/{id}          # Get document
GET    /api/documents/{id}/text     # Get extracted text
POST   /api/documents/classify      # Auto-classify documents
POST   /api/documents/link          # Link related documents
```

### Evidence (BWC)
```
POST   /api/evidence/bwc/upload        # Upload BWC footage
POST   /api/evidence/bwc/sync          # Sync multiple POVs
POST   /api/evidence/bwc/harmonize     # Audio processing
GET    /api/evidence/bwc/{id}/frames   # Get key frames
POST   /api/evidence/bwc/scan-folder   # Scan local BWC folder
POST   /api/evidence/bwc/scan-root     # Scan .bwc root folder
POST   /api/evidence/bwc/import-folder # Import BWC from folder
GET    /api/evidence/bwc/timeline/{id} # Get sync group timeline
```

### Analysis
```
POST   /api/analysis/constitutional  # Run constitutional scan
POST   /api/analysis/liability       # Liability assessment
GET    /api/analysis/{id}/violations # Get violations found
POST   /api/analysis/pattern         # Cross-case patterns
POST   /api/analysis/officer/{badge} # Officer history
```

### Pleadings (NJ Civil)
```
GET    /api/v1/pleadings/templates      # List available templates
POST   /api/v1/pleadings/generate       # Generate pleading from facts
GET    /api/v1/pleadings/download/{fn}  # Download generated document
POST   /api/v1/pleadings/extract-text   # Extract text from PDFs/DOCX
POST   /api/v1/pleadings/build-index    # Build FTS5 search index
POST   /api/v1/pleadings/search         # Full-text search filings
POST   /api/v1/pleadings/grep           # Regex search without index
POST   /api/v1/pleadings/merge-exhibits # Merge PDFs for exhibits
POST   /api/v1/pleadings/upload-facts   # Upload YAML facts file
GET    /api/v1/pleadings/facts-template # Get example facts YAML
```

### Exports
```
POST   /api/exports/timeline         # Generate timeline PDF
POST   /api/exports/violations       # Violations report
POST   /api/exports/evidence-binder  # Compiled evidence
POST   /api/exports/settlement       # Settlement analysis
```

## 📋 Constitutional Violation Categories

| Amendment | Violation Type | Detection Triggers |
|-----------|---------------|-------------------|
| 4th | Unlawful Search | Warrant issues, consent problems, scope exceeded |
| 4th | Unlawful Seizure | Detention without RS, arrest without PC |
| 4th | Excessive Force | Force disproportionate to threat |
| 5th | Miranda Violation | Custodial interrogation without warnings |
| 5th | Due Process | Procedural failures, evidence destruction |
| 6th | Right to Counsel | Questioning after invocation |
| 8th | Cruel Punishment | Conditions of confinement |
| 14th | Equal Protection | Discriminatory enforcement patterns |

## 🔒 Security & Compliance

- **Encryption**: AES-256 for data at rest
- **Access Control**: Role-based permissions
- **Audit Trail**: Complete action logging
- **CJIS Compliance**: Law enforcement data handling
- **Attorney-Client Privilege**: Protected document marking
- **Chain of Custody**: Evidence integrity tracking

## 🛠️ Dependencies

### Core
- FastAPI 0.104+
- SQLAlchemy 2.0+
- Pydantic 2.0+
- Python 3.11+

### Document Processing
- PyMuPDF (fitz)
- Tesseract OCR
- pdf2image
- python-magic

### Video/Audio Processing
- FFmpeg
- MoviePy
- librosa
- pydub

### Analysis
- spaCy (NLP)
- transformers (BERT)
- scikit-learn
- pandas

## ⚠️ Legal Disclaimer

This software is a tool to assist legal professionals. It does not provide legal advice and should not be relied upon as a substitute for professional legal judgment. All analysis results should be verified by qualified attorneys.

## 📜 License

Proprietary - BarberX Legal Services
Copyright © 2024-2026 All Rights Reserved
