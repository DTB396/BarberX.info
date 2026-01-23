# BarberX Legal AI Suite - System Architecture

**Version:** 4.0.0-dataprocessing  
**Last Updated:** January 23, 2026  
**Classification:** Internal Technical Documentation

---

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [System Components](#system-components)
3. [Data Architecture](#data-architecture)
4. [API Architecture](#api-architecture)
5. [AI/ML Architecture](#aiml-architecture)
6. [Security Architecture](#security-architecture)
7. [Infrastructure Architecture](#infrastructure-architecture)
8. [Integration Architecture](#integration-architecture)
9. [Deployment Architecture](#deployment-architecture)
10. [Performance & Scalability](#performance--scalability)

---

## 🏗️ Architecture Overview

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE LAYER                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Web App   │  │  Mobile App │  │  Desktop    │  │  API Client │    │
│  │  (React)    │  │  (Native)   │  │  (Electron) │  │  (CLI/SDK)  │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
└─────────┼─────────────────┼─────────────────┼─────────────────┼─────────┘
          │                 │                 │                 │
          └─────────────────┴─────────────────┴─────────────────┘
                                      │
┌─────────────────────────────────────┼─────────────────────────────────────┐
│                         API GATEWAY LAYER                                  │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │  FastAPI Application Server (Uvicorn/Gunicorn)                     │  │
│  │  - Authentication & Authorization (JWT, OAuth2)                    │  │
│  │  - Rate Limiting & Throttling                                      │  │
│  │  - Request Validation & Response Formatting                        │  │
│  │  - CORS, Security Headers, Error Handling                          │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────┼────────────────────────────────────┐
│                      APPLICATION LAYER (FastAPI Routers)                 │
│  ┌────────┬────────┬────────┬────────┬────────┬────────┬────────┐      │
│  │ Cases  │ Docs   │Evidence│Analysis│Premium │  AI    │ Local  │      │
│  │        │        │  &BWC  │        │ Legal  │ Tools  │  AI    │      │
│  └────────┴────────┴────────┴────────┴────────┴────────┴────────┘      │
│  ┌────────┬────────┬────────┬────────┬────────┬────────┬────────┐      │
│  │ Legal  │  NLP   │E-Disc. │AV Foren│  Viz   │Medical │Privacy │      │
│  │Research│Intel   │Advanced│Advanced│Advanced│Analysis│Advanced│      │
│  └────────┴────────┴────────┴────────┴────────┴────────┴────────┘      │
│  ┌────────┬────────┬────────┬────────┬────────┬────────┬────────┐      │
│  │  Data  │  Firm  │Subscrip│Pleading│ Batch  │ Audio  │Settings│      │
│  │Process │  Mgmt  │ tions  │        │ Upload │Analysis│        │      │
│  └────────┴────────┴────────┴────────┴────────┴────────┴────────┘      │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────┼────────────────────────────────────┐
│                         BUSINESS LOGIC LAYER                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │ Case Management │  │ Document        │  │ Evidence        │          │
│  │ - CRUD ops      │  │ Processing      │  │ Chain-of-Custody│          │
│  │ - Workflows     │  │ - OCR           │  │ - Integrity     │          │
│  │ - Assignments   │  │ - Classification│  │ - Analysis      │          │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │ AI/ML           │  │ Data Quality    │  │ Compliance      │          │
│  │ - Predictions   │  │ - Validation    │  │ - Privilege     │          │
│  │ - Analysis      │  │ - Profiling     │  │ - Privacy       │          │
│  │ - Explainability│  │ - Monitoring    │  │ - Audit Trails  │          │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘          │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────┼────────────────────────────────────┐
│                          DATA ACCESS LAYER                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │ ORM (SQLAlchemy)│  │ Query Builder   │  │ Transaction Mgmt│          │
│  │ - Models        │  │ - Optimized SQL │  │ - ACID          │          │
│  │ - Relationships │  │ - Indexing      │  │ - Rollback      │          │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘          │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────┼────────────────────────────────────┐
│                         DATA STORAGE LAYER                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │ PostgreSQL      │  │ DuckDB          │  │ Elasticsearch   │          │
│  │ (Primary DB)    │  │ (Analytics)     │  │ (Search/E-Disc) │          │
│  │ - Case data     │  │ - Fast OLAP     │  │ - Full-text     │          │
│  │ - Metadata      │  │ - Aggregations  │  │ - Document index│          │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │ Redis           │  │ MinIO/S3        │  │ ChromaDB        │          │
│  │ (Cache/Queue)   │  │ (Object Storage)│  │ (Vector DB)     │          │
│  │ - Sessions      │  │ - Files         │  │ - Embeddings    │          │
│  │ - Celery tasks  │  │ - Videos        │  │ - Semantic      │          │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘          │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────┼────────────────────────────────────┐
│                        PROCESSING LAYER                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │ Celery Workers  │  │ Apache Spark    │  │ ML Models       │          │
│  │ - Async tasks   │  │ - Big data      │  │ - Local (Ollama)│          │
│  │ - Job queues    │  │ - Batch jobs    │  │ - Cloud (OpenAI)│          │
│  │ - Scheduling    │  │ - ETL           │  │ - Custom trained│          │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │ FFmpeg          │  │ Whisper/Vosk    │  │ Computer Vision │          │
│  │ - Video proc.   │  │ - Transcription │  │ - Face detect   │          │
│  │ - Encoding      │  │ - Speech-to-text│  │ - OCR           │          │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘          │
└───────────────────────────────────────────────────────────────────────────┘
```

### Architecture Principles

1. **Modularity**: Loosely coupled components, clear boundaries
2. **Scalability**: Horizontal scaling capability at each layer
3. **Security**: Defense in depth, encryption everywhere
4. **Privacy**: Data minimization, local processing option
5. **Reliability**: Fault tolerance, graceful degradation
6. **Performance**: Optimized at every layer
7. **Maintainability**: Clean code, comprehensive docs

---

## 🧩 System Components

### Frontend Components

#### Web Application (React/TypeScript)
- **Framework**: React 18+ with TypeScript
- **State Management**: Redux Toolkit or Zustand
- **UI Library**: Material-UI or Ant Design
- **Features**:
  - Case management dashboard
  - Document viewer (PDF.js, video player)
  - Evidence timeline and synchronization
  - Real-time collaboration
  - Responsive design

#### Mobile Application
- **Framework**: React Native or Flutter
- **Platforms**: iOS, Android
- **Features**:
  - Field evidence collection
  - BWC footage upload
  - Offline mode
  - Push notifications

### Backend Components

#### FastAPI Application Server
- **Framework**: FastAPI 0.108+
- **ASGI Server**: Uvicorn (dev), Gunicorn + Uvicorn workers (production)
- **Features**:
  - RESTful API (20+ routers, 180+ endpoints)
  - WebSocket support (real-time updates)
  - Automatic OpenAPI/Swagger documentation
  - Request validation (Pydantic)
  - Async/await throughout

#### API Routers (20 modules)

**Core Routers**:
1. `cases.py` - Case CRUD, workflows
2. `documents.py` - Document management, OCR
3. `evidence.py` - Evidence tracking, chain-of-custody
4. `analysis.py` - Constitutional violation analysis
5. `exports.py` - Report generation, data export
6. `settings.py` - User preferences, system config
7. `pleadings.py` - Pleading generation, templates
8. `ai.py` - AI/GPT integration
9. `bwc_analysis.py` - BWC footage analysis
10. `batch_upload.py` - Bulk document processing
11. `audio_analysis.py` - Audio forensics

**Premium Routers**:
12. `premium_legal.py` - E-discovery, depositions, strategy
13. `firm_management.py` - Conflicts, billing, research

**Extended AI Routers** (7 modules):
14. `legal_research.py` - Case law, citations, Shepardizing
15. `nlp_intelligence.py` - NLP, summarization, contracts
16. `ediscovery_advanced.py` - Tika, Elasticsearch, dedup
17. `av_forensics_advanced.py` - Whisper, speaker ID, video analysis
18. `visualization_advanced.py` - Plotly charts, maps, reports
19. `medical_analysis.py` - Medical NER, injury assessment
20. `privacy_advanced.py` - PII detection, redaction

**Local AI Routers** (2 modules):
21. `local_ai.py` - Local LLM, embeddings, semantic search
22. `local_av_processing.py` - Offline transcription, face detection

**Data Processing Router** (NEW):
23. `data_processing.py` - DuckDB, Polars, ML predictions, data quality

---

## 💾 Data Architecture

### Database Schema

#### Primary Database: PostgreSQL

**Core Tables**:
```sql
-- Cases
CREATE TABLE cases (
    id UUID PRIMARY KEY,
    case_number VARCHAR(50) UNIQUE NOT NULL,
    title VARCHAR(500) NOT NULL,
    client_name VARCHAR(200),
    case_type VARCHAR(50), -- civil_rights, excessive_force, etc.
    status VARCHAR(50), -- active, closed, archived
    assigned_attorney_id UUID,
    filed_date DATE,
    statute_of_limitations DATE,
    tier INTEGER, -- Data classification (1-4)
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);

-- Documents
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    case_id UUID REFERENCES cases(id),
    document_type VARCHAR(100), -- pleading, evidence, correspondence
    file_path VARCHAR(1000),
    file_name VARCHAR(500),
    file_size BIGINT,
    file_hash VARCHAR(64), -- SHA-256
    mime_type VARCHAR(100),
    classification_tier INTEGER, -- 1=Public, 2=Confidential, 3=Privileged, 4=PPI
    privilege_status VARCHAR(50), -- privileged, non-privileged, partially_privileged
    ocr_text TEXT,
    metadata JSONB,
    uploaded_by UUID,
    uploaded_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_case_id (case_id),
    INDEX idx_classification (classification_tier),
    INDEX idx_file_hash (file_hash)
);

-- Evidence
CREATE TABLE evidence (
    id UUID PRIMARY KEY,
    case_id UUID REFERENCES cases(id),
    evidence_type VARCHAR(100), -- bwc_video, audio, photo, document
    source VARCHAR(200), -- officer name, camera ID, etc.
    file_path VARCHAR(1000),
    file_hash VARCHAR(64),
    collected_date TIMESTAMP,
    chain_of_custody JSONB, -- Array of custody transfers
    analysis_results JSONB,
    classification_tier INTEGER,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_case_evidence (case_id, evidence_type),
    INDEX idx_hash (file_hash)
);

-- BWC Footage
CREATE TABLE bwc_footage (
    id UUID PRIMARY KEY,
    evidence_id UUID REFERENCES evidence(id),
    case_id UUID REFERENCES cases(id),
    camera_id VARCHAR(100),
    officer_name VARCHAR(200),
    recording_start TIMESTAMP,
    recording_end TIMESTAMP,
    duration_seconds INTEGER,
    video_file_path VARCHAR(1000),
    transcription TEXT,
    transcription_timestamps JSONB, -- Word-level timestamps
    constitutional_violations JSONB, -- Detected violations
    synchronized_with UUID[], -- Array of other footage IDs
    metadata JSONB,
    INDEX idx_case_bwc (case_id),
    INDEX idx_officer (officer_name)
);

-- Constitutional Violations
CREATE TABLE violations (
    id UUID PRIMARY KEY,
    case_id UUID REFERENCES cases(id),
    evidence_id UUID REFERENCES evidence(id),
    violation_type VARCHAR(100), -- 4th_amendment, 5th_amendment, etc.
    severity VARCHAR(50), -- minor, moderate, severe, egregious
    description TEXT,
    timestamp_in_evidence INTERVAL, -- Time within video/audio
    confidence_score FLOAT, -- AI detection confidence
    reviewed_by_attorney BOOLEAN DEFAULT FALSE,
    attorney_notes TEXT,
    supporting_caselaw JSONB, -- Array of case citations
    created_at TIMESTAMP DEFAULT NOW()
);

-- Users & Authentication
CREATE TABLE users (
    id UUID PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(200) UNIQUE NOT NULL,
    password_hash VARCHAR(255),
    role VARCHAR(50), -- admin, attorney, paralegal, investigator
    firm_id UUID,
    access_tiers INTEGER[], -- Array of allowed tiers [1,2,3]
    mfa_enabled BOOLEAN DEFAULT FALSE,
    last_login TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Audit Logs
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    action VARCHAR(100), -- view, create, update, delete, export
    resource_type VARCHAR(50), -- case, document, evidence
    resource_id UUID,
    data_tier INTEGER, -- Tier of data accessed
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP DEFAULT NOW(),
    INDEX idx_user_audit (user_id, timestamp),
    INDEX idx_resource_audit (resource_type, resource_id),
    INDEX idx_tier_audit (data_tier, timestamp)
);
```

#### Analytics Database: DuckDB

**Purpose**: Fast analytical queries, aggregations, OLAP  
**Data**: Imported from PostgreSQL for analysis  
**Query Speed**: 10-100x faster than pandas  

**Use Cases**:
- Case statistics and trends
- Performance metrics
- Time-series analysis
- Complex aggregations

#### Search Engine: Elasticsearch

**Indices**:
- `documents` - Full-text search across all documents
- `evidence` - Evidence metadata and analysis results
- `transcriptions` - BWC and audio transcriptions

**Features**:
- Full-text search (millisecond response)
- Fuzzy matching
- Faceted search (filter by type, date, case, etc.)
- Highlighting

#### Vector Database: ChromaDB

**Purpose**: Semantic search, AI embeddings  
**Storage**: Local files (no cloud dependency)  
**Collections**:
- `case_documents` - Document embeddings
- `legal_research` - Case law embeddings
- `evidence_text` - Evidence transcription embeddings

**Features**:
- Semantic similarity search
- Hybrid search (keyword + semantic)
- Metadata filtering
- Persistent storage

#### Object Storage: MinIO (or AWS S3)

**Buckets**:
- `documents` - PDF files, Office docs
- `evidence-video` - BWC footage, surveillance video
- `evidence-audio` - Audio recordings
- `evidence-images` - Photos, screenshots
- `reports` - Generated reports

**Features**:
- Versioning enabled
- Encryption at rest (AES-256)
- Access logs
- Lifecycle policies (auto-archive old data)

#### Cache & Queue: Redis

**Use Cases**:
- Session storage (user sessions, JWT tokens)
- API rate limiting
- Celery task queue
- Real-time pub/sub (notifications)
- Cache (frequently accessed data)

**Data Structures**:
- Strings: Session tokens, rate limit counters
- Hashes: User session data
- Lists: Task queues
- Pub/Sub: Real-time notifications
- Sorted Sets: Leaderboards, recent activity

---

## 🔌 API Architecture

### API Design Principles

1. **RESTful**: Resource-oriented URLs, HTTP methods
2. **Versioned**: `/api/v1/...` for backward compatibility
3. **Consistent**: Standard response format, error codes
4. **Documented**: Auto-generated OpenAPI/Swagger
5. **Secure**: Authentication, authorization, rate limiting
6. **Performant**: Pagination, filtering, caching

### API Structure

```
/api/v1/
├── /cases/                    # Case management
│   ├── GET /cases              # List all cases
│   ├── POST /cases             # Create case
│   ├── GET /cases/{id}         # Get case details
│   ├── PUT /cases/{id}         # Update case
│   ├── DELETE /cases/{id}      # Delete case
│   └── GET /cases/{id}/timeline # Case timeline
│
├── /documents/                # Document management
│   ├── GET /documents          # List documents
│   ├── POST /documents/upload  # Upload document
│   ├── GET /documents/{id}     # Get document
│   ├── POST /documents/{id}/ocr # Run OCR
│   └── POST /documents/{id}/classify # Classify (tier, privilege)
│
├── /evidence/                 # Evidence management
│   ├── GET /evidence           # List evidence
│   ├── POST /evidence/upload   # Upload evidence
│   ├── GET /evidence/{id}      # Get evidence
│   ├── POST /evidence/{id}/analyze # Analyze evidence
│   └── GET /evidence/{id}/chain-of-custody # Get custody log
│
├── /analysis/                 # AI analysis
│   ├── POST /analysis/constitutional # Detect violations
│   ├── POST /analysis/liability # Assess liability
│   ├── POST /analysis/damages  # Calculate damages
│   └── POST /analysis/settlement # Settlement valuation
│
├── /data-processing/          # Data processing (NEW)
│   ├── POST /data-quality/validate # Validate data quality
│   ├── POST /data-quality/profile # Profile dataset
│   ├── POST /bigdata/analyze-duckdb # DuckDB analytics
│   ├── POST /bigdata/analyze-polars # Polars DataFrame
│   ├── POST /ml/predict-outcome # ML predictions
│   ├── POST /ml/detect-bias    # Bias detection
│   └── POST /statistics/analyze # Statistical analysis
│
├── /local-ai/                 # Local AI (offline)
│   ├── POST /chat              # Chat with local LLM
│   ├── POST /index-document    # Index in local vector DB
│   ├── POST /semantic-search   # Semantic search locally
│   └── POST /summarize-local   # Local summarization
│
└── /legal-research/           # Legal research
    ├── GET /search-caselaw     # Search court opinions
    ├── POST /extract-citations # Extract citations
    ├── POST /shepardize        # Shepardize citation
    └── GET /search-statutes    # Search statutes
```

### Authentication & Authorization

**Authentication**: JWT (JSON Web Tokens)
```
Authorization: Bearer <JWT_TOKEN>
```

**Token Structure**:
```json
{
  "sub": "user_id",
  "role": "attorney",
  "access_tiers": [1, 2, 3],
  "firm_id": "firm_uuid",
  "exp": 1706025600
}
```

**Authorization**: Role-Based + Attribute-Based Access Control

**Roles**:
- `admin` - Full access
- `attorney` - Access to assigned cases (Tier 1-3)
- `paralegal` - Limited access (Tier 1-2)
- `investigator` - Read-only (Tier 1)

**Permissions Checked**:
- User role
- Data tier classification
- Case assignment
- Firm membership

### Rate Limiting

**Tiers**:
- Anonymous: 10 requests/minute
- Authenticated: 100 requests/minute
- Premium: 1000 requests/minute
- Enterprise: Unlimited

**Implementation**: Redis-backed sliding window

### Response Format

**Success**:
```json
{
  "success": true,
  "data": { ... },
  "metadata": {
    "timestamp": "2026-01-23T12:00:00Z",
    "version": "4.0.0",
    "processing_time_ms": 45
  }
}
```

**Error**:
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid case_id format",
    "details": { ... }
  },
  "metadata": {
    "timestamp": "2026-01-23T12:00:00Z",
    "request_id": "uuid"
  }
}
```

---

## 🤖 AI/ML Architecture

### Model Repository Structure

```
models/
├── local/                      # Local models (Ollama, etc.)
│   ├── llama3.2-8b/
│   ├── mistral-7b/
│   ├── whisper-medium/
│   └── yolov8n/
│
├── cloud/                      # Cloud model configs
│   ├── openai/
│   ├── anthropic/
│   └── cohere/
│
└── custom/                     # Custom-trained models
    ├── legal-bert-ner/
    ├── constitutional-classifier/
    ├── settlement-predictor/
    └── bias-detector/
```

### Model Serving

**Local Models** (via Ollama):
- Endpoint: `http://localhost:11434`
- Models: Llama, Mistral, Phi-3, etc.
- Zero cost, full privacy

**Cloud Models** (via API):
- OpenAI GPT-4/GPT-5
- Anthropic Claude
- Cohere models

**Custom Models** (MLflow):
- Trained on legal data
- Versioned, tracked
- A/B testing capability

### ML Pipeline

```
Data → Preprocessing → Feature Engineering → Model Training → Validation → Deployment → Monitoring
  ↓          ↓               ↓                     ↓             ↓            ↓            ↓
Clean      Normalize      Extract           Train/Test     Bias Check   Serve API   Drift Detect
Transform  Tokenize       Features          Cross-val      Explainability Version    Retrain
Validate   Augment        Selection         Tune           Accuracy      A/B Test    Alert
```

### Model Explainability

**Methods**:
- SHAP (SHapley Additive exPlanations) - Feature importance
- LIME (Local Interpretable Model-agnostic Explanations)
- Attention visualization (for transformers)
- Decision tree surrogates

**Output**:
- Feature importance rankings
- Prediction explanations
- Counterfactual examples
- Confidence intervals

---

## 🔒 Security Architecture

### Defense in Depth

```
Layer 1: Network Security
  ├── Firewall (allow-list only)
  ├── DDoS protection
  ├── VPN for remote access
  └── Network segmentation

Layer 2: Application Security
  ├── Authentication (JWT, OAuth2)
  ├── Authorization (RBAC + ABAC)
  ├── Input validation (Pydantic)
  ├── SQL injection prevention (ORMs)
  ├── XSS prevention (CSP headers)
  └── CSRF protection

Layer 3: Data Security
  ├── Encryption at rest (AES-256)
  ├── Encryption in transit (TLS 1.3)
  ├── Field-level encryption (PPI)
  ├── Key management (HSM/KMS)
  └── Data masking

Layer 4: Operational Security
  ├── Audit logging (all actions)
  ├── Intrusion detection
  ├── Vulnerability scanning
  ├── Security monitoring
  └── Incident response

Layer 5: Physical Security
  ├── Access control (badge, biometric)
  ├── Surveillance
  ├── Secure disposal
  └── Environmental controls
```

### Encryption Keys

**Key Hierarchy**:
- Master Key (HSM-protected)
  - Data Encryption Keys (rotated every 90 days)
    - Document encryption
    - Database encryption
    - Backup encryption

**Key Storage**:
- Production: Hardware Security Module (HSM) or Azure Key Vault
- Development: Environment variables (never in code)

### Security Monitoring

**Tools**:
- SIEM (Security Information and Event Management)
- IDS/IPS (Intrusion Detection/Prevention)
- Log aggregation (ELK stack or Splunk)
- Vulnerability scanner (Nessus, OpenVAS)

**Alerts**:
- Failed login attempts (>5 in 5 minutes)
- Privilege escalation
- After-hours access to Tier 3/4 data
- Large data exports
- Unusual query patterns

---

## 🌐 Infrastructure Architecture

### Development Environment

```
Developer Laptop
├── Frontend (localhost:3000)
├── Backend (localhost:8000)
├── PostgreSQL (localhost:5432)
├── Redis (localhost:6379)
├── Ollama (localhost:11434)
└── ChromaDB (local files)
```

### Production Environment (Cloud)

```
Load Balancer (HTTPS)
   ↓
Application Servers (3+ instances)
   ├── FastAPI app (Gunicorn + Uvicorn)
   ├── Celery workers
   └── Health checks
   ↓
Database Cluster
   ├── PostgreSQL (Primary + 2 Replicas)
   ├── Redis (Cluster mode)
   └── Elasticsearch (3-node cluster)
   ↓
Object Storage
   └── MinIO or S3 (multi-region replication)
```

### Deployment Options

**Option 1: Cloud (AWS, Azure, GCP)**
- Managed services (RDS, ElastiCache, S3)
- Auto-scaling
- High availability
- Geographic distribution

**Option 2: On-Premise**
- Full control
- Data sovereignty
- Hardware costs
- Manual scaling

**Option 3: Hybrid**
- Sensitive data on-premise (Tier 3/4)
- Public data in cloud (Tier 1)
- Best of both worlds

### Monitoring & Observability

**Metrics**:
- Application: Request rate, latency, errors
- Infrastructure: CPU, memory, disk, network
- Business: Cases processed, documents analyzed

**Logging**:
- Application logs (structured JSON)
- Access logs (who, what, when)
- Security logs (auth, violations)
- Audit logs (compliance)

**Tracing**:
- Distributed tracing (OpenTelemetry)
- Request flow visualization
- Performance bottleneck identification

---

## 📈 Performance & Scalability

### Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| API Response Time (p95) | <200ms | TBD |
| Database Query Time (p95) | <50ms | TBD |
| Document OCR | <5 sec/page | TBD |
| Video Transcription | Real-time | TBD |
| Search Results | <100ms | TBD |
| Concurrent Users | 1000+ | TBD |

### Scalability Strategy

**Horizontal Scaling**:
- Application servers (stateless)
- Celery workers (task queues)
- Database read replicas

**Vertical Scaling**:
- Database primary (CPU, RAM)
- ML model servers (GPU)

**Caching Strategy**:
- Redis L1 cache (hot data)
- CDN (static assets)
- Database query cache

### Performance Optimization

**Database**:
- Indexing (composite indexes on common queries)
- Query optimization (EXPLAIN ANALYZE)
- Connection pooling
- Read replicas (read-heavy queries)

**Application**:
- Async/await throughout
- Database connection pooling
- Lazy loading
- Pagination (limit large result sets)

**Caching**:
- User sessions (Redis)
- API responses (Redis, 5-60 minute TTL)
- Static assets (CDN, long TTL)

---

## 🔄 Integration Architecture

### External Integrations

**Legal Data Sources**:
- CourtListener API (case law)
- PACER (federal court dockets)
- State court APIs
- Statute databases

**AI/ML Services**:
- OpenAI API (GPT-4/GPT-5)
- Anthropic API (Claude)
- Whisper API (transcription)
- Local Ollama (offline LLMs)

**Document Processing**:
- Apache Tika (document parsing)
- Tesseract OCR
- FFmpeg (media processing)

**Communication**:
- Email (SMTP)
- SMS (Twilio)
- Push notifications (Firebase)

### Integration Patterns

**Synchronous** (REST APIs):
- Legal research (CourtListener)
- AI predictions (OpenAI)

**Asynchronous** (Message Queue):
- Document processing (Celery + Redis)
- Video transcription (Celery + Redis)
- Batch jobs (Apache Airflow)

**Event-Driven** (Pub/Sub):
- Real-time notifications (Redis Pub/Sub)
- Data pipelines (Apache Kafka)

---

## 📚 Technology Stack Summary

### Frontend
- React 18+, TypeScript
- Material-UI or Ant Design
- Redux Toolkit / Zustand
- PDF.js, Video.js

### Backend
- FastAPI 0.108+
- Python 3.10+
- Uvicorn / Gunicorn
- Pydantic, SQLAlchemy

### Databases
- PostgreSQL 15+ (primary)
- DuckDB (analytics)
- Elasticsearch 8+ (search)
- Redis 7+ (cache, queue)
- ChromaDB (vectors)

### Storage
- MinIO / AWS S3 (objects)
- Local filesystem (development)

### AI/ML
- Ollama (local LLMs)
- OpenAI API (cloud)
- scikit-learn, XGBoost
- Whisper, YOLOv8

### Data Processing
- Polars (DataFrames)
- Apache Spark (big data)
- Great Expectations (quality)
- Pandas, NumPy

### DevOps
- Docker, Docker Compose
- Kubernetes (optional)
- GitHub Actions (CI/CD)
- Prometheus + Grafana (monitoring)

---

**Document Version**: 1.0  
**Last Updated**: January 23, 2026  
**Maintained By**: Technical Architecture Team  
**Review Cycle**: Quarterly
