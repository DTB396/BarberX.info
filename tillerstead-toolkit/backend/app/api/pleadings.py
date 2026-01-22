"""
BarberX Legal Case Management Pro Suite
Pleadings API Router
NJ Civil Document Generation Endpoints
"""
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import json

router = APIRouter(prefix="/pleadings", tags=["pleadings"])


# ============================================================================
# Pydantic Schemas
# ============================================================================

class CourtInfo(BaseModel):
    """Court information"""
    county: str = Field(..., description="NJ County (e.g., 'Atlantic')")
    division: str = Field(default="LAW DIVISION - CIVIL PART")


class PlaintiffInfo(BaseModel):
    """Plaintiff information"""
    name: str
    address: str
    city: str
    phone: Optional[str] = None
    email: Optional[str] = None


class CaseInfo(BaseModel):
    """Case information"""
    docket_new: str = Field(default="__________ (NEW ACTION)")
    verification_date: Optional[str] = None
    motion_date: Optional[str] = None
    related_actions_statement: Optional[str] = None


class FactsInfo(BaseModel):
    """Case facts"""
    stop_date: Optional[str] = None
    release_time: Optional[str] = None
    vehicle_year: Optional[int] = None
    vehicle_make: Optional[str] = None
    vehicle_model: Optional[str] = None
    vehicle_vin: Optional[str] = None
    vehicle_plate: Optional[str] = None
    
    class Config:
        extra = "allow"  # Allow additional facts


class DefendantsInfo(BaseModel):
    """Defendants information"""
    caption_block: str = Field(..., description="Full defendants caption block")


class PleadingRequest(BaseModel):
    """Request to generate a pleading"""
    pleading_type: str = Field(
        ...,
        description="Type: verified_complaint, motion_interim_relief, certification_support"
    )
    court: CourtInfo
    plaintiff: PlaintiffInfo
    defendants: DefendantsInfo
    case: CaseInfo
    facts: FactsInfo
    output_format: str = Field(default="docx", description="docx, txt, or html")


class PleadingResponse(BaseModel):
    """Response from pleading generation"""
    success: bool
    pleading_type: str
    content: Optional[str] = None
    download_url: Optional[str] = None
    word_count: Optional[int] = None
    generated_at: str
    error: Optional[str] = None


class TemplateInfo(BaseModel):
    """Template information"""
    type: str
    template: str
    available: bool
    description: str


class SearchRequest(BaseModel):
    """Full-text search request"""
    query: str = Field(..., description="FTS5 query (supports NEAR, AND, OR)")
    limit: int = Field(default=20, ge=1, le=100)


class SearchResultItem(BaseModel):
    """Search result item"""
    document: str
    snippet: str
    score: float
    context: Optional[str] = None


class GrepRequest(BaseModel):
    """Regex search request"""
    pattern: str
    literal: bool = Field(default=False, description="Treat as literal text")
    context_chars: int = Field(default=80, ge=20, le=500)
    max_hits: int = Field(default=50, ge=1, le=200)


class ExhibitMergeRequest(BaseModel):
    """Request to merge exhibits"""
    exhibit_pdfs: List[str] = Field(..., description="List of PDF paths to merge")
    output_name: str = Field(default="Exhibits.pdf")


class ExtractRequest(BaseModel):
    """Request to extract text from filings"""
    input_dir: str
    output_dir: str
    patterns: List[str] = Field(default=["*.pdf", "*.docx"])
    overwrite: bool = False


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/templates", response_model=List[TemplateInfo])
async def list_templates():
    """List available pleading templates"""
    from ..generators.pleadings import NJCivilPleadingsGenerator
    
    generator = NJCivilPleadingsGenerator()
    return generator.get_available_templates()


@router.post("/generate", response_model=PleadingResponse)
async def generate_pleading(request: PleadingRequest):
    """
    Generate a NJ Civil pleading document.
    
    Supports:
    - verified_complaint: Full complaint with counts
    - motion_interim_relief: Motion for emergency/interim relief
    - certification_support: Supporting certification
    """
    from ..generators.pleadings import NJCivilPleadingsGenerator, PleadingType
    
    # Map string to enum
    type_map = {
        "verified_complaint": PleadingType.VERIFIED_COMPLAINT,
        "motion_interim_relief": PleadingType.MOTION_INTERIM_RELIEF,
        "certification_support": PleadingType.CERTIFICATION_SUPPORT,
    }
    
    pleading_type = type_map.get(request.pleading_type)
    if not pleading_type:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown pleading type: {request.pleading_type}. "
                   f"Available: {list(type_map.keys())}"
        )
    
    # Build facts dictionary
    facts = {
        "court": request.court.model_dump(),
        "plaintiff": request.plaintiff.model_dump(),
        "defendants": request.defendants.model_dump(),
        "case": request.case.model_dump(),
        "facts": request.facts.model_dump(),
    }
    
    # Generate output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("./exports/pleadings")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ext = request.output_format
    output_path = output_dir / f"{request.pleading_type}_{timestamp}.{ext}"
    
    # Generate pleading
    generator = NJCivilPleadingsGenerator()
    result = generator.generate_pleading(
        pleading_type,
        facts,
        output_path,
        request.output_format
    )
    
    if not result.success:
        raise HTTPException(status_code=500, detail=result.error)
    
    return PleadingResponse(
        success=True,
        pleading_type=request.pleading_type,
        content=result.content[:5000] if result.content else None,  # Truncate preview
        download_url=f"/api/v1/pleadings/download/{output_path.name}",
        word_count=result.metadata.get("word_count"),
        generated_at=result.metadata.get("generated_at", datetime.now().isoformat()),
        error=None
    )


@router.get("/download/{filename}")
async def download_pleading(filename: str):
    """Download a generated pleading document"""
    file_path = Path("./exports/pleadings") / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine media type
    media_types = {
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".txt": "text/plain",
        ".html": "text/html",
        ".pdf": "application/pdf"
    }
    media_type = media_types.get(file_path.suffix, "application/octet-stream")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_type
    )


@router.post("/generate-from-case/{case_id}", response_model=PleadingResponse)
async def generate_from_case(
    case_id: int,
    pleading_type: str,
    output_format: str = "docx"
):
    """
    Generate pleading from existing case data.
    
    Pulls facts from case record in database.
    """
    # This would fetch from database - placeholder for now
    raise HTTPException(
        status_code=501,
        detail="Database integration pending. Use /generate with manual facts."
    )


@router.post("/extract-text")
async def extract_text_from_filings(request: ExtractRequest):
    """
    Extract text from PDF/DOCX filings for indexing.
    
    Processes all matching files in input_dir and saves
    extracted text to output_dir.
    """
    from ..generators.pleadings import FilingsSearchEngine
    
    input_dir = Path(request.input_dir)
    output_dir = Path(request.output_dir)
    
    if not input_dir.exists():
        raise HTTPException(status_code=404, detail=f"Input directory not found: {input_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = FilingsSearchEngine()
    
    results = {"processed": [], "failed": []}
    
    for pattern in request.patterns:
        for file_path in input_dir.rglob(pattern):
            out_txt = output_dir / (file_path.name + ".txt")
            
            if out_txt.exists() and not request.overwrite:
                continue
            
            try:
                if file_path.suffix.lower() == ".pdf":
                    text = engine.extract_text_from_pdf(file_path)
                elif file_path.suffix.lower() == ".docx":
                    text = engine.extract_text_from_docx(file_path)
                else:
                    continue
                
                out_txt.write_text(text, encoding="utf-8")
                results["processed"].append({
                    "source": str(file_path),
                    "output": str(out_txt),
                    "chars": len(text)
                })
            except Exception as e:
                results["failed"].append({
                    "source": str(file_path),
                    "error": str(e)
                })
    
    return results


@router.post("/build-index")
async def build_search_index(
    txt_dir: str,
    db_path: str = "./index/filings.sqlite"
):
    """
    Build FTS5 search index from extracted text files.
    """
    from ..generators.pleadings import FilingsSearchEngine
    
    txt_path = Path(txt_dir)
    if not txt_path.exists():
        raise HTTPException(status_code=404, detail="Text directory not found")
    
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    
    engine = FilingsSearchEngine()
    count = engine.build_index(txt_path, db_file)
    
    return {
        "success": True,
        "documents_indexed": count,
        "database": str(db_file)
    }


@router.post("/search", response_model=List[SearchResultItem])
async def search_filings(
    request: SearchRequest,
    db_path: str = "./index/filings.sqlite"
):
    """
    Full-text search of indexed filings.
    
    Supports FTS5 query syntax:
    - "tow NEAR/10 notice" - words within 10 tokens
    - "excessive AND force" - both required
    - "miranda OR rights" - either word
    - "body-worn camera" - phrase search
    """
    from ..generators.pleadings import FilingsSearchEngine
    
    db_file = Path(db_path)
    if not db_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Search index not found. Run /build-index first."
        )
    
    engine = FilingsSearchEngine(db_file)
    results = engine.search(request.query, request.limit)
    
    return [
        SearchResultItem(
            document=r.document,
            snippet=r.snippet,
            score=r.score,
            context=r.context
        )
        for r in results
    ]


@router.post("/grep", response_model=List[SearchResultItem])
async def grep_filings(
    request: GrepRequest,
    txt_dir: str
):
    """
    Regex search without database index.
    
    Useful for quick searches or patterns that don't
    work well with FTS5.
    """
    from ..generators.pleadings import FilingsSearchEngine
    
    txt_path = Path(txt_dir)
    if not txt_path.exists():
        raise HTTPException(status_code=404, detail="Text directory not found")
    
    engine = FilingsSearchEngine()
    results = engine.grep_search(
        txt_path,
        request.pattern,
        request.literal,
        request.context_chars,
        request.max_hits
    )
    
    return [
        SearchResultItem(
            document=r.document,
            snippet=r.snippet,
            score=r.score,
            context=r.context
        )
        for r in results
    ]


@router.post("/merge-exhibits")
async def merge_exhibits(request: ExhibitMergeRequest):
    """
    Merge multiple PDFs into a single exhibit package.
    """
    from ..generators.pleadings import ExhibitMerger
    
    pdf_paths = [Path(p) for p in request.exhibit_pdfs]
    
    for p in pdf_paths:
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"PDF not found: {p}")
    
    output_dir = Path("./exports/exhibits")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{timestamp}_{request.output_name}"
    
    try:
        ExhibitMerger.merge_pdfs(output_path, pdf_paths)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {
        "success": True,
        "output_path": str(output_path),
        "download_url": f"/api/v1/pleadings/download-exhibit/{output_path.name}",
        "page_count": sum(1 for _ in pdf_paths)  # Simplified
    }


@router.get("/download-exhibit/{filename}")
async def download_exhibit(filename: str):
    """Download a merged exhibit PDF"""
    file_path = Path("./exports/exhibits") / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/pdf"
    )


@router.post("/upload-facts")
async def upload_facts_yaml(file: UploadFile = File(...)):
    """
    Upload a YAML facts file and return parsed content.
    
    Use this to validate and preview facts before generation.
    """
    import yaml
    
    content = await file.read()
    
    try:
        facts = yaml.safe_load(content.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")
    
    return {
        "success": True,
        "filename": file.filename,
        "facts": facts,
        "sections": list(facts.keys()) if isinstance(facts, dict) else []
    }


@router.get("/facts-template")
async def get_facts_template():
    """
    Get the example YAML facts template.
    """
    template_path = Path(__file__).parent.parent.parent.parent / "njcivil_terminal_tool" / "facts.example.yml"
    
    if not template_path.exists():
        # Return inline template
        return {
            "template": """court:
  county: Atlantic

plaintiff:
  name: "Plaintiff Name"
  address: "123 Main Street\\nCity, NJ 08401"
  city: "City"
  phone: "(555) 123-4567"
  email: "email@example.com"

defendants:
  caption_block: |
    DEFENDANT ONE,
    DEFENDANT TWO,
    John/Jane Doe Defendants 1-10

case:
  docket_new: "__________ (NEW ACTION)"
  verification_date: "January __, 2026"
  motion_date: "January __, 2026"

facts:
  stop_date: "November 29, 2025"
  vehicle_year: 2019
  vehicle_make: "Make"
  vehicle_model: "Model"
  vehicle_vin: "VIN"
  vehicle_plate: "PLATE"
"""
        }
    
    return {"template": template_path.read_text(encoding="utf-8")}
