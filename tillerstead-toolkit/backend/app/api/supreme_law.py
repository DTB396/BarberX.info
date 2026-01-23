"""
BarberX Legal Case Management Pro Suite
Supreme Law API Router - Legal Research Automation Endpoints
"""
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field

from app.services.supreme_law_service import (
    supreme_law_service,
    CaseLawEntry,
    SupremeCourtOpinion,
    ResearchUpdate
)


router = APIRouter(prefix="/api/v1/supreme-law", tags=["Supreme Law Research"])


# ============================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================

class CaseLawQuery(BaseModel):
    """Query for case law search"""
    query: str
    court: Optional[str] = "scotus"  # scotus, ca3, nj, all
    topic_tags: Optional[List[str]] = []
    filed_after: Optional[str] = None
    limit: int = Field(default=50, le=500)


class CaseLawResponse(BaseModel):
    """Case law entry response"""
    case_name: str
    citation_bluebook: str
    court: str
    decision_date: str
    key_holding: str
    topic_tags: List[str]
    pinpoint_cite: str
    official_url: str
    amendments_cited: List[str]


class ResearchUpdateResponse(BaseModel):
    """Research update summary"""
    update_id: str
    update_date: str
    cases_added: int
    new_scotus_count: int
    new_third_circuit_count: int
    new_nj_count: int
    summary: str


class BluebookCitationRequest(BaseModel):
    """Request to generate Bluebook citation"""
    case_name: str
    volume: str
    reporter: str
    page: str
    court: Optional[str] = ""
    year: str


class AddCaseRequest(BaseModel):
    """Request to manually add case to database"""
    case_name: str
    citation_bluebook: str
    court: str
    decision_date: str
    binding_level: str
    key_holding: str
    use_in_brief: str
    official_url: str
    pinpoint_cite: Optional[str] = ""
    topic_tags: Optional[List[str]] = []
    amendments_cited: Optional[List[str]] = []


# ============================================================
# ENDPOINTS
# ============================================================

@router.get("/scotus/slip-opinions", response_model=List[dict])
async def get_scotus_slip_opinions(
    term: Optional[str] = Query(None, description="Court term (e.g., '25' for 2025-2026)")
):
    """
    Fetch Supreme Court slip opinions for given term.
    
    Returns list of recently published SCOTUS opinions with download links.
    """
    try:
        opinions = await supreme_law_service.fetch_scotus_slip_opinions(term)
        return [
            {
                "term": op.term,
                "docket_number": op.docket_number,
                "case_name": op.case_name,
                "opinion_date": op.opinion_date,
                "slip_opinion_url": op.slip_opinion_url,
                "citations_official": op.citations_official,
                "status": op.status
            }
            for op in opinions
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch SCOTUS opinions: {str(e)}")


@router.post("/search", response_model=List[CaseLawResponse])
async def search_case_law(query: CaseLawQuery):
    """
    Search case law database and CourtListener API.
    
    Searches local database first, then queries CourtListener for additional results.
    """
    try:
        # Query CourtListener
        results = await supreme_law_service.query_courtlistener(
            query=query.query,
            court=query.court,
            filed_after=query.filed_after,
            limit=query.limit
        )
        
        # Convert to response format
        responses = []
        for case in results:
            responses.append(CaseLawResponse(
                case_name=case.get('caseName', ''),
                citation_bluebook=case.get('citation', ''),
                court=case.get('court', ''),
                decision_date=case.get('dateFiled', ''),
                key_holding=case.get('snippet', '')[:500],
                topic_tags=[],
                pinpoint_cite="",
                official_url=case.get('absolute_url', ''),
                amendments_cited=[]
            ))
        
        return responses[:query.limit]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/generate-citation", response_model=dict)
async def generate_bluebook_citation(request: BluebookCitationRequest):
    """
    Generate proper Bluebook citation for a case.
    
    Returns formatted citation string following Bluebook rules.
    """
    try:
        citation = supreme_law_service.generate_bluebook_citation(
            case_name=request.case_name,
            volume=request.volume,
            reporter=request.reporter,
            page=request.page,
            court=request.court,
            year=request.year
        )
        
        return {
            "citation_bluebook": citation,
            "case_name": request.case_name,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Citation generation failed: {str(e)}")


@router.post("/add-case", response_model=dict)
async def add_case_to_database(request: AddCaseRequest):
    """
    Manually add verified case to database.
    
    Requires all fields to be filled with verified information.
    Will reject duplicates based on case name hash.
    """
    try:
        # Create CaseLawEntry
        case_entry = CaseLawEntry(
            case_name=request.case_name,
            citation_bluebook=request.citation_bluebook,
            court=request.court,
            decision_date=request.decision_date,
            binding_level=request.binding_level,
            key_holding=request.key_holding,
            use_in_brief=request.use_in_brief,
            pinpoint_cite=request.pinpoint_cite,
            topic_tags=request.topic_tags,
            amendments_cited=request.amendments_cited,
            verification_source="Manual Entry",
            official_url=request.official_url,
            verified_date=datetime.now().isoformat()[:10]
        )
        
        # Add to database
        added = await supreme_law_service.add_case_to_database(case_entry)
        
        if added:
            return {
                "success": True,
                "message": f"Added {request.case_name} to database",
                "case_hash": case_entry.case_hash
            }
        else:
            return {
                "success": False,
                "message": "Case already exists in database (duplicate)",
                "case_name": request.case_name
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add case: {str(e)}")


@router.post("/update/run", response_model=ResearchUpdateResponse)
async def run_automated_update(background_tasks: BackgroundTasks):
    """
    Trigger automated case law update.
    
    Runs background job to:
    1. Fetch new SCOTUS slip opinions
    2. Query CourtListener for recent 3d Circuit opinions
    3. Query CourtListener for recent NJ published opinions
    4. Update database and log results
    """
    try:
        # Run update in background
        update_result = await supreme_law_service.automated_daily_update()
        
        return ResearchUpdateResponse(
            update_id=update_result.update_id,
            update_date=update_result.update_date,
            cases_added=update_result.cases_added,
            new_scotus_count=update_result.new_scotus_count,
            new_third_circuit_count=update_result.new_third_circuit_count,
            new_nj_count=update_result.new_nj_count,
            summary=update_result.summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")


@router.get("/export/json")
async def export_database_json(background_tasks: BackgroundTasks):
    """
    Export case law database to JSON.
    
    Generates JSON file in exports/ directory for web publishing.
    """
    try:
        background_tasks.add_task(
            supreme_law_service.export_to_json,
            "./exports/caselaw.json"
        )
        
        return {
            "success": True,
            "message": "JSON export started",
            "output_path": "./exports/caselaw.json"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.get("/export/html")
async def generate_html_portal(background_tasks: BackgroundTasks):
    """
    Generate HTML portal for case law browsing.
    
    Creates static HTML pages in _site/supreme-law/ for public viewing.
    """
    try:
        background_tasks.add_task(
            supreme_law_service.generate_html_portal,
            "./_site/supreme-law/"
        )
        
        return {
            "success": True,
            "message": "HTML portal generation started",
            "output_path": "./_site/supreme-law/index.html"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Portal generation failed: {str(e)}")


@router.get("/stats", response_model=dict)
async def get_database_stats():
    """
    Get case law database statistics.
    
    Returns counts by court, topic, and recent additions.
    """
    try:
        import pandas as pd
        
        df = pd.read_excel(supreme_law_service.database_path, sheet_name='CaseLaw_DB')
        
        stats = {
            "total_cases": len(df),
            "scotus_cases": len(df[df['Court'] == 'U.S. Supreme Court']),
            "third_circuit_cases": len(df[df['Court'].str.contains('3d Cir', na=False)]),
            "nj_cases": len(df[df['Court'].str.contains('N.J.', na=False)]),
            "last_updated": datetime.now().isoformat(),
            "topics": {}
        }
        
        # Count by topic
        all_tags = []
        for tags in df['Topic_Tags'].dropna():
            all_tags.extend(str(tags).split(', '))
        
        from collections import Counter
        topic_counts = Counter(all_tags)
        stats["topics"] = dict(topic_counts.most_common(10))
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")


@router.get("/topics", response_model=dict)
async def get_available_topics():
    """
    Get list of all available topic tags with descriptions.
    
    Returns topic taxonomy used for auto-tagging.
    """
    return {
        "topics": {
            topic: {
                "tag": topic,
                "keywords": keywords,
                "description": topic.replace('_', ' ').title()
            }
            for topic, keywords in supreme_law_service.TOPIC_KEYWORDS.items()
        }
    }
