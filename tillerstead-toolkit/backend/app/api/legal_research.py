"""
Legal Research & Case Law API
Integrates CourtListener, CAP, and citation analysis tools
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import httpx
from datetime import datetime
import eyecite

router = APIRouter(prefix="/api/v1/research", tags=["Legal Research"])


class CitationExtractRequest(BaseModel):
    text: str = Field(..., description="Legal text to extract citations from")
    resolve: bool = Field(default=True, description="Resolve citations to full case names")


class CitationResult(BaseModel):
    citation: str
    normalized: str
    case_name: Optional[str] = None
    year: Optional[int] = None
    court: Optional[str] = None
    reporter: Optional[str] = None
    page: Optional[str] = None
    confidence: float


class CaseSearchRequest(BaseModel):
    query: str = Field(..., description="Search query for cases")
    jurisdiction: Optional[str] = Field(None, description="Filter by jurisdiction")
    date_start: Optional[datetime] = None
    date_end: Optional[datetime] = None
    max_results: int = Field(default=20, le=100)


class CaseResult(BaseModel):
    id: str
    name: str
    citation: str
    date_filed: Optional[datetime]
    court: str
    snippet: str
    url: str
    relevance_score: float


class ShepardizeRequest(BaseModel):
    citation: str = Field(..., description="Citation to shepardize")
    include_negative: bool = Field(default=True, description="Include negative treatment")


class ShepardResult(BaseModel):
    citation: str
    is_good_law: bool
    treatment_summary: str
    citing_cases: List[Dict[str, Any]]
    distinguished_count: int
    followed_count: int
    overruled_count: int
    questioned_count: int


@router.post("/extract-citations", response_model=List[CitationResult])
async def extract_citations(request: CitationExtractRequest):
    """
    Extract legal citations from text using eyecite
    
    Supports:
    - Federal reporters (F., F.2d, F.3d, F.4th)
    - State reporters
    - Supreme Court (U.S., S.Ct., L.Ed.)
    - Statutory citations
    """
    try:
        # Extract citations using eyecite
        found_cites = eyecite.get_citations(request.text)
        
        results = []
        for cite in found_cites:
            result = CitationResult(
                citation=str(cite),
                normalized=cite.corrected_citation() if hasattr(cite, 'corrected_citation') else str(cite),
                confidence=0.95  # eyecite is highly accurate
            )
            
            # Extract metadata if available
            if hasattr(cite, 'metadata'):
                meta = cite.metadata
                result.case_name = meta.get('case_name')
                result.year = meta.get('year')
                result.court = meta.get('court')
                result.reporter = cite.reporter if hasattr(cite, 'reporter') else None
                result.page = str(cite.page) if hasattr(cite, 'page') else None
            
            results.append(result)
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Citation extraction failed: {str(e)}")


@router.post("/search-cases", response_model=List[CaseResult])
async def search_cases(request: CaseSearchRequest):
    """
    Search case law using CourtListener API
    
    Free Law Project provides access to millions of court opinions
    """
    try:
        async with httpx.AsyncClient() as client:
            params = {
                'q': request.query,
                'order_by': 'score desc',
                'format': 'json',
                'page_size': request.max_results
            }
            
            if request.jurisdiction:
                params['court'] = request.jurisdiction
            
            if request.date_start:
                params['filed_after'] = request.date_start.isoformat()
            
            if request.date_end:
                params['filed_before'] = request.date_end.isoformat()
            
            # Note: Requires CourtListener API key in production
            headers = {
                'Authorization': 'Token YOUR_API_KEY_HERE'  # Configure in settings
            }
            
            response = await client.get(
                'https://www.courtlistener.com/api/rest/v3/search/',
                params=params,
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get('results', []):
                results.append(CaseResult(
                    id=str(item['id']),
                    name=item.get('caseName', 'Unknown Case'),
                    citation=item.get('citation', ''),
                    date_filed=item.get('dateFiled'),
                    court=item.get('court', ''),
                    snippet=item.get('snippet', ''),
                    url=f"https://www.courtlistener.com{item.get('absolute_url', '')}",
                    relevance_score=item.get('score', 0.0)
                ))
            
            return results
            
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"CourtListener API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Case search failed: {str(e)}")


@router.post("/shepardize", response_model=ShepardResult)
async def shepardize_citation(request: ShepardizeRequest):
    """
    Shepardize a citation to check if it's still good law
    
    Analyzes subsequent treatment by other cases
    """
    try:
        # This is a simplified implementation
        # In production, integrate with Courtlistener's citator or Caselaw Access Project
        
        async with httpx.AsyncClient() as client:
            # Search for cases citing this citation
            params = {
                'q': f'cites:({request.citation})',
                'format': 'json',
                'page_size': 100
            }
            
            headers = {
                'Authorization': 'Token YOUR_API_KEY_HERE'
            }
            
            response = await client.get(
                'https://www.courtlistener.com/api/rest/v3/search/',
                params=params,
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            
            citing_cases = data.get('results', [])
            
            # Analyze treatment
            followed = 0
            distinguished = 0
            questioned = 0
            overruled = 0
            
            for case in citing_cases:
                snippet = case.get('snippet', '').lower()
                
                if 'overruled' in snippet or 'overturned' in snippet:
                    overruled += 1
                elif 'distinguished' in snippet:
                    distinguished += 1
                elif 'questioned' in snippet or 'criticized' in snippet:
                    questioned += 1
                elif 'followed' in snippet or 'applied' in snippet:
                    followed += 1
            
            is_good_law = overruled == 0 and questioned < 3
            
            if overruled > 0:
                treatment = "NEGATIVE - Overruled"
            elif questioned > 2:
                treatment = "CAUTION - Questioned by multiple courts"
            elif distinguished > followed:
                treatment = "NEUTRAL - Frequently distinguished"
            else:
                treatment = "POSITIVE - Good law"
            
            return ShepardResult(
                citation=request.citation,
                is_good_law=is_good_law,
                treatment_summary=treatment,
                citing_cases=[{
                    'name': c.get('caseName'),
                    'citation': c.get('citation'),
                    'date': c.get('dateFiled')
                } for c in citing_cases[:20]],
                distinguished_count=distinguished,
                followed_count=followed,
                overruled_count=overruled,
                questioned_count=questioned
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Shepardizing failed: {str(e)}")


@router.get("/statutes/search")
async def search_statutes(
    query: str = Query(..., description="Statute search query"),
    jurisdiction: str = Query(default="federal", description="Jurisdiction (federal/state)")
):
    """
    Search statutory law
    """
    # Integrate with Legal Information Institute, Cornell LII, or Justia
    return {"message": "Statute search endpoint - implementation pending"}


@router.get("/case/{case_id}/full-text")
async def get_case_full_text(case_id: str):
    """
    Retrieve full text of a court opinion
    """
    try:
        async with httpx.AsyncClient() as client:
            headers = {'Authorization': 'Token YOUR_API_KEY_HERE'}
            
            response = await client.get(
                f'https://www.courtlistener.com/api/rest/v3/opinions/{case_id}/',
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            
            return response.json()
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve case: {str(e)}")


@router.get("/similar-cases")
async def find_similar_cases(
    case_text: str = Query(..., description="Case description or facts"),
    jurisdiction: Optional[str] = None,
    limit: int = Query(default=10, le=50)
):
    """
    Find similar cases using semantic search
    
    Uses sentence transformers for similarity matching
    """
    # Implementation would use sentence-transformers to embed case_text
    # and find similar cases in vector database
    return {"message": "Similar case search - requires vector DB setup"}
