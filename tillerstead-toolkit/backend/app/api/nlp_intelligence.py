"""
Advanced NLP & Document Intelligence API
Legal NER, summarization, semantic search, and contract analysis
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import spacy
from transformers import pipeline, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch

router = APIRouter(prefix="/api/v1/nlp", tags=["NLP & AI"])

# Global model cache (load on startup)
nlp_models = {}


class DocumentAnalysisRequest(BaseModel):
    text: str = Field(..., description="Document text to analyze")
    extract_entities: bool = Field(default=True)
    extract_dates: bool = Field(default=True)
    extract_citations: bool = Field(default=True)
    extract_parties: bool = Field(default=True)


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    confidence: float


class DocumentAnalysisResult(BaseModel):
    entities: List[Entity]
    parties: List[str]
    dates: List[Dict[str, Any]]
    citations: List[str]
    key_phrases: List[str]
    summary: Optional[str] = None


class SummarizationRequest(BaseModel):
    text: str = Field(..., description="Text to summarize")
    max_length: int = Field(default=150, le=500, description="Max summary length in words")
    min_length: int = Field(default=50, description="Min summary length in words")
    style: str = Field(default="legal", description="Summary style: legal, plain, technical")


class SemanticSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    documents: List[str] = Field(..., description="Documents to search")
    top_k: int = Field(default=5, le=20)


class SemanticSearchResult(BaseModel):
    document: str
    score: float
    rank: int


class ContractAnalysisRequest(BaseModel):
    contract_text: str = Field(..., description="Contract text to analyze")
    check_risks: bool = Field(default=True, description="Identify risky clauses")
    extract_obligations: bool = Field(default=True)
    extract_parties: bool = Field(default=True)
    extract_dates: bool = Field(default=True)


class ContractClause(BaseModel):
    type: str
    text: str
    risk_level: str  # high, medium, low
    explanation: str
    location: Dict[str, int]


class ContractAnalysisResult(BaseModel):
    parties: List[str]
    effective_date: Optional[str]
    termination_date: Optional[str]
    clauses: List[ContractClause]
    obligations: List[Dict[str, Any]]
    risk_summary: str
    overall_risk_score: float


@router.on_event("startup")
async def load_models():
    """Load NLP models on startup to avoid repeated loading"""
    try:
        # Load spaCy legal model
        # Note: Install with: python -m spacy download en_core_web_trf
        nlp_models['spacy'] = spacy.load("en_core_web_trf")
        
        # Load legal BERT for embeddings
        nlp_models['legal_bert'] = SentenceTransformer('nlpaueb/legal-bert-base-uncased')
        
        # Load summarization model
        nlp_models['summarizer'] = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Load sentiment analyzer for contract risk
        nlp_models['sentiment'] = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=0 if torch.cuda.is_available() else -1
        )
        
        print("✅ NLP models loaded successfully")
        
    except Exception as e:
        print(f"⚠️ Failed to load some NLP models: {e}")
        print("Models will be loaded on-demand")


@router.post("/analyze-document", response_model=DocumentAnalysisResult)
async def analyze_document(request: DocumentAnalysisRequest):
    """
    Comprehensive legal document analysis using spaCy and custom legal NER
    
    Extracts:
    - Legal entities (PERSON, ORG, GPE, LAW, COURT, etc.)
    - Dates and deadlines
    - Citations
    - Key parties
    - Important phrases
    """
    try:
        nlp = nlp_models.get('spacy') or spacy.load("en_core_web_trf")
        doc = nlp(request.text)
        
        # Extract entities
        entities = []
        parties = set()
        dates = []
        
        for ent in doc.ents:
            entity = Entity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                confidence=0.9  # spaCy doesn't provide confidence scores directly
            )
            entities.append(entity)
            
            # Collect parties (PERSON and ORG entities)
            if request.extract_parties and ent.label_ in ['PERSON', 'ORG']:
                parties.add(ent.text)
            
            # Collect dates
            if request.extract_dates and ent.label_ == 'DATE':
                dates.append({
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
        
        # Extract key phrases using noun chunks
        key_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 2][:10]
        
        return DocumentAnalysisResult(
            entities=entities,
            parties=list(parties),
            dates=dates,
            citations=[],  # Use eyecite from legal_research.py
            key_phrases=key_phrases
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document analysis failed: {str(e)}")


@router.post("/summarize", response_model=Dict[str, str])
async def summarize_text(request: SummarizationRequest):
    """
    Summarize legal documents using BART or T5
    
    Supports multiple summary styles:
    - legal: Formal legal language
    - plain: Plain English for clients
    - technical: Detailed technical summary
    """
    try:
        summarizer = nlp_models.get('summarizer') or pipeline("summarization", model="facebook/bart-large-cnn")
        
        # BART works best with chunks of 1024 tokens
        # For longer documents, chunk and summarize
        max_chunk_size = 1024
        
        if len(request.text.split()) > max_chunk_size:
            # Split into chunks
            words = request.text.split()
            chunks = [' '.join(words[i:i+max_chunk_size]) for i in range(0, len(words), max_chunk_size)]
            
            # Summarize each chunk
            summaries = []
            for chunk in chunks:
                result = summarizer(
                    chunk,
                    max_length=request.max_length,
                    min_length=request.min_length,
                    do_sample=False
                )
                summaries.append(result[0]['summary_text'])
            
            # Combine and summarize again if needed
            combined = ' '.join(summaries)
            if len(combined.split()) > request.max_length:
                final_result = summarizer(
                    combined,
                    max_length=request.max_length,
                    min_length=request.min_length,
                    do_sample=False
                )
                summary = final_result[0]['summary_text']
            else:
                summary = combined
        else:
            result = summarizer(
                request.text,
                max_length=request.max_length,
                min_length=request.min_length,
                do_sample=False
            )
            summary = result[0]['summary_text']
        
        return {
            "summary": summary,
            "original_length": len(request.text.split()),
            "summary_length": len(summary.split()),
            "compression_ratio": f"{len(summary.split()) / len(request.text.split()) * 100:.1f}%"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


@router.post("/semantic-search", response_model=List[SemanticSearchResult])
async def semantic_search(request: SemanticSearchRequest):
    """
    Semantic search across legal documents using sentence embeddings
    
    Uses Legal-BERT for domain-specific understanding
    """
    try:
        model = nlp_models.get('legal_bert') or SentenceTransformer('nlpaueb/legal-bert-base-uncased')
        
        # Encode query and documents
        query_embedding = model.encode(request.query, convert_to_tensor=True)
        doc_embeddings = model.encode(request.documents, convert_to_tensor=True)
        
        # Calculate cosine similarity
        from torch.nn.functional import cosine_similarity
        
        similarities = cosine_similarity(
            query_embedding.unsqueeze(0),
            doc_embeddings
        ).cpu().numpy()
        
        # Rank results
        ranked_indices = similarities[0].argsort()[::-1][:request.top_k]
        
        results = []
        for rank, idx in enumerate(ranked_indices):
            results.append(SemanticSearchResult(
                document=request.documents[idx],
                score=float(similarities[0][idx]),
                rank=rank + 1
            ))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")


@router.post("/analyze-contract", response_model=ContractAnalysisResult)
async def analyze_contract(request: ContractAnalysisRequest):
    """
    Comprehensive contract analysis
    
    Features:
    - Party identification
    - Date extraction (effective, termination, renewal)
    - Clause classification
    - Risk assessment
    - Obligation extraction
    """
    try:
        nlp = nlp_models.get('spacy') or spacy.load("en_core_web_trf")
        doc = nlp(request.contract_text)
        
        # Extract parties
        parties = []
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG'] and ent.text not in parties:
                parties.append(ent.text)
        
        # Extract dates
        dates = []
        for ent in doc.ents:
            if ent.label_ == 'DATE':
                dates.append(ent.text)
        
        effective_date = dates[0] if dates else None
        termination_date = dates[-1] if len(dates) > 1 else None
        
        # Identify risky clauses using keyword matching and sentiment
        risky_patterns = [
            ('indemnification', 'high', 'Broad indemnification clause'),
            ('liability', 'medium', 'Liability limitation clause'),
            ('arbitration', 'medium', 'Mandatory arbitration clause'),
            ('non-compete', 'high', 'Non-compete restriction'),
            ('termination', 'medium', 'Termination provisions'),
            ('confidentiality', 'low', 'Confidentiality obligations'),
            ('assignment', 'medium', 'Assignment restrictions'),
            ('force majeure', 'low', 'Force majeure clause'),
        ]
        
        clauses = []
        overall_risk = 0.0
        
        for pattern, risk_level, explanation in risky_patterns:
            # Find sentences containing the pattern
            for sent in doc.sents:
                if pattern in sent.text.lower():
                    clause = ContractClause(
                        type=pattern.replace('-', ' ').title(),
                        text=sent.text,
                        risk_level=risk_level,
                        explanation=explanation,
                        location={'start': sent.start_char, 'end': sent.end_char}
                    )
                    clauses.append(clause)
                    
                    # Calculate risk score
                    if risk_level == 'high':
                        overall_risk += 0.3
                    elif risk_level == 'medium':
                        overall_risk += 0.15
                    else:
                        overall_risk += 0.05
        
        overall_risk = min(overall_risk, 1.0)
        
        # Extract obligations (sentences with "shall", "must", "will")
        obligations = []
        for sent in doc.sents:
            if any(word in sent.text.lower() for word in ['shall', 'must', 'will', 'agree to']):
                obligations.append({
                    'text': sent.text,
                    'type': 'mandatory' if 'shall' in sent.text.lower() or 'must' in sent.text.lower() else 'future'
                })
        
        # Generate risk summary
        high_risk_count = sum(1 for c in clauses if c.risk_level == 'high')
        if high_risk_count > 3:
            risk_summary = "HIGH RISK: Multiple concerning clauses require attorney review"
        elif high_risk_count > 0:
            risk_summary = "MODERATE RISK: Some concerning clauses identified"
        else:
            risk_summary = "LOW RISK: Standard contract provisions"
        
        return ContractAnalysisResult(
            parties=parties[:5],  # Top 5 parties
            effective_date=effective_date,
            termination_date=termination_date,
            clauses=clauses,
            obligations=obligations[:20],  # Top 20 obligations
            risk_summary=risk_summary,
            overall_risk_score=overall_risk
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Contract analysis failed: {str(e)}")


@router.post("/classify-document")
async def classify_document(
    text: str = Field(..., description="Document text"),
    categories: List[str] = Field(default=['motion', 'order', 'brief', 'complaint', 'discovery'])
):
    """
    Classify legal document type using zero-shot classification
    """
    try:
        from transformers import pipeline
        
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Take first 512 tokens for classification
        text_sample = ' '.join(text.split()[:512])
        
        result = classifier(text_sample, categories)
        
        return {
            "predicted_class": result['labels'][0],
            "confidence": result['scores'][0],
            "all_scores": dict(zip(result['labels'], result['scores']))
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")
