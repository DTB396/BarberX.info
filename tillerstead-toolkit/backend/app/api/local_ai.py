"""
Local AI API - Zero-Cost, Offline AI Processing
Uses Ollama, local embeddings, ChromaDB, and other offline tools
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import httpx
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime

router = APIRouter(prefix="/api/v1/local-ai", tags=["Local AI - Offline"])

# Global instances (load models on startup)
local_models = {}
chroma_client = None


class LocalChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    model: str = Field(default="llama3.2:8b", description="Local model to use")
    context: Optional[List[Dict[str, str]]] = Field(None, description="Chat history")
    system_prompt: Optional[str] = Field(None, description="System instructions")


class LocalChatResponse(BaseModel):
    response: str
    model: str
    tokens_used: int
    processing_time: float


class DocumentIndexRequest(BaseModel):
    document_id: str
    text: str
    metadata: Dict[str, Any] = {}
    collection: str = Field(default="legal_documents")


class SemanticSearchRequest(BaseModel):
    query: str
    collection: str = Field(default="legal_documents")
    n_results: int = Field(default=10, le=100)
    filter: Optional[Dict[str, Any]] = None


class LocalSummarizeRequest(BaseModel):
    text: str
    max_length: int = Field(default=150, description="Max words in summary")
    model: str = Field(default="llama3.2:8b")


class LocalAnalysisRequest(BaseModel):
    document_text: str
    analysis_type: str = Field(
        ...,
        description="constitutional, contract, discovery, medical, etc."
    )
    model: str = Field(default="llama3.2:8b")


@router.on_event("startup")
async def initialize_local_ai():
    """Initialize local AI models and databases"""
    global local_models, chroma_client
    
    try:
        # Initialize ChromaDB (local vector database)
        from chromadb.config import Settings
        chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./local_ai_data/chroma"
        ))
        print("✅ ChromaDB initialized (local vector database)")
        
        # Load local embedding model (sentence-transformers)
        local_models['embeddings_mini'] = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Loaded all-MiniLM-L6-v2 (80MB, local embeddings)")
        
        # Try to load legal-specific embeddings
        try:
            local_models['embeddings_legal'] = SentenceTransformer('nlpaueb/legal-bert-base-uncased')
            print("✅ Loaded Legal-BERT (400MB, legal embeddings)")
        except:
            print("⚠️ Legal-BERT not available, using MiniLM for legal docs")
            local_models['embeddings_legal'] = local_models['embeddings_mini']
        
        print("✅ Local AI initialized - All processing happens on your machine!")
        
    except Exception as e:
        print(f"⚠️ Failed to initialize some local AI components: {e}")


@router.post("/chat", response_model=LocalChatResponse)
async def local_chat(request: LocalChatRequest):
    """
    Chat with local LLM (Ollama)
    
    Requires Ollama running locally: https://ollama.ai
    
    Supported models:
    - llama3.2:8b (5GB, best quality)
    - llama3.2:3b (2GB, balanced)
    - phi3:mini (2.3GB, fastest)
    - mistral:7b (4GB, great reasoning)
    - codellama:7b (4GB, code/extraction)
    
    ALL PROCESSING HAPPENS LOCALLY - NO API CALLS!
    """
    try:
        import time
        start_time = time.time()
        
        # Build messages for Ollama
        messages = []
        
        # Add system prompt if provided
        if request.system_prompt:
            messages.append({
                "role": "system",
                "content": request.system_prompt
            })
        
        # Add context (chat history)
        if request.context:
            messages.extend(request.context)
        
        # Add current message
        messages.append({
            "role": "user",
            "content": request.message
        })
        
        # Call local Ollama API (runs on your machine!)
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": request.model,
                    "messages": messages,
                    "stream": False
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=503,
                    detail="Ollama not running. Install from https://ollama.ai and run: ollama pull llama3.2:8b"
                )
            
            result = response.json()
            processing_time = time.time() - start_time
            
            return LocalChatResponse(
                response=result['message']['content'],
                model=request.model,
                tokens_used=result.get('eval_count', 0),
                processing_time=processing_time
            )
        
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Ollama not running. Start it with: ollama serve"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Local chat failed: {str(e)}")


@router.post("/index-document")
async def index_document_locally(request: DocumentIndexRequest):
    """
    Index document in local vector database (ChromaDB)
    
    Stores embeddings locally for semantic search
    NO CLOUD SERVICES - ALL LOCAL!
    """
    try:
        # Get or create collection
        try:
            collection = chroma_client.get_collection(request.collection)
        except:
            collection = chroma_client.create_collection(
                name=request.collection,
                metadata={"description": "Local legal documents"}
            )
        
        # Generate embedding using local model
        embedding_model = local_models.get('embeddings_legal') or local_models.get('embeddings_mini')
        embedding = embedding_model.encode(request.text).tolist()
        
        # Add to local vector database
        collection.add(
            ids=[request.document_id],
            embeddings=[embedding],
            documents=[request.text],
            metadatas=[request.metadata]
        )
        
        return {
            "status": "indexed",
            "document_id": request.document_id,
            "collection": request.collection,
            "embedding_dimension": len(embedding),
            "storage": "local"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Local indexing failed: {str(e)}")


@router.post("/semantic-search")
async def semantic_search_local(request: SemanticSearchRequest):
    """
    Semantic search using local embeddings and ChromaDB
    
    Searches your local vector database - NO EXTERNAL APIS!
    """
    try:
        # Get collection
        collection = chroma_client.get_collection(request.collection)
        
        # Generate query embedding locally
        embedding_model = local_models.get('embeddings_legal') or local_models.get('embeddings_mini')
        query_embedding = embedding_model.encode(request.query).tolist()
        
        # Search locally
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=request.n_results,
            where=request.filter
        )
        
        return {
            "query": request.query,
            "results": [
                {
                    "document_id": id,
                    "text": doc,
                    "metadata": meta,
                    "similarity_score": 1 - dist  # Convert distance to similarity
                }
                for id, doc, meta, dist in zip(
                    results['ids'][0],
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )
            ],
            "storage": "local",
            "processing": "offline"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Local search failed: {str(e)}")


@router.post("/summarize-local")
async def summarize_locally(request: LocalSummarizeRequest):
    """
    Summarize text using local LLM
    
    Uses your local Ollama models - NO API COSTS!
    """
    try:
        # Build prompt for summarization
        prompt = f"""Summarize the following legal document in approximately {request.max_length} words. 
Focus on the key facts, legal issues, and conclusions. Be concise and accurate.

Document:
{request.text}

Summary:"""
        
        # Call local Ollama
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": request.model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            result = response.json()
            
            return {
                "summary": result['response'],
                "original_length": len(request.text.split()),
                "summary_length": len(result['response'].split()),
                "model": request.model,
                "processing": "local"
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Local summarization failed: {str(e)}")


@router.post("/analyze-local")
async def analyze_locally(request: LocalAnalysisRequest):
    """
    Perform specialized legal analysis using local LLM
    
    Analysis types:
    - constitutional: 4th/5th/6th/8th/14th Amendment violations
    - contract: Risk assessment, clause analysis
    - discovery: Responsive document analysis
    - medical: Medical record analysis
    - deposition: Testimony analysis
    
    ALL LOCAL - NO EXTERNAL APIs!
    """
    try:
        # Build specialized prompt based on analysis type
        prompts = {
            "constitutional": f"""Analyze this document for constitutional violations. Focus on:
- 4th Amendment (search/seizure)
- 5th Amendment (due process, self-incrimination)
- 6th Amendment (right to counsel)
- 8th Amendment (excessive force, cruel punishment)
- 14th Amendment (equal protection)

Document:
{request.document_text}

Analysis:""",
            
            "contract": f"""Analyze this contract for risks and concerning clauses. Identify:
- High-risk provisions (indemnification, liability, non-compete)
- Unfavorable terms
- Missing protections
- Recommended changes

Contract:
{request.document_text}

Risk Assessment:""",
            
            "discovery": f"""Analyze whether this document is responsive to discovery requests. Identify:
- Relevant facts
- Key parties mentioned
- Important dates
- Potential privilege issues

Document:
{request.document_text}

Discovery Analysis:""",
            
            "medical": f"""Analyze this medical record. Extract:
- Diagnoses
- Injuries
- Treatments
- Causation issues
- Permanency indicators

Medical Record:
{request.document_text}

Medical Analysis:""",
            
            "deposition": f"""Analyze this deposition testimony. Identify:
- Inconsistencies
- Admissions
- Credibility issues
- Key testimony

Transcript:
{request.document_text}

Deposition Analysis:"""
        }
        
        prompt = prompts.get(request.analysis_type, f"Analyze this legal document:\n\n{request.document_text}\n\nAnalysis:")
        
        # Call local LLM
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": request.model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            result = response.json()
            
            return {
                "analysis_type": request.analysis_type,
                "analysis": result['response'],
                "model": request.model,
                "processing": "local",
                "cost": "$0"
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Local analysis failed: {str(e)}")


@router.get("/models/available")
async def list_available_local_models():
    """
    List locally installed Ollama models
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags")
            result = response.json()
            
            models = []
            for model in result.get('models', []):
                models.append({
                    "name": model['name'],
                    "size": model['size'],
                    "modified": model['modified_at']
                })
            
            return {
                "installed_models": models,
                "recommended_downloads": [
                    {"name": "llama3.2:8b", "size": "5GB", "use": "Best quality"},
                    {"name": "phi3:mini", "size": "2.3GB", "use": "Fastest"},
                    {"name": "mistral:7b", "size": "4GB", "use": "Reasoning"}
                ],
                "download_command": "ollama pull <model_name>"
            }
        
    except:
        return {
            "installed_models": [],
            "status": "Ollama not running",
            "install": "Download from https://ollama.ai"
        }


@router.post("/models/download")
async def download_local_model(model_name: str):
    """
    Download a model to local storage
    
    This happens once - then it's free forever!
    """
    try:
        async with httpx.AsyncClient(timeout=3600.0) as client:
            # Trigger model pull
            response = await client.post(
                "http://localhost:11434/api/pull",
                json={"name": model_name}
            )
            
            return {
                "status": "downloading",
                "model": model_name,
                "note": "Model will be stored locally and available offline"
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@router.get("/storage/stats")
async def get_local_storage_stats():
    """
    Get statistics about local storage usage
    """
    try:
        import os
        
        stats = {
            "chroma_db": {
                "path": "./local_ai_data/chroma",
                "exists": os.path.exists("./local_ai_data/chroma")
            },
            "models_loaded": {
                "embeddings_mini": "embeddings_mini" in local_models,
                "embeddings_legal": "embeddings_legal" in local_models
            },
            "processing": "100% local",
            "cost": "$0",
            "privacy": "Complete - data never leaves your machine"
        }
        
        # Get ChromaDB collections
        if chroma_client:
            collections = chroma_client.list_collections()
            stats["vector_collections"] = [
                {
                    "name": col.name,
                    "count": col.count()
                }
                for col in collections
            ]
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats failed: {str(e)}")


@router.post("/extract-entities-local")
async def extract_entities_locally(text: str, model: str = "llama3.2:8b"):
    """
    Extract legal entities using local LLM
    
    Extracts: parties, judges, dates, citations, violations, etc.
    """
    try:
        prompt = f"""Extract legal entities from this text. Return as JSON with:
- parties: list of people/organizations
- judges: list of judges
- dates: list of important dates
- citations: list of legal citations
- violations: list of alleged violations
- locations: list of locations

Text:
{text}

JSON:"""
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False
                }
            )
            
            result = response.json()
            
            import json
            entities = json.loads(result['response'])
            
            return {
                "entities": entities,
                "processing": "local",
                "model": model
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Entity extraction failed: {str(e)}")
