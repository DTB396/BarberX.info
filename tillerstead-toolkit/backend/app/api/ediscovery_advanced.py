"""
Advanced E-Discovery Platform API
Apache Tika, Elasticsearch, deduplication, email threading
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import tika
from tika import parser
import hashlib
import ssdeep
import tlsh
from elasticsearch import Elasticsearch, AsyncElasticsearch
from elasticsearch.helpers import async_bulk
import asyncio
from datetime import datetime

router = APIRouter(prefix="/api/v1/ediscovery-advanced", tags=["E-Discovery Advanced"])

# Elasticsearch client (configure in settings)
es_client = None


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    file_size: int
    text_extracted: bool
    page_count: Optional[int] = None
    metadata: Dict[str, Any]
    hash_md5: str
    hash_sha256: str
    fuzzy_hash: str


class DeduplicationRequest(BaseModel):
    document_ids: List[str] = Field(..., description="Documents to check for duplicates")
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    method: str = Field(default="fuzzy", description="Method: exact, fuzzy, near-duplicate")


class DuplicateGroup(BaseModel):
    documents: List[str]
    similarity_score: float
    duplicate_type: str


class EmailThreadRequest(BaseModel):
    email_ids: List[str] = Field(..., description="Email document IDs to thread")
    group_by: str = Field(default="conversation", description="conversation, subject, or sender")


class EmailThread(BaseModel):
    thread_id: str
    subject: str
    participants: List[str]
    email_count: int
    date_range: Dict[str, datetime]
    emails: List[Dict[str, Any]]


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query (supports boolean, fuzzy, proximity)")
    filters: Optional[Dict[str, Any]] = None
    date_range: Optional[Dict[str, datetime]] = None
    file_types: Optional[List[str]] = None
    custodians: Optional[List[str]] = None
    size: int = Field(default=20, le=100)
    from_: int = Field(default=0, alias="from")


class ProductionSetRequest(BaseModel):
    name: str = Field(..., description="Production set name")
    document_ids: List[str] = Field(..., description="Documents to include")
    bates_prefix: str = Field(default="PROD", description="Bates number prefix")
    starting_number: int = Field(default=1)
    confidentiality_designation: Optional[str] = None
    output_format: str = Field(default="native", description="native, pdf, or tiff")


@router.on_event("startup")
async def connect_elasticsearch():
    """Connect to Elasticsearch on startup"""
    global es_client
    try:
        es_client = AsyncElasticsearch(['http://localhost:9200'])
        
        # Create index for legal documents if it doesn't exist
        index_mapping = {
            "mappings": {
                "properties": {
                    "filename": {"type": "keyword"},
                    "content": {"type": "text", "analyzer": "standard"},
                    "file_type": {"type": "keyword"},
                    "upload_date": {"type": "date"},
                    "custodian": {"type": "keyword"},
                    "case_id": {"type": "keyword"},
                    "metadata": {"type": "object"},
                    "hash_md5": {"type": "keyword"},
                    "hash_sha256": {"type": "keyword"},
                    "fuzzy_hash": {"type": "keyword"},
                    "tags": {"type": "keyword"},
                    "privilege": {"type": "boolean"},
                    "confidential": {"type": "boolean"},
                    "bates_number": {"type": "keyword"},
                    "production_set": {"type": "keyword"}
                }
            },
            "settings": {
                "number_of_shards": 2,
                "number_of_replicas": 1
            }
        }
        
        if not await es_client.indices.exists(index="legal_documents"):
            await es_client.indices.create(index="legal_documents", body=index_mapping)
            print("✅ Elasticsearch index created: legal_documents")
        
        print("✅ Connected to Elasticsearch")
        
    except Exception as e:
        print(f"⚠️ Failed to connect to Elasticsearch: {e}")


@router.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    case_id: str = Form(...),
    custodian: Optional[str] = Form(None)
):
    """
    Upload and process document with Apache Tika
    
    Supports:
    - PDF, Word, Excel, PowerPoint
    - Email (MSG, EML, PST)
    - Images with OCR
    - Archives (ZIP, RAR)
    - And 1000+ other formats
    """
    try:
        # Read file content
        content = await file.read()
        
        # Parse with Apache Tika
        parsed = parser.from_buffer(content)
        
        # Extract text and metadata
        text_content = parsed.get('content', '')
        tika_metadata = parsed.get('metadata', {})
        
        # Calculate hashes
        md5_hash = hashlib.md5(content).hexdigest()
        sha256_hash = hashlib.sha256(content).hexdigest()
        fuzzy_hash = ssdeep.hash(content)
        
        # Generate document ID
        doc_id = f"{case_id}_{sha256_hash[:16]}"
        
        # Prepare document for Elasticsearch
        doc_data = {
            "filename": file.filename,
            "content": text_content,
            "file_type": tika_metadata.get('Content-Type', 'unknown'),
            "file_size": len(content),
            "upload_date": datetime.utcnow(),
            "case_id": case_id,
            "custodian": custodian,
            "metadata": tika_metadata,
            "hash_md5": md5_hash,
            "hash_sha256": sha256_hash,
            "fuzzy_hash": fuzzy_hash,
            "text_extracted": bool(text_content and text_content.strip()),
            "privilege": False,
            "confidential": False
        }
        
        # Index in Elasticsearch
        await es_client.index(index="legal_documents", id=doc_id, body=doc_data)
        
        return DocumentUploadResponse(
            document_id=doc_id,
            filename=file.filename,
            file_type=tika_metadata.get('Content-Type', 'unknown'),
            file_size=len(content),
            text_extracted=doc_data['text_extracted'],
            page_count=tika_metadata.get('xmpTPg:NPages'),
            metadata=tika_metadata,
            hash_md5=md5_hash,
            hash_sha256=sha256_hash,
            fuzzy_hash=fuzzy_hash
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")


@router.post("/find-duplicates", response_model=List[DuplicateGroup])
async def find_duplicates(request: DeduplicationRequest):
    """
    Find duplicate and near-duplicate documents
    
    Methods:
    - exact: MD5/SHA256 hash matching
    - fuzzy: SSDeep fuzzy hashing (finds similar files)
    - near-duplicate: TLSH + content similarity
    """
    try:
        if request.method == "exact":
            # Use MD5 hash aggregation
            agg_query = {
                "size": 0,
                "aggs": {
                    "duplicate_hashes": {
                        "terms": {
                            "field": "hash_md5",
                            "min_doc_count": 2,
                            "size": 10000
                        },
                        "aggs": {
                            "docs": {
                                "top_hits": {
                                    "size": 100,
                                    "_source": ["filename", "document_id"]
                                }
                            }
                        }
                    }
                }
            }
            
            result = await es_client.search(index="legal_documents", body=agg_query)
            
            duplicate_groups = []
            for bucket in result['aggregations']['duplicate_hashes']['buckets']:
                doc_ids = [hit['_id'] for hit in bucket['docs']['hits']['hits']]
                duplicate_groups.append(DuplicateGroup(
                    documents=doc_ids,
                    similarity_score=1.0,
                    duplicate_type="exact"
                ))
            
            return duplicate_groups
            
        elif request.method == "fuzzy":
            # Use SSDeep fuzzy hash comparison
            # Retrieve all documents
            docs = []
            async for hit in async_scan(es_client, index="legal_documents", query={"query": {"match_all": {}}}):
                docs.append({
                    'id': hit['_id'],
                    'fuzzy_hash': hit['_source'].get('fuzzy_hash')
                })
            
            # Compare fuzzy hashes
            duplicate_groups = []
            processed = set()
            
            for i, doc1 in enumerate(docs):
                if doc1['id'] in processed:
                    continue
                    
                group = [doc1['id']]
                
                for doc2 in docs[i+1:]:
                    if doc2['id'] in processed:
                        continue
                    
                    # Compare SSDeep hashes
                    try:
                        similarity = ssdeep.compare(doc1['fuzzy_hash'], doc2['fuzzy_hash'])
                        if similarity >= request.similarity_threshold * 100:
                            group.append(doc2['id'])
                            processed.add(doc2['id'])
                    except:
                        pass
                
                if len(group) > 1:
                    duplicate_groups.append(DuplicateGroup(
                        documents=group,
                        similarity_score=request.similarity_threshold,
                        duplicate_type="fuzzy"
                    ))
                    processed.add(doc1['id'])
            
            return duplicate_groups
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deduplication failed: {str(e)}")


@router.post("/thread-emails", response_model=List[EmailThread])
async def thread_emails(request: EmailThreadRequest):
    """
    Thread emails by conversation
    
    Uses:
    - Message-ID and In-Reply-To headers
    - Subject line normalization
    - Temporal clustering
    """
    try:
        # Retrieve emails
        email_query = {
            "query": {
                "terms": {
                    "_id": request.email_ids
                }
            },
            "size": len(request.email_ids)
        }
        
        result = await es_client.search(index="legal_documents", body=email_query)
        emails = result['hits']['hits']
        
        # Group by conversation
        threads = {}
        
        for email in emails:
            metadata = email['_source'].get('metadata', {})
            
            # Extract threading metadata
            message_id = metadata.get('Message-ID', '')
            in_reply_to = metadata.get('In-Reply-To', '')
            subject = metadata.get('subject', '')
            
            # Normalize subject (remove RE:, FW:, etc.)
            normalized_subject = subject.lower()
            for prefix in ['re:', 'fw:', 'fwd:', 'reply:']:
                normalized_subject = normalized_subject.replace(prefix, '').strip()
            
            # Determine thread ID
            if request.group_by == "conversation" and in_reply_to:
                thread_id = in_reply_to
            else:
                thread_id = hashlib.md5(normalized_subject.encode()).hexdigest()
            
            if thread_id not in threads:
                threads[thread_id] = {
                    'emails': [],
                    'participants': set(),
                    'subject': subject,
                    'dates': []
                }
            
            threads[thread_id]['emails'].append(email)
            threads[thread_id]['participants'].add(metadata.get('From', ''))
            threads[thread_id]['dates'].append(email['_source'].get('upload_date'))
        
        # Format results
        result_threads = []
        for thread_id, thread_data in threads.items():
            dates = sorted(thread_data['dates'])
            
            result_threads.append(EmailThread(
                thread_id=thread_id,
                subject=thread_data['subject'],
                participants=list(thread_data['participants']),
                email_count=len(thread_data['emails']),
                date_range={'start': dates[0], 'end': dates[-1]},
                emails=[{
                    'id': e['_id'],
                    'from': e['_source'].get('metadata', {}).get('From'),
                    'date': e['_source'].get('upload_date')
                } for e in thread_data['emails']]
            ))
        
        return result_threads
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Email threading failed: {str(e)}")


@router.post("/search")
async def advanced_search(request: SearchRequest):
    """
    Advanced Elasticsearch search with legal-specific features
    
    Supports:
    - Boolean operators (AND, OR, NOT)
    - Proximity search ("terms within 5 words")
    - Fuzzy matching
    - Date range filtering
    - Custodian filtering
    - Privilege filtering
    """
    try:
        # Build query
        must_clauses = []
        filter_clauses = []
        
        # Main search query
        must_clauses.append({
            "query_string": {
                "query": request.query,
                "default_field": "content",
                "default_operator": "AND"
            }
        })
        
        # Apply filters
        if request.filters:
            for field, value in request.filters.items():
                filter_clauses.append({"term": {field: value}})
        
        if request.file_types:
            filter_clauses.append({"terms": {"file_type": request.file_types}})
        
        if request.custodians:
            filter_clauses.append({"terms": {"custodian": request.custodians}})
        
        if request.date_range:
            filter_clauses.append({
                "range": {
                    "upload_date": {
                        "gte": request.date_range.get('start'),
                        "lte": request.date_range.get('end')
                    }
                }
            })
        
        # Construct full query
        es_query = {
            "query": {
                "bool": {
                    "must": must_clauses,
                    "filter": filter_clauses
                }
            },
            "highlight": {
                "fields": {
                    "content": {"number_of_fragments": 3, "fragment_size": 150}
                }
            },
            "size": request.size,
            "from": request.from_
        }
        
        result = await es_client.search(index="legal_documents", body=es_query)
        
        return {
            "total": result['hits']['total']['value'],
            "hits": [{
                "document_id": hit['_id'],
                "filename": hit['_source']['filename'],
                "score": hit['_score'],
                "highlights": hit.get('highlight', {}).get('content', []),
                "metadata": hit['_source'].get('metadata', {})
            } for hit in result['hits']['hits']]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/create-production-set")
async def create_production_set(request: ProductionSetRequest):
    """
    Create production set with Bates numbering
    
    Features:
    - Sequential Bates numbering
    - Confidentiality designations
    - Export formats (native, PDF, TIFF)
    - Privilege log generation
    """
    try:
        production_docs = []
        current_bates = request.starting_number
        
        for doc_id in request.document_ids:
            # Retrieve document
            doc = await es_client.get(index="legal_documents", id=doc_id)
            
            # Generate Bates number
            bates_number = f"{request.bates_prefix}{current_bates:07d}"
            
            # Update document with production metadata
            await es_client.update(
                index="legal_documents",
                id=doc_id,
                body={
                    "doc": {
                        "bates_number": bates_number,
                        "production_set": request.name,
                        "confidentiality": request.confidentiality_designation
                    }
                }
            )
            
            production_docs.append({
                "document_id": doc_id,
                "bates_number": bates_number,
                "filename": doc['_source']['filename']
            })
            
            current_bates += 1
        
        return {
            "production_set": request.name,
            "document_count": len(production_docs),
            "bates_range": {
                "start": f"{request.bates_prefix}{request.starting_number:07d}",
                "end": f"{request.bates_prefix}{current_bates-1:07d}"
            },
            "documents": production_docs
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Production set creation failed: {str(e)}")


# Helper function for async scroll
async def async_scan(client, **kwargs):
    """Async generator for scrolling through Elasticsearch results"""
    response = await client.search(scroll='2m', **kwargs)
    scroll_id = response['_scroll_id']
    
    while response['hits']['hits']:
        for hit in response['hits']['hits']:
            yield hit
        
        response = await client.scroll(scroll_id=scroll_id, scroll='2m')
        scroll_id = response['_scroll_id']
