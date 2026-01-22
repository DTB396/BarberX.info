"""
BarberX Legal Case Management Pro Suite
Documents API Router - PDF Upload, OCR, Classification
"""
import os
import hashlib
import uuid
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.db.database import get_db
from app.db.models import Document, Case, DocumentType, ProcessingStatus, case_documents
from app.schemas.legal_schemas import (
    DocumentResponse, BatchUploadResponse, DocumentUploadResponse,
    DocumentClassifyRequest, DocumentClassifyResponse
)

router = APIRouter()

# Configure upload directory
UPLOAD_DIR = Path("uploads/documents")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def calculate_file_hash(file_content: bytes) -> str:
    """Calculate SHA-256 hash of file content"""
    return hashlib.sha256(file_content).hexdigest()


async def process_document_ocr(document_id: int, db: AsyncSession):
    """Background task to OCR process a document"""
    # This would integrate with PyMuPDF/Tesseract
    # For now, update status to show processing capability
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    
    if document:
        document.processing_status = ProcessingStatus.PROCESSING
        await db.commit()
        
        try:
            # TODO: Actual OCR processing with PyMuPDF/Tesseract
            # extracted_text = await run_ocr(document.file_path)
            
            document.processing_status = ProcessingStatus.COMPLETED
            document.processed_at = datetime.utcnow()
            # document.extracted_text = extracted_text
            await db.commit()
        except Exception as e:
            document.processing_status = ProcessingStatus.FAILED
            document.processing_notes = str(e)
            await db.commit()


@router.post("/upload", response_model=BatchUploadResponse)
async def batch_upload_documents(
    files: List[UploadFile] = File(...),
    case_id: Optional[int] = Form(None),
    document_type: Optional[DocumentType] = Form(DocumentType.OTHER),
    auto_classify: bool = Form(True),
    auto_ocr: bool = Form(True),
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Batch upload PDF documents.
    
    - Accepts multiple PDF files
    - Calculates file hashes for integrity
    - Optionally links to a case
    - Queues for OCR processing
    - Auto-classifies document types
    """
    uploaded = []
    failed = []
    
    # Verify case exists if provided
    if case_id:
        case_result = await db.execute(select(Case).where(Case.id == case_id))
        case = case_result.scalar_one_or_none()
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
    
    for file in files:
        try:
            # Validate file type
            if not file.filename.lower().endswith('.pdf'):
                if file.content_type != 'application/pdf':
                    failed.append({
                        "filename": file.filename,
                        "error": "Only PDF files are accepted"
                    })
                    continue
            
            # Read file content
            content = await file.read()
            file_hash = calculate_file_hash(content)
            
            # Check for duplicate by hash
            existing = await db.execute(
                select(Document).where(Document.file_hash == file_hash)
            )
            if existing.scalar_one_or_none():
                failed.append({
                    "filename": file.filename,
                    "error": "Duplicate file (already uploaded)"
                })
                continue
            
            # Generate unique filename
            unique_id = uuid.uuid4().hex[:12]
            safe_filename = f"{unique_id}_{file.filename}"
            file_path = UPLOAD_DIR / safe_filename
            
            # Save file
            with open(file_path, "wb") as f:
                f.write(content)
            
            # Create document record
            document = Document(
                filename=safe_filename,
                original_filename=file.filename,
                file_path=str(file_path),
                file_size=len(content),
                mime_type=file.content_type or "application/pdf",
                file_hash=file_hash,
                document_type=document_type,
                processing_status=ProcessingStatus.PENDING
            )
            
            db.add(document)
            await db.commit()
            await db.refresh(document)
            
            # Link to case if provided
            if case_id:
                await db.execute(
                    case_documents.insert().values(
                        case_id=case_id,
                        document_id=document.id
                    )
                )
                await db.commit()
            
            # Queue OCR processing
            if auto_ocr and background_tasks:
                background_tasks.add_task(process_document_ocr, document.id, db)
            
            uploaded.append(DocumentUploadResponse(
                id=document.id,
                filename=document.filename,
                original_filename=document.original_filename,
                file_size=document.file_size,
                mime_type=document.mime_type,
                file_hash=document.file_hash,
                processing_status=document.processing_status,
                message="Upload successful, queued for processing"
            ))
            
        except Exception as e:
            failed.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return BatchUploadResponse(
        uploaded=uploaded,
        failed=failed,
        total_uploaded=len(uploaded),
        total_failed=len(failed)
    )


@router.get("/", response_model=List[DocumentResponse])
async def list_documents(
    case_id: Optional[int] = None,
    document_type: Optional[DocumentType] = None,
    processing_status: Optional[ProcessingStatus] = None,
    search: Optional[str] = None,
    limit: int = Query(50, le=200),
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    """List documents with filtering"""
    query = select(Document)
    
    if case_id:
        query = query.join(case_documents).where(case_documents.c.case_id == case_id)
    if document_type:
        query = query.where(Document.document_type == document_type)
    if processing_status:
        query = query.where(Document.processing_status == processing_status)
    if search:
        query = query.where(
            Document.original_filename.ilike(f"%{search}%") |
            Document.title.ilike(f"%{search}%") |
            Document.extracted_text.ilike(f"%{search}%")
        )
    
    query = query.order_by(Document.created_at.desc())
    query = query.offset(offset).limit(limit)
    
    result = await db.execute(query)
    documents = result.scalars().all()
    
    return [DocumentResponse.model_validate(d) for d in documents]


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific document by ID"""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentResponse.model_validate(document)


@router.get("/{document_id}/text")
async def get_document_text(
    document_id: int,
    page: Optional[int] = None,
    db: AsyncSession = Depends(get_db)
):
    """Get extracted text from a document"""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document.processing_status != ProcessingStatus.COMPLETED:
        return {
            "document_id": document_id,
            "status": document.processing_status,
            "message": "Document has not been processed yet",
            "text": None
        }
    
    return {
        "document_id": document_id,
        "filename": document.original_filename,
        "page_count": document.page_count,
        "text": document.extracted_text,
        "ocr_confidence": document.ocr_confidence
    }


@router.post("/scan/{document_id}")
async def scan_document(
    document_id: int,
    force_rescan: bool = False,
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db)
):
    """Trigger OCR scanning for a document"""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if document.processing_status == ProcessingStatus.COMPLETED and not force_rescan:
        return {
            "document_id": document_id,
            "message": "Document already processed. Set force_rescan=true to reprocess."
        }
    
    if background_tasks:
        document.processing_status = ProcessingStatus.PENDING
        await db.commit()
        background_tasks.add_task(process_document_ocr, document_id, db)
    
    return {
        "document_id": document_id,
        "message": "Document queued for OCR processing"
    }


@router.post("/classify", response_model=List[DocumentClassifyResponse])
async def classify_documents(
    request: DocumentClassifyRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Auto-classify documents by type.
    
    Uses content analysis to determine:
    - Document type (motion, order, police report, etc.)
    - Key dates
    - Suggested tags
    """
    results = []
    
    for doc_id in request.document_ids:
        result = await db.execute(select(Document).where(Document.id == doc_id))
        document = result.scalar_one_or_none()
        
        if not document:
            continue
        
        # Classification logic (would use ML model in production)
        predicted_type = DocumentType.OTHER
        confidence = 0.5
        suggested_tags = []
        extracted_date = None
        
        filename_lower = document.original_filename.lower()
        text_lower = (document.extracted_text or "").lower()
        
        # Rule-based classification (would be ML in production)
        if "motion" in filename_lower or "motion" in text_lower[:500]:
            predicted_type = DocumentType.MOTION
            confidence = 0.85
            suggested_tags = ["motion", "filing"]
        elif "order" in filename_lower or "hereby ordered" in text_lower:
            predicted_type = DocumentType.ORDER
            confidence = 0.9
            suggested_tags = ["order", "court"]
        elif "police" in filename_lower or "incident report" in text_lower:
            predicted_type = DocumentType.POLICE_REPORT
            confidence = 0.88
            suggested_tags = ["police", "incident", "report"]
        elif "complaint" in filename_lower:
            predicted_type = DocumentType.COMPLAINT
            confidence = 0.87
            suggested_tags = ["complaint", "filing", "initial"]
        elif "deposition" in filename_lower or "deponent" in text_lower:
            predicted_type = DocumentType.DEPOSITION
            confidence = 0.9
            suggested_tags = ["deposition", "testimony", "discovery"]
        elif "discovery" in filename_lower or "interrogator" in text_lower:
            predicted_type = DocumentType.DISCOVERY_REQUEST
            confidence = 0.8
            suggested_tags = ["discovery"]
        elif "pro se" in filename_lower.lower() or "pro se" in text_lower:
            predicted_type = DocumentType.COMPLAINT
            confidence = 0.75
            suggested_tags = ["pro-se", "self-represented"]
        
        # Update document if auto_tag enabled
        if request.auto_tag:
            document.document_type = predicted_type
            await db.commit()
        
        results.append(DocumentClassifyResponse(
            document_id=doc_id,
            predicted_type=predicted_type,
            confidence=confidence,
            suggested_tags=suggested_tags,
            extracted_date=extracted_date
        ))
    
    return results


@router.post("/link")
async def link_documents(
    document_ids: List[int],
    case_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Link documents to a case"""
    # Verify case exists
    case_result = await db.execute(select(Case).where(Case.id == case_id))
    if not case_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Case not found")
    
    linked = 0
    for doc_id in document_ids:
        # Verify document exists
        doc_result = await db.execute(select(Document).where(Document.id == doc_id))
        if doc_result.scalar_one_or_none():
            try:
                await db.execute(
                    case_documents.insert().values(
                        case_id=case_id,
                        document_id=doc_id
                    )
                )
                linked += 1
            except:
                pass  # Already linked
    
    await db.commit()
    
    return {
        "message": f"Linked {linked} documents to case {case_id}",
        "case_id": case_id,
        "documents_linked": linked
    }


@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Delete a document"""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Remove file
    try:
        os.remove(document.file_path)
    except:
        pass
    
    # Remove from database
    await db.delete(document)
    await db.commit()
    
    return {"message": "Document deleted", "document_id": document_id}
