"""
BarberX Legal Case Management Pro Suite
AI API Router - GPT-5.2 Document Analysis & Conversations
"""
import uuid
from datetime import datetime
from typing import List, Optional
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.database import get_db
from app.db.models import Document, Case, AnalysisJob, ProcessingStatus
from app.services.ai_service import ai_service, DocumentAnalysis, AIConversation

router = APIRouter()


# ============================================================
# PYDANTIC SCHEMAS
# ============================================================

class BatchAnalysisRequest(BaseModel):
    """Request for batch PDF analysis"""
    document_ids: List[int] = Field(..., description="List of document IDs to analyze")
    case_id: Optional[int] = Field(None, description="Associated case ID for context")
    additional_context: Optional[str] = Field(None, description="Extra context about the case")
    start_conversation: bool = Field(True, description="Start AI conversation after analysis")


class BatchAnalysisResponse(BaseModel):
    """Response from batch analysis"""
    analyses: List[dict]
    conversation_id: Optional[str] = None
    total_documents: int
    violations_found: int
    processing_time_ms: int


class StartConversationRequest(BaseModel):
    """Request to start a new AI conversation"""
    case_id: Optional[int] = None
    document_ids: Optional[List[int]] = None
    initial_message: Optional[str] = None


class StartConversationResponse(BaseModel):
    """Response when starting conversation"""
    conversation_id: str
    model: str
    created_at: str
    initial_response: Optional[str] = None


class SendMessageRequest(BaseModel):
    """Request to send message in conversation"""
    message: str
    document_refs: Optional[List[str]] = None


class SendMessageResponse(BaseModel):
    """Response from sending message"""
    response: str
    conversation_id: str
    timestamp: str


class ConversationHistoryResponse(BaseModel):
    """Conversation history"""
    conversation_id: str
    case_id: Optional[int]
    model: str
    messages: List[dict]
    document_context: List[dict]
    created_at: str
    updated_at: str


class AIStatusResponse(BaseModel):
    """AI service status"""
    available: bool
    model: str
    active_conversations: int


# ============================================================
# ENDPOINTS
# ============================================================

@router.get("/status", response_model=AIStatusResponse)
async def get_ai_status():
    """
    Check if AI service is available and configured.
    """
    return AIStatusResponse(
        available=ai_service.is_available,
        model=ai_service.model,
        active_conversations=len(ai_service.conversations)
    )


@router.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_documents_batch(
    request: BatchAnalysisRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Analyze multiple documents with GPT-5.2.
    
    - Extracts constitutional violations
    - Identifies key facts and dates
    - Finds legal citations
    - Optionally starts an AI conversation for follow-up questions
    
    Returns analysis results and conversation ID for chat interface.
    """
    start_time = datetime.utcnow()
    
    # Fetch documents from database
    result = await db.execute(
        select(Document).where(Document.id.in_(request.document_ids))
    )
    documents = result.scalars().all()
    
    if not documents:
        raise HTTPException(status_code=404, detail="No documents found")
    
    # Get case context if provided
    case_context = request.additional_context
    if request.case_id:
        case_result = await db.execute(select(Case).where(Case.id == request.case_id))
        case = case_result.scalar_one_or_none()
        if case:
            case_context = f"Case: {case.docket_number or case.title}. {case.description or ''}"
    
    # Prepare documents for analysis
    docs_for_analysis = [
        {
            "text": doc.extracted_text or doc.raw_content or "",
            "filename": doc.filename,
            "id": doc.id
        }
        for doc in documents
        if doc.extracted_text or doc.raw_content
    ]
    
    if not docs_for_analysis:
        raise HTTPException(
            status_code=400, 
            detail="No document text available. Run OCR first."
        )
    
    # Run batch analysis
    analyses = await ai_service.analyze_batch(docs_for_analysis, case_context)
    
    # Count violations
    total_violations = sum(
        len(a.violations_detected) for a in analyses
    )
    
    # Optionally start conversation
    conversation_id = None
    if request.start_conversation:
        conversation_id = str(uuid.uuid4())
        await ai_service.start_conversation(
            conversation_id=conversation_id,
            case_id=request.case_id,
            document_analyses=analyses,
            initial_context=case_context
        )
    
    # Calculate processing time
    processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
    
    return BatchAnalysisResponse(
        analyses=[asdict(a) for a in analyses],
        conversation_id=conversation_id,
        total_documents=len(analyses),
        violations_found=total_violations,
        processing_time_ms=processing_time
    )


@router.post("/analyze/upload")
async def analyze_uploaded_files(
    files: List[UploadFile] = File(...),
    case_id: Optional[int] = Form(None),
    context: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload and immediately analyze PDF files with GPT-5.2.
    
    This endpoint combines upload + OCR + AI analysis in one step.
    For previously uploaded documents, use /analyze/batch.
    """
    # Import here to avoid circular imports
    from app.processors.pdf_processor import extract_text_from_pdf
    
    analyses = []
    
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            continue
            
        content = await file.read()
        
        # Extract text (simplified - real impl would use PyMuPDF)
        try:
            text = extract_text_from_pdf(content)
        except Exception:
            text = ""
        
        if text:
            analysis = await ai_service.analyze_document(
                text=text,
                filename=file.filename,
                additional_context=context
            )
            analyses.append(asdict(analysis))
    
    # Start conversation with results
    conversation_id = str(uuid.uuid4())
    if analyses:
        await ai_service.start_conversation(
            conversation_id=conversation_id,
            case_id=case_id,
            document_analyses=[DocumentAnalysis(**a) for a in analyses]
        )
    
    return {
        "analyses": analyses,
        "conversation_id": conversation_id,
        "files_processed": len(analyses)
    }


@router.post("/conversation/start", response_model=StartConversationResponse)
async def start_conversation(
    request: StartConversationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Start a new AI conversation.
    
    Optionally include document IDs to load their analyses as context.
    """
    conversation_id = str(uuid.uuid4())
    
    # Load document analyses if provided
    document_analyses = None
    if request.document_ids:
        result = await db.execute(
            select(Document).where(Document.id.in_(request.document_ids))
        )
        documents = result.scalars().all()
        
        # Create quick analyses for context
        document_analyses = [
            DocumentAnalysis(
                document_id=doc.id,
                filename=doc.filename,
                summary=doc.description or "",
                document_type=doc.document_type.value if doc.document_type else "other"
            )
            for doc in documents
        ]
    
    # Start conversation
    conversation = await ai_service.start_conversation(
        conversation_id=conversation_id,
        case_id=request.case_id,
        document_analyses=document_analyses
    )
    
    # Send initial message if provided
    initial_response = None
    if request.initial_message:
        initial_response = await ai_service.send_message(
            conversation_id=conversation_id,
            user_message=request.initial_message
        )
    
    return StartConversationResponse(
        conversation_id=conversation_id,
        model=conversation.model,
        created_at=conversation.created_at,
        initial_response=initial_response
    )


@router.post("/conversation/{conversation_id}/message", response_model=SendMessageResponse)
async def send_message(
    conversation_id: str,
    request: SendMessageRequest
):
    """
    Send a message in an existing conversation.
    
    The AI will respond with context from previously analyzed documents.
    """
    try:
        response = await ai_service.send_message(
            conversation_id=conversation_id,
            user_message=request.message,
            document_refs=request.document_refs
        )
        
        return SendMessageResponse(
            response=response,
            conversation_id=conversation_id,
            timestamp=datetime.utcnow().isoformat()
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/conversation/{conversation_id}/stream")
async def stream_message(
    conversation_id: str,
    message: str
):
    """
    Stream AI response in real-time.
    
    Use this for a ChatGPT-like typing effect in the UI.
    """
    async def generate():
        async for chunk in ai_service.stream_message(conversation_id, message):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@router.get("/conversation/{conversation_id}", response_model=ConversationHistoryResponse)
async def get_conversation(conversation_id: str):
    """
    Get conversation history.
    """
    conversation = ai_service.get_conversation(conversation_id)
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return ConversationHistoryResponse(
        conversation_id=conversation.conversation_id,
        case_id=conversation.case_id,
        model=conversation.model,
        messages=[
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
                "document_refs": m.document_refs
            }
            for m in conversation.messages
            if m.role != "system"  # Don't expose system prompt
        ],
        document_context=[asdict(d) for d in conversation.document_context],
        created_at=conversation.created_at,
        updated_at=conversation.updated_at
    )


@router.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation.
    """
    if conversation_id in ai_service.conversations:
        del ai_service.conversations[conversation_id]
        return {"status": "deleted", "conversation_id": conversation_id}
    
    raise HTTPException(status_code=404, detail="Conversation not found")


@router.post("/analyze/text")
async def analyze_raw_text(
    text: str = Form(...),
    filename: str = Form("unnamed_document.txt"),
    context: Optional[str] = Form(None)
):
    """
    Analyze raw text (for pasted content or quick analysis).
    """
    analysis = await ai_service.analyze_document(
        text=text,
        filename=filename,
        additional_context=context
    )
    
    return asdict(analysis)
