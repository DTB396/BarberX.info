"""
BarberX Legal Case Management Pro Suite
AI Service - OpenAI GPT-5.2 Integration for Document Analysis
"""
import os
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class AIModel(str, Enum):
    """Available AI models"""
    GPT_5_2 = "gpt-5.2"
    GPT_5_2_TURBO = "gpt-5.2-turbo"
    GPT_4O = "gpt-4o"
    GPT_4_TURBO = "gpt-4-turbo"


@dataclass
class DocumentAnalysis:
    """Result of AI document analysis"""
    document_id: Optional[int] = None
    filename: str = ""
    summary: str = ""
    document_type: str = ""
    key_facts: List[str] = field(default_factory=list)
    violations_detected: List[Dict[str, Any]] = field(default_factory=list)
    parties_mentioned: List[str] = field(default_factory=list)
    dates_mentioned: List[str] = field(default_factory=list)
    legal_citations: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    analysis_timestamp: str = ""
    model_used: str = ""


@dataclass
class ConversationMessage:
    """A message in an AI conversation"""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: str = ""
    document_refs: List[str] = field(default_factory=list)


@dataclass
class AIConversation:
    """An AI conversation session"""
    conversation_id: str
    case_id: Optional[int] = None
    model: str = AIModel.GPT_5_2.value
    messages: List[ConversationMessage] = field(default_factory=list)
    document_context: List[DocumentAnalysis] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


class AIService:
    """
    OpenAI GPT-5.2 integration for legal document analysis.
    
    Features:
    - Batch PDF analysis with constitutional violation detection
    - Interactive conversations about uploaded documents
    - Legal citation extraction and case law matching
    - Damage assessment suggestions
    """
    
    SYSTEM_PROMPT = """You are an expert legal analyst specializing in civil rights litigation, 
police misconduct cases, and constitutional law. Your role is to:

1. Analyze legal documents for constitutional violations (4th, 5th, 6th, 8th, 14th Amendments)
2. Identify excessive force, false arrest, malicious prosecution patterns
3. Extract key facts, dates, parties, and legal citations
4. Suggest relevant case law and legal arguments
5. Provide damage assessment guidance based on precedent
6. Flag Brady/Giglio material and discovery opportunities

Always cite specific constitutional provisions and relevant case law.
Be thorough but concise. Focus on actionable legal insights.

When analyzing police reports or BWC transcripts, pay special attention to:
- Officer actions that may constitute excessive force
- Statements that suggest pretextual stops or racial profiling
- Evidence of procedural violations
- Potential exculpatory evidence that may have been withheld"""

    def __init__(self, api_key: Optional[str] = None, model: str = AIModel.GPT_5_2.value):
        """Initialize AI service with OpenAI API key"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None
        self.conversations: Dict[str, AIConversation] = {}
        
        if OPENAI_AVAILABLE and self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key)
    
    @property
    def is_available(self) -> bool:
        """Check if AI service is available"""
        return OPENAI_AVAILABLE and self.client is not None
    
    async def analyze_document(
        self, 
        text: str, 
        filename: str,
        document_id: Optional[int] = None,
        additional_context: Optional[str] = None
    ) -> DocumentAnalysis:
        """
        Analyze a single document with GPT-5.2.
        
        Args:
            text: Extracted text from document
            filename: Original filename
            document_id: Database ID if available
            additional_context: Extra context about the case
        
        Returns:
            DocumentAnalysis with AI insights
        """
        if not self.is_available:
            return self._fallback_analysis(text, filename, document_id)
        
        prompt = f"""Analyze this legal document and provide a structured analysis.

Filename: {filename}
{f'Context: {additional_context}' if additional_context else ''}

Document Text:
{text[:15000]}  # Limit to ~15k chars for context window

Provide your analysis in the following JSON format:
{{
    "summary": "Brief 2-3 sentence summary",
    "document_type": "complaint|motion|order|police_report|deposition|etc",
    "key_facts": ["fact1", "fact2", ...],
    "violations_detected": [
        {{
            "type": "4th_excessive_force|5th_miranda|14th_due_process|etc",
            "description": "What happened",
            "severity": 1-5,
            "confidence": 0.0-1.0,
            "supporting_text": "Relevant quote from document"
        }}
    ],
    "parties_mentioned": ["Party Name (role)", ...],
    "dates_mentioned": ["2025-11-29 - incident date", ...],
    "legal_citations": ["Case Name, Citation (year)", ...],
    "recommended_actions": ["action1", "action2", ...],
    "confidence_score": 0.0-1.0
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return DocumentAnalysis(
                document_id=document_id,
                filename=filename,
                summary=result.get("summary", ""),
                document_type=result.get("document_type", "other"),
                key_facts=result.get("key_facts", []),
                violations_detected=result.get("violations_detected", []),
                parties_mentioned=result.get("parties_mentioned", []),
                dates_mentioned=result.get("dates_mentioned", []),
                legal_citations=result.get("legal_citations", []),
                recommended_actions=result.get("recommended_actions", []),
                confidence_score=result.get("confidence_score", 0.0),
                analysis_timestamp=datetime.utcnow().isoformat(),
                model_used=self.model
            )
            
        except Exception as e:
            # Log error and return fallback analysis
            print(f"AI analysis error: {e}")
            return self._fallback_analysis(text, filename, document_id)
    
    async def analyze_batch(
        self, 
        documents: List[Dict[str, Any]],
        case_context: Optional[str] = None
    ) -> List[DocumentAnalysis]:
        """
        Analyze multiple documents in batch.
        
        Args:
            documents: List of {"text": str, "filename": str, "id": int}
            case_context: Context about the overall case
        
        Returns:
            List of DocumentAnalysis results
        """
        results = []
        
        # Process concurrently with rate limiting
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent requests
        
        async def analyze_with_limit(doc):
            async with semaphore:
                return await self.analyze_document(
                    text=doc["text"],
                    filename=doc["filename"],
                    document_id=doc.get("id"),
                    additional_context=case_context
                )
        
        tasks = [analyze_with_limit(doc) for doc in documents]
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def start_conversation(
        self, 
        conversation_id: str,
        case_id: Optional[int] = None,
        document_analyses: Optional[List[DocumentAnalysis]] = None,
        initial_context: Optional[str] = None
    ) -> AIConversation:
        """
        Start a new AI conversation about documents.
        
        Args:
            conversation_id: Unique ID for this conversation
            case_id: Associated case ID
            document_analyses: Pre-analyzed documents for context
            initial_context: Additional context to include
        
        Returns:
            New AIConversation object
        """
        conversation = AIConversation(
            conversation_id=conversation_id,
            case_id=case_id,
            model=self.model,
            document_context=document_analyses or [],
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat()
        )
        
        # Build context from document analyses
        if document_analyses:
            context_parts = []
            for analysis in document_analyses:
                context_parts.append(f"""
Document: {analysis.filename}
Type: {analysis.document_type}
Summary: {analysis.summary}
Key Facts: {', '.join(analysis.key_facts[:5])}
Violations: {', '.join([v.get('type', '') for v in analysis.violations_detected])}
""")
            
            doc_context = "\n---\n".join(context_parts)
            
            # Add system context message
            conversation.messages.append(ConversationMessage(
                role="system",
                content=f"{self.SYSTEM_PROMPT}\n\n--- DOCUMENT CONTEXT ---\n{doc_context}",
                timestamp=datetime.utcnow().isoformat()
            ))
        else:
            conversation.messages.append(ConversationMessage(
                role="system",
                content=self.SYSTEM_PROMPT,
                timestamp=datetime.utcnow().isoformat()
            ))
        
        self.conversations[conversation_id] = conversation
        return conversation
    
    async def send_message(
        self, 
        conversation_id: str, 
        user_message: str,
        document_refs: Optional[List[str]] = None
    ) -> str:
        """
        Send a message in an existing conversation.
        
        Args:
            conversation_id: ID of the conversation
            user_message: User's message
            document_refs: References to specific documents
        
        Returns:
            AI assistant's response
        """
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        conversation = self.conversations[conversation_id]
        
        # Add user message
        conversation.messages.append(ConversationMessage(
            role="user",
            content=user_message,
            timestamp=datetime.utcnow().isoformat(),
            document_refs=document_refs or []
        ))
        
        if not self.is_available:
            fallback_response = "AI service is not available. Please configure OPENAI_API_KEY."
            conversation.messages.append(ConversationMessage(
                role="assistant",
                content=fallback_response,
                timestamp=datetime.utcnow().isoformat()
            ))
            return fallback_response
        
        try:
            # Build messages for API
            api_messages = [
                {"role": m.role, "content": m.content}
                for m in conversation.messages
            ]
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                temperature=0.5,
                max_tokens=2000
            )
            
            assistant_message = response.choices[0].message.content
            
            # Add assistant response to conversation
            conversation.messages.append(ConversationMessage(
                role="assistant",
                content=assistant_message,
                timestamp=datetime.utcnow().isoformat()
            ))
            
            conversation.updated_at = datetime.utcnow().isoformat()
            
            return assistant_message
            
        except Exception as e:
            error_response = f"Error communicating with AI: {str(e)}"
            conversation.messages.append(ConversationMessage(
                role="assistant",
                content=error_response,
                timestamp=datetime.utcnow().isoformat()
            ))
            return error_response
    
    async def stream_message(
        self, 
        conversation_id: str, 
        user_message: str
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response for real-time display.
        
        Args:
            conversation_id: ID of the conversation
            user_message: User's message
        
        Yields:
            Chunks of the AI response
        """
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        conversation = self.conversations[conversation_id]
        
        # Add user message
        conversation.messages.append(ConversationMessage(
            role="user",
            content=user_message,
            timestamp=datetime.utcnow().isoformat()
        ))
        
        if not self.is_available:
            yield "AI service is not available. Please configure OPENAI_API_KEY."
            return
        
        try:
            api_messages = [
                {"role": m.role, "content": m.content}
                for m in conversation.messages
            ]
            
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=api_messages,
                temperature=0.5,
                max_tokens=2000,
                stream=True
            )
            
            full_response = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
            
            # Add complete response to conversation
            conversation.messages.append(ConversationMessage(
                role="assistant",
                content=full_response,
                timestamp=datetime.utcnow().isoformat()
            ))
            
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def get_conversation(self, conversation_id: str) -> Optional[AIConversation]:
        """Get a conversation by ID"""
        return self.conversations.get(conversation_id)
    
    def _fallback_analysis(
        self, 
        text: str, 
        filename: str, 
        document_id: Optional[int] = None
    ) -> DocumentAnalysis:
        """
        Fallback keyword-based analysis when AI is unavailable.
        """
        text_lower = text.lower()
        
        # Simple keyword detection
        violations = []
        
        if any(word in text_lower for word in ["struck", "punched", "force", "tased", "shot"]):
            violations.append({
                "type": "4th_excessive_force",
                "description": "Potential excessive force indicators found",
                "severity": 3,
                "confidence": 0.5
            })
        
        if any(word in text_lower for word in ["without warrant", "no consent", "searched"]):
            violations.append({
                "type": "4th_unlawful_search",
                "description": "Potential unlawful search indicators found",
                "severity": 2,
                "confidence": 0.4
            })
        
        return DocumentAnalysis(
            document_id=document_id,
            filename=filename,
            summary="Automated keyword analysis (AI unavailable)",
            document_type="unknown",
            violations_detected=violations,
            confidence_score=0.3,
            analysis_timestamp=datetime.utcnow().isoformat(),
            model_used="keyword_fallback"
        )


# Singleton instance
ai_service = AIService()
