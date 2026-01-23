"""
BarberX Legal Case Management Pro Suite
Enhanced AI Service - Infinite Context & Advanced Legal Analysis
"""
import os
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class ComplaintSection(str, Enum):
    """Sections of a verified complaint"""
    CAPTION = "caption"
    JURISDICTION = "jurisdiction"
    PARTIES = "parties"
    FACTS = "facts"
    CAUSES_OF_ACTION = "causes_of_action"
    DAMAGES = "damages"
    PRAYER = "prayer"
    VERIFICATION = "verification"


@dataclass
class LegalResearchResult:
    """Result from legal research query"""
    query: str = ""
    relevant_cases: List[Dict[str, Any]] = field(default_factory=list)
    statutes: List[str] = field(default_factory=list)
    constitutional_provisions: List[str] = field(default_factory=list)
    secondary_sources: List[str] = field(default_factory=list)
    research_summary: str = ""
    confidence_score: float = 0.0


@dataclass
class ComplaintDraft:
    """AI-generated verified complaint"""
    case_id: Optional[int] = None
    case_caption: str = ""
    docket_number: str = ""
    plaintiff_name: str = ""
    defendant_names: List[str] = field(default_factory=list)
    
    # Sections
    jurisdiction_venue: str = ""
    parties_section: str = ""
    factual_allegations: List[str] = field(default_factory=list)
    causes_of_action: List[Dict[str, Any]] = field(default_factory=list)
    damages_claimed: Dict[str, Any] = field(default_factory=dict)
    prayer_for_relief: List[str] = field(default_factory=list)
    verification: str = ""
    
    # Exhibits
    exhibits: List[str] = field(default_factory=list)
    
    # Metadata
    generated_at: str = ""
    model_used: str = ""
    confidence_score: float = 0.0


@dataclass
class ConversationMemory:
    """Persistent conversation memory with infinite context"""
    conversation_id: str
    total_messages: int = 0
    key_topics: List[str] = field(default_factory=list)
    entities_mentioned: Dict[str, List[str]] = field(default_factory=dict)  # {"officers": [...], "dates": [...]}
    decisions_made: List[str] = field(default_factory=list)
    pending_questions: List[str] = field(default_factory=list)
    document_summaries: Dict[str, str] = field(default_factory=dict)  # {doc_id: summary}
    context_hash: str = ""
    last_summary: str = ""
    created_at: str = ""
    updated_at: str = ""


class EnhancedAIService:
    """
    Enhanced AI service with infinite context, legal research, and automated drafting.
    
    Advanced Features:
    - Infinite context through conversation summarization
    - Multi-document cross-referencing
    - Legal research with case law matching
    - Automated verified complaint generation
    - Streaming responses for real-time feedback
    - Persistent conversation memory
    - Constitutional violation pattern detection
    - Damages calculation based on precedent
    """
    
    LEGAL_RESEARCH_PROMPT = """You are an expert legal researcher specializing in civil rights litigation.
Your role is to find relevant case law, statutes, and constitutional provisions for the given query.

Focus on:
1. Supreme Court precedents (especially § 1983, qualified immunity, excessive force)
2. Circuit court decisions from relevant jurisdictions
3. Applicable constitutional amendments and provisions
4. Federal statutes (42 U.S.C. § 1983, § 1985, § 1988)
5. State tort law where applicable

Provide full citations in Bluebook format.
Include brief summaries of key holdings.
Note any circuit splits or evolving doctrines."""

    COMPLAINT_DRAFTING_PROMPT = """You are an expert civil rights attorney drafting a verified complaint.

Requirements:
1. Follow Federal Rules of Civil Procedure (Rule 8, Rule 11)
2. Use plain statement pleading standard (Twombly/Iqbal)
3. Be specific with dates, locations, and factual allegations
4. Clearly state each cause of action with legal elements
5. Calculate damages based on similar precedents
6. Include proper verification language
7. Cite constitutional provisions and case law
8. Use professional, persuasive language

Draft for filing in U.S. District Court or State Superior Court as appropriate."""

    INFINITE_CONTEXT_PROMPT = """You are maintaining an infinite context conversation about a legal case.

Your task is to:
1. Summarize the conversation so far into key points
2. Extract and track: parties, dates, violations, evidence, decisions
3. Note any pending questions or action items
4. Create concise summaries of analyzed documents
5. Maintain continuity across unlimited messages

Return a structured summary that preserves all critical information while reducing token usage."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """Initialize enhanced AI service"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None
        self.conversation_memories: Dict[str, ConversationMemory] = {}
        
        if OPENAI_AVAILABLE and self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key)
    
    @property
    def is_available(self) -> bool:
        """Check if AI service is available"""
        return OPENAI_AVAILABLE and self.client is not None
    
    async def research_legal_issue(
        self,
        query: str,
        jurisdiction: str = "federal",
        focus_areas: Optional[List[str]] = None
    ) -> LegalResearchResult:
        """
        Conduct legal research for a specific issue.
        
        Args:
            query: Legal question or issue to research
            jurisdiction: "federal", "new_jersey", "third_circuit", etc.
            focus_areas: Specific areas to emphasize (e.g., ["qualified_immunity", "excessive_force"])
        
        Returns:
            LegalResearchResult with relevant case law and analysis
        """
        if not self.is_available:
            return self._fallback_research(query)
        
        focus_text = f"\n\nFocus particularly on: {', '.join(focus_areas)}" if focus_areas else ""
        jurisdiction_text = f"\nJurisdiction: {jurisdiction}"
        
        prompt = f"""{self.LEGAL_RESEARCH_PROMPT}

Research Query: {query}{jurisdiction_text}{focus_text}

Provide your research in JSON format:
{{
    "relevant_cases": [
        {{
            "citation": "Full Bluebook citation",
            "name": "Case Name",
            "year": 2023,
            "court": "Supreme Court / 3rd Circuit / etc.",
            "holding": "Brief summary of holding",
            "relevance": "How this applies to the query",
            "key_quote": "Most relevant quote from opinion"
        }}
    ],
    "statutes": ["42 U.S.C. § 1983 - Civil Rights", ...],
    "constitutional_provisions": ["Fourth Amendment - Unreasonable Search and Seizure", ...],
    "secondary_sources": ["Law review articles, treatises", ...],
    "research_summary": "Overall analysis and application to query",
    "confidence_score": 0.0-1.0
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert legal researcher."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return LegalResearchResult(
                query=query,
                relevant_cases=result.get("relevant_cases", []),
                statutes=result.get("statutes", []),
                constitutional_provisions=result.get("constitutional_provisions", []),
                secondary_sources=result.get("secondary_sources", []),
                research_summary=result.get("research_summary", ""),
                confidence_score=result.get("confidence_score", 0.0)
            )
            
        except Exception as e:
            print(f"Legal research error: {e}")
            return self._fallback_research(query)
    
    async def generate_verified_complaint(
        self,
        case_data: Dict[str, Any],
        violations: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
        jurisdiction: str = "federal",
        include_state_claims: bool = False
    ) -> ComplaintDraft:
        """
        Generate a complete verified complaint ready for filing.
        
        Args:
            case_data: {"plaintiff": str, "defendants": List[str], "incident_date": str, "location": str}
            violations: List of detected constitutional violations
            documents: List of supporting documents with summaries
            jurisdiction: "federal" or "state"
            include_state_claims: Whether to include state tort claims
        
        Returns:
            ComplaintDraft with all sections populated
        """
        if not self.is_available:
            return self._fallback_complaint(case_data)
        
        # Build comprehensive prompt
        violations_text = json.dumps(violations, indent=2)
        documents_text = "\n".join([f"- {doc['filename']}: {doc['summary']}" for doc in documents])
        
        prompt = f"""{self.COMPLAINT_DRAFTING_PROMPT}

CASE INFORMATION:
Plaintiff: {case_data.get('plaintiff', 'PLAINTIFF_NAME')}
Defendants: {', '.join(case_data.get('defendants', []))}
Incident Date: {case_data.get('incident_date', 'DATE')}
Location: {case_data.get('location', 'LOCATION')}
Jurisdiction: {jurisdiction.upper()}

VIOLATIONS DETECTED:
{violations_text}

SUPPORTING DOCUMENTS:
{documents_text}

{'Include both federal § 1983 claims AND state tort claims (assault, battery, false imprisonment).' if include_state_claims else 'Federal § 1983 claims only.'}

Generate a complete verified complaint with the following structure in JSON format:
{{
    "case_caption": "PLAINTIFF NAME v. DEFENDANT NAMES",
    "docket_number": "To be assigned by Clerk",
    "jurisdiction_venue": "Full jurisdiction and venue section with legal basis",
    "parties_section": "Detailed description of all parties",
    "factual_allegations": ["Paragraph 1", "Paragraph 2", ...],
    "causes_of_action": [
        {{
            "count": "COUNT I",
            "title": "42 U.S.C. § 1983 - Fourth Amendment Excessive Force",
            "elements": ["Element 1", "Element 2", ...],
            "factual_basis": "How facts satisfy elements",
            "legal_citations": ["Graham v. Connor, 490 U.S. 386", ...]
        }}
    ],
    "damages_claimed": {{
        "compensatory": "Specific damages with amounts",
        "punitive": "Punitive damages justification",
        "total_range": "$XXX,XXX - $XXX,XXX"
    }},
    "prayer_for_relief": ["Relief item 1", "Relief item 2", ...],
    "verification": "Verification statement for plaintiff signature",
    "exhibits": ["Exhibit A - Police Report", "Exhibit B - Medical Records", ...],
    "confidence_score": 0.0-1.0
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert civil rights attorney."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return ComplaintDraft(
                case_id=case_data.get('case_id'),
                case_caption=result.get("case_caption", ""),
                docket_number=result.get("docket_number", ""),
                plaintiff_name=case_data.get('plaintiff', ''),
                defendant_names=case_data.get('defendants', []),
                jurisdiction_venue=result.get("jurisdiction_venue", ""),
                parties_section=result.get("parties_section", ""),
                factual_allegations=result.get("factual_allegations", []),
                causes_of_action=result.get("causes_of_action", []),
                damages_claimed=result.get("damages_claimed", {}),
                prayer_for_relief=result.get("prayer_for_relief", []),
                verification=result.get("verification", ""),
                exhibits=result.get("exhibits", []),
                generated_at=datetime.utcnow().isoformat(),
                model_used=self.model,
                confidence_score=result.get("confidence_score", 0.0)
            )
            
        except Exception as e:
            print(f"Complaint generation error: {e}")
            return self._fallback_complaint(case_data)
    
    async def stream_analysis(
        self,
        text: str,
        analysis_type: str = "constitutional_violations"
    ) -> AsyncGenerator[str, None]:
        """
        Stream analysis results in real-time for UI responsiveness.
        
        Args:
            text: Document text to analyze
            analysis_type: Type of analysis to perform
        
        Yields:
            JSON chunks as they're generated
        """
        if not self.is_available:
            yield json.dumps({"error": "AI service unavailable"})
            return
        
        prompt = f"""Analyze this document for {analysis_type}.
Stream your response as you find issues.

Document:
{text[:10000]}

Provide findings one at a time in JSON format:
{{"type": "violation_found", "data": {{...}}}}"""

        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield json.dumps({"error": str(e)})
    
    async def maintain_infinite_context(
        self,
        conversation_id: str,
        messages: List[Dict[str, str]],
        documents: Optional[List[Dict[str, Any]]] = None
    ) -> ConversationMemory:
        """
        Maintain infinite context by summarizing and compressing conversation history.
        
        Args:
            conversation_id: Unique conversation ID
            messages: Full message history
            documents: Associated documents with summaries
        
        Returns:
            ConversationMemory with compressed context
        """
        if not self.is_available:
            return self._create_basic_memory(conversation_id, messages)
        
        # Check if we need to summarize (every 20 messages)
        if len(messages) < 20:
            return self.conversation_memories.get(conversation_id, 
                ConversationMemory(conversation_id=conversation_id, created_at=datetime.utcnow().isoformat()))
        
        messages_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in messages[-50:]  # Last 50 messages
        ])
        
        docs_text = ""
        if documents:
            docs_text = "\n\nDocuments analyzed:\n" + "\n".join([
                f"- {doc['filename']}: {doc.get('summary', 'No summary')}"
                for doc in documents
            ])
        
        prompt = f"""{self.INFINITE_CONTEXT_PROMPT}

Conversation ID: {conversation_id}
Total Messages: {len(messages)}

Recent Messages:
{messages_text}{docs_text}

Extract and return in JSON:
{{
    "key_topics": ["topic1", "topic2", ...],
    "entities_mentioned": {{
        "officers": ["Officer names"],
        "dates": ["Important dates"],
        "locations": ["Locations mentioned"],
        "violations": ["Violation types"]
    }},
    "decisions_made": ["Decision 1", ...],
    "pending_questions": ["Question 1", ...],
    "last_summary": "Concise summary of conversation state (max 500 words)"
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Create context hash for change detection
            context_str = json.dumps(result, sort_keys=True)
            context_hash = hashlib.sha256(context_str.encode()).hexdigest()[:16]
            
            memory = ConversationMemory(
                conversation_id=conversation_id,
                total_messages=len(messages),
                key_topics=result.get("key_topics", []),
                entities_mentioned=result.get("entities_mentioned", {}),
                decisions_made=result.get("decisions_made", []),
                pending_questions=result.get("pending_questions", []),
                context_hash=context_hash,
                last_summary=result.get("last_summary", ""),
                created_at=self.conversation_memories.get(conversation_id, ConversationMemory(
                    conversation_id=conversation_id, 
                    created_at=datetime.utcnow().isoformat()
                )).created_at,
                updated_at=datetime.utcnow().isoformat()
            )
            
            self.conversation_memories[conversation_id] = memory
            return memory
            
        except Exception as e:
            print(f"Context maintenance error: {e}")
            return self._create_basic_memory(conversation_id, messages)
    
    async def cross_reference_documents(
        self,
        documents: List[Dict[str, Any]],
        find_contradictions: bool = True
    ) -> Dict[str, Any]:
        """
        Cross-reference multiple documents to find corroborations and contradictions.
        
        Args:
            documents: List of analyzed documents
            find_contradictions: Whether to actively look for contradictions
        
        Returns:
            Cross-reference analysis with connections and discrepancies
        """
        if not self.is_available or len(documents) < 2:
            return {"error": "Need at least 2 documents and AI availability"}
        
        docs_summaries = "\n\n".join([
            f"DOCUMENT {i+1}: {doc['filename']}\n{doc.get('summary', 'No summary')}"
            for i, doc in enumerate(documents)
        ])
        
        prompt = f"""Cross-reference these documents to find:
1. Corroborating evidence (facts mentioned in multiple documents)
2. Contradictions or discrepancies
3. Timeline of events
4. Gaps in the evidence
5. Key witnesses or parties mentioned across documents

Documents:
{docs_summaries}

Return JSON:
{{
    "corroborations": [
        {{"fact": "...", "documents": [1, 2], "significance": "..."}}
    ],
    "contradictions": [
        {{"issue": "...", "doc1_says": "...", "doc2_says": "...", "documents": [1, 3]}}
    ],
    "timeline": [
        {{"date": "2025-11-29", "event": "...", "source_docs": [1, 2]}}
    ],
    "evidence_gaps": ["gap1", "gap2"],
    "key_witnesses": [
        {{"name": "...", "role": "...", "mentioned_in": [1, 2, 3]}}
    ]
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            print(f"Cross-reference error: {e}")
            return {"error": str(e)}
    
    # Fallback methods
    def _fallback_research(self, query: str) -> LegalResearchResult:
        """Fallback when AI unavailable"""
        return LegalResearchResult(
            query=query,
            research_summary="AI service unavailable. Manual research required.",
            confidence_score=0.0
        )
    
    def _fallback_complaint(self, case_data: Dict[str, Any]) -> ComplaintDraft:
        """Fallback when AI unavailable"""
        return ComplaintDraft(
            plaintiff_name=case_data.get('plaintiff', ''),
            defendant_names=case_data.get('defendants', []),
            generated_at=datetime.utcnow().isoformat(),
            confidence_score=0.0
        )
    
    def _create_basic_memory(self, conversation_id: str, messages: List[Dict[str, str]]) -> ConversationMemory:
        """Create basic memory without AI"""
        return ConversationMemory(
            conversation_id=conversation_id,
            total_messages=len(messages),
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat()
        )


# Global instance
enhanced_ai_service = EnhancedAIService()
