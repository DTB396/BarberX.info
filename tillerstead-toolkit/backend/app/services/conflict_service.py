"""
BarberX Legal Case Management Pro Suite
Conflict of Interest Checking Service

Premium service for identifying potential conflicts of interest
across clients, matters, parties, and related entities.
"""
import re
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set
import json

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class ConflictType(Enum):
    """Types of conflicts of interest"""
    DIRECT_ADVERSE = "direct_adverse"  # Current client vs current client
    FORMER_CLIENT = "former_client"  # Matter related to former client
    RELATED_PARTY = "related_party"  # Party related to existing client
    CORPORATE_AFFILIATE = "corporate_affiliate"  # Corporate family conflict
    BUSINESS_TRANSACTION = "business_transaction"  # Attorney-client transaction
    PERSONAL_INTEREST = "personal_interest"  # Attorney's personal interest
    WITNESS_CONFLICT = "witness_conflict"  # Attorney as witness
    POSITIONAL_CONFLICT = "positional_conflict"  # Adverse positions in different matters
    IMPUTED_CONFLICT = "imputed_conflict"  # Conflict imputed from other attorney


class ConflictSeverity(Enum):
    """Severity levels for conflicts"""
    CRITICAL = "critical"  # Absolute bar, cannot represent
    HIGH = "high"  # Likely requires declination or withdrawal
    MODERATE = "moderate"  # May be waivable with consent
    LOW = "low"  # Potential issue, investigate further
    CLEARED = "cleared"  # No conflict identified


class EntityType(Enum):
    """Types of entities in conflict system"""
    CLIENT = "client"
    FORMER_CLIENT = "former_client"
    ADVERSE_PARTY = "adverse_party"
    WITNESS = "witness"
    EXPERT = "expert"
    JUDGE = "judge"
    ATTORNEY = "attorney"
    CORPORATE_ENTITY = "corporate_entity"
    INDIVIDUAL = "individual"
    GOVERNMENT_AGENCY = "government_agency"


@dataclass
class Entity:
    """Entity in conflict database"""
    entity_id: str
    name: str
    entity_type: EntityType
    aliases: List[str] = field(default_factory=list)
    related_entities: List[str] = field(default_factory=list)  # Entity IDs
    corporate_family: List[str] = field(default_factory=list)  # Parent/subsidiary names
    addresses: List[str] = field(default_factory=list)
    phone_numbers: List[str] = field(default_factory=list)
    email_addresses: List[str] = field(default_factory=list)
    identifiers: Dict[str, str] = field(default_factory=dict)  # Tax ID, EIN, SSN, etc.
    notes: str = ""
    created_date: str = ""
    updated_date: str = ""


@dataclass
class Matter:
    """Legal matter in conflict database"""
    matter_id: str
    matter_number: str
    matter_name: str
    client_id: str
    matter_type: str
    status: str  # active, closed, prospective
    adverse_parties: List[str] = field(default_factory=list)  # Entity IDs
    related_parties: List[str] = field(default_factory=list)  # Entity IDs
    attorneys: List[str] = field(default_factory=list)  # Attorney names/IDs
    open_date: str = ""
    close_date: str = ""
    practice_area: str = ""
    description: str = ""
    jurisdiction: str = ""


@dataclass
class ConflictHit:
    """Individual conflict hit"""
    hit_id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    source_entity: str  # Entity being checked
    conflicting_entity: str  # Entity causing conflict
    conflicting_matter: Optional[str]  # Matter ID if applicable
    relationship: str  # Description of relationship
    rule_triggered: str  # ABA Model Rule or state rule
    explanation: str
    waivable: bool
    waiver_requirements: List[str] = field(default_factory=list)
    detected_date: str = ""


@dataclass
class ConflictCheck:
    """Complete conflict check"""
    check_id: str
    requested_by: str
    check_date: str
    new_matter_name: str
    prospective_client: Entity
    adverse_parties: List[Entity]
    related_parties: List[Entity]
    hits: List[ConflictHit] = field(default_factory=list)
    overall_status: ConflictSeverity = ConflictSeverity.CLEARED
    recommendation: str = ""
    cleared_by: str = ""
    cleared_date: str = ""
    waiver_obtained: bool = False
    waiver_date: str = ""
    notes: str = ""


@dataclass
class Waiver:
    """Conflict waiver record"""
    waiver_id: str
    conflict_check_id: str
    hit_id: str
    waiving_client_id: str
    waiver_date: str
    waiver_type: str  # informed_consent, advance_waiver, etc.
    scope: str
    obtained_by: str
    waiver_letter_path: str = ""
    expiration_date: str = ""
    conditions: List[str] = field(default_factory=list)


class ConflictCheckService:
    """
    Comprehensive conflict of interest checking service.
    
    Features:
    - Entity matching with fuzzy logic
    - Corporate family tracking
    - AI-powered relationship detection
    - Rule-based conflict analysis
    - Waiver management
    """
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.matters: Dict[str, Matter] = {}
        self.conflict_checks: Dict[str, ConflictCheck] = {}
        self.waivers: Dict[str, Waiver] = {}
        self.client = AsyncOpenAI() if OPENAI_AVAILABLE else None
        
        # Name matching thresholds
        self.exact_match_threshold = 1.0
        self.fuzzy_match_threshold = 0.85
        
    # ========================================
    # ENTITY MANAGEMENT
    # ========================================
    
    def add_entity(
        self,
        name: str,
        entity_type: EntityType,
        aliases: List[str] = None,
        corporate_family: List[str] = None,
        identifiers: Dict[str, str] = None,
        **kwargs
    ) -> Entity:
        """Add entity to conflict database"""
        entity_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        entity = Entity(
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            aliases=aliases or [],
            corporate_family=corporate_family or [],
            identifiers=identifiers or {},
            created_date=now,
            updated_date=now,
            **kwargs
        )
        
        self.entities[entity_id] = entity
        return entity
    
    def add_matter(
        self,
        matter_number: str,
        matter_name: str,
        client_id: str,
        matter_type: str,
        status: str = "active",
        adverse_parties: List[str] = None,
        **kwargs
    ) -> Matter:
        """Add matter to conflict database"""
        matter_id = str(uuid.uuid4())
        
        matter = Matter(
            matter_id=matter_id,
            matter_number=matter_number,
            matter_name=matter_name,
            client_id=client_id,
            matter_type=matter_type,
            status=status,
            adverse_parties=adverse_parties or [],
            open_date=datetime.now().isoformat(),
            **kwargs
        )
        
        self.matters[matter_id] = matter
        return matter
    
    # ========================================
    # NAME MATCHING
    # ========================================
    
    def _normalize_name(self, name: str) -> str:
        """Normalize name for comparison"""
        # Convert to lowercase
        normalized = name.lower().strip()
        
        # Remove common suffixes
        suffixes = [
            "inc", "inc.", "incorporated", "corp", "corp.", "corporation",
            "llc", "l.l.c.", "llp", "l.l.p.", "ltd", "limited", "co", "co.",
            "company", "pllc", "p.l.l.c.", "pc", "p.c.", "pa", "p.a.",
            "esq", "esq.", "esquire", "jr", "jr.", "sr", "sr.", "ii", "iii", "iv"
        ]
        
        for suffix in suffixes:
            normalized = re.sub(rf'\b{re.escape(suffix)}\.?\b', '', normalized)
        
        # Remove punctuation and extra spaces
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _calculate_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two names using multiple methods"""
        norm1 = self._normalize_name(name1)
        norm2 = self._normalize_name(name2)
        
        # Exact match after normalization
        if norm1 == norm2:
            return 1.0
        
        # Check if one is contained in the other
        if norm1 in norm2 or norm2 in norm1:
            return 0.9
        
        # Levenshtein-based similarity
        distance = self._levenshtein_distance(norm1, norm2)
        max_len = max(len(norm1), len(norm2))
        if max_len == 0:
            return 0.0
        
        return 1 - (distance / max_len)
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def find_matching_entities(
        self,
        search_name: str,
        include_aliases: bool = True,
        include_corporate_family: bool = True
    ) -> List[tuple]:
        """Find entities matching a name"""
        matches = []
        
        for entity in self.entities.values():
            # Check main name
            similarity = self._calculate_similarity(search_name, entity.name)
            if similarity >= self.fuzzy_match_threshold:
                matches.append((entity, similarity, "primary_name"))
            
            # Check aliases
            if include_aliases:
                for alias in entity.aliases:
                    alias_similarity = self._calculate_similarity(search_name, alias)
                    if alias_similarity >= self.fuzzy_match_threshold:
                        matches.append((entity, alias_similarity, "alias"))
            
            # Check corporate family
            if include_corporate_family:
                for corp_name in entity.corporate_family:
                    corp_similarity = self._calculate_similarity(search_name, corp_name)
                    if corp_similarity >= self.fuzzy_match_threshold:
                        matches.append((entity, corp_similarity, "corporate_family"))
        
        # Sort by similarity descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    # ========================================
    # CONFLICT CHECKING
    # ========================================
    
    async def run_conflict_check(
        self,
        new_matter_name: str,
        prospective_client_name: str,
        adverse_party_names: List[str],
        related_party_names: List[str] = None,
        matter_type: str = "",
        requested_by: str = "",
        use_ai_analysis: bool = True
    ) -> ConflictCheck:
        """
        Run comprehensive conflict check.
        
        Checks for:
        - Direct adverse conflicts with current clients
        - Former client conflicts
        - Corporate family conflicts
        - Related party conflicts
        - Positional conflicts
        """
        check_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        # Create or find prospective client entity
        prospective_matches = self.find_matching_entities(prospective_client_name)
        if prospective_matches:
            prospective_entity = prospective_matches[0][0]
        else:
            prospective_entity = Entity(
                entity_id=str(uuid.uuid4()),
                name=prospective_client_name,
                entity_type=EntityType.CLIENT,
                created_date=now,
                updated_date=now
            )
        
        # Process adverse parties
        adverse_entities = []
        for name in adverse_party_names:
            matches = self.find_matching_entities(name)
            if matches:
                adverse_entities.append(matches[0][0])
            else:
                adverse_entities.append(Entity(
                    entity_id=str(uuid.uuid4()),
                    name=name,
                    entity_type=EntityType.ADVERSE_PARTY,
                    created_date=now,
                    updated_date=now
                ))
        
        # Process related parties
        related_entities = []
        for name in (related_party_names or []):
            matches = self.find_matching_entities(name)
            if matches:
                related_entities.append(matches[0][0])
            else:
                related_entities.append(Entity(
                    entity_id=str(uuid.uuid4()),
                    name=name,
                    entity_type=EntityType.INDIVIDUAL,
                    created_date=now,
                    updated_date=now
                ))
        
        # Create conflict check
        conflict_check = ConflictCheck(
            check_id=check_id,
            requested_by=requested_by,
            check_date=now,
            new_matter_name=new_matter_name,
            prospective_client=prospective_entity,
            adverse_parties=adverse_entities,
            related_parties=related_entities,
            overall_status=ConflictSeverity.CLEARED
        )
        
        # Run rule-based checks
        hits = self._check_direct_adverse_conflicts(
            prospective_entity, adverse_entities, related_entities
        )
        conflict_check.hits.extend(hits)
        
        # Run corporate family checks
        corp_hits = self._check_corporate_family_conflicts(
            prospective_entity, adverse_entities
        )
        conflict_check.hits.extend(corp_hits)
        
        # Run former client checks
        former_hits = self._check_former_client_conflicts(
            prospective_entity, adverse_entities, new_matter_name, matter_type
        )
        conflict_check.hits.extend(former_hits)
        
        # Run AI-powered analysis for complex relationships
        if use_ai_analysis and OPENAI_AVAILABLE:
            ai_hits = await self._ai_conflict_analysis(
                conflict_check, matter_type
            )
            conflict_check.hits.extend(ai_hits)
        
        # Determine overall status
        if any(h.severity == ConflictSeverity.CRITICAL for h in conflict_check.hits):
            conflict_check.overall_status = ConflictSeverity.CRITICAL
            conflict_check.recommendation = "DECLINE REPRESENTATION - Absolute bar to representation"
        elif any(h.severity == ConflictSeverity.HIGH for h in conflict_check.hits):
            conflict_check.overall_status = ConflictSeverity.HIGH
            conflict_check.recommendation = "LIKELY DECLINE - Consider withdrawal/declination unless waiver possible"
        elif any(h.severity == ConflictSeverity.MODERATE for h in conflict_check.hits):
            conflict_check.overall_status = ConflictSeverity.MODERATE
            conflict_check.recommendation = "WAIVER REQUIRED - May proceed with informed consent from affected clients"
        elif any(h.severity == ConflictSeverity.LOW for h in conflict_check.hits):
            conflict_check.overall_status = ConflictSeverity.LOW
            conflict_check.recommendation = "REVIEW RECOMMENDED - Investigate potential issues before proceeding"
        else:
            conflict_check.recommendation = "CLEARED - No conflicts identified"
        
        self.conflict_checks[check_id] = conflict_check
        return conflict_check
    
    def _check_direct_adverse_conflicts(
        self,
        prospective_client: Entity,
        adverse_parties: List[Entity],
        related_parties: List[Entity]
    ) -> List[ConflictHit]:
        """Check for direct adverse conflicts with current clients"""
        hits = []
        
        for adverse in adverse_parties:
            # Check if adverse party is a current client
            for matter in self.matters.values():
                if matter.status != "active":
                    continue
                
                client = self.entities.get(matter.client_id)
                if not client:
                    continue
                
                # Check name match
                similarity = self._calculate_similarity(adverse.name, client.name)
                if similarity >= self.fuzzy_match_threshold:
                    hits.append(ConflictHit(
                        hit_id=str(uuid.uuid4()),
                        conflict_type=ConflictType.DIRECT_ADVERSE,
                        severity=ConflictSeverity.CRITICAL,
                        source_entity=adverse.name,
                        conflicting_entity=client.name,
                        conflicting_matter=matter.matter_id,
                        relationship=f"Adverse party matches current client in matter {matter.matter_number}",
                        rule_triggered="ABA Model Rule 1.7(a)(1) - Direct adversity to current client",
                        explanation=f"Cannot represent {prospective_client.name} against {client.name} who is a current client",
                        waivable=True,  # Sometimes waivable with informed consent
                        waiver_requirements=[
                            "Written informed consent from both clients",
                            "Reasonable belief representation not adversely affected",
                            "Matters must be unrelated",
                            "No assertion of privileged information"
                        ],
                        detected_date=datetime.now().isoformat()
                    ))
        
        return hits
    
    def _check_corporate_family_conflicts(
        self,
        prospective_client: Entity,
        adverse_parties: List[Entity]
    ) -> List[ConflictHit]:
        """Check for corporate family/affiliate conflicts"""
        hits = []
        
        for adverse in adverse_parties:
            # Check against all entities with corporate family relationships
            for entity in self.entities.values():
                if entity.entity_type not in [EntityType.CLIENT, EntityType.FORMER_CLIENT]:
                    continue
                
                # Check if adverse is in entity's corporate family
                for corp_name in entity.corporate_family:
                    similarity = self._calculate_similarity(adverse.name, corp_name)
                    if similarity >= self.fuzzy_match_threshold:
                        # Check if entity has active matters
                        has_active = any(
                            m.client_id == entity.entity_id and m.status == "active"
                            for m in self.matters.values()
                        )
                        
                        hits.append(ConflictHit(
                            hit_id=str(uuid.uuid4()),
                            conflict_type=ConflictType.CORPORATE_AFFILIATE,
                            severity=ConflictSeverity.HIGH if has_active else ConflictSeverity.MODERATE,
                            source_entity=adverse.name,
                            conflicting_entity=entity.name,
                            conflicting_matter=None,
                            relationship=f"Adverse party is corporate affiliate of {'current' if has_active else 'former'} client",
                            rule_triggered="ABA Model Rule 1.7 - Corporate family conflict",
                            explanation=f"{adverse.name} appears to be related to {entity.name} through corporate family: {corp_name}",
                            waivable=True,
                            waiver_requirements=[
                                "Confirm corporate relationship",
                                "Evaluate operational relationship between entities",
                                "Consider confidential information sharing",
                                "Obtain informed consent if proceeding"
                            ],
                            detected_date=datetime.now().isoformat()
                        ))
        
        return hits
    
    def _check_former_client_conflicts(
        self,
        prospective_client: Entity,
        adverse_parties: List[Entity],
        new_matter_name: str,
        matter_type: str
    ) -> List[ConflictHit]:
        """Check for former client conflicts"""
        hits = []
        
        for adverse in adverse_parties:
            # Find closed matters where adverse was the client
            for matter in self.matters.values():
                if matter.status != "closed":
                    continue
                
                client = self.entities.get(matter.client_id)
                if not client:
                    continue
                
                similarity = self._calculate_similarity(adverse.name, client.name)
                if similarity >= self.fuzzy_match_threshold:
                    # Check if matters are substantially related
                    related = self._are_matters_related(
                        matter.matter_type, matter_type,
                        matter.matter_name, new_matter_name
                    )
                    
                    hits.append(ConflictHit(
                        hit_id=str(uuid.uuid4()),
                        conflict_type=ConflictType.FORMER_CLIENT,
                        severity=ConflictSeverity.HIGH if related else ConflictSeverity.MODERATE,
                        source_entity=adverse.name,
                        conflicting_entity=client.name,
                        conflicting_matter=matter.matter_id,
                        relationship=f"Adverse party was former client in matter {matter.matter_number}",
                        rule_triggered="ABA Model Rule 1.9 - Duties to Former Clients",
                        explanation=f"Previously represented {client.name} in {matter.matter_type} matter. New matter {'appears substantially related' if related else 'may be related'}.",
                        waivable=True,
                        waiver_requirements=[
                            "Written informed consent from former client",
                            "Determine if matters are 'substantially related'",
                            "Evaluate confidential information obtained",
                            "Assess if information would be advantageous"
                        ],
                        detected_date=datetime.now().isoformat()
                    ))
        
        return hits
    
    def _are_matters_related(
        self,
        old_type: str,
        new_type: str,
        old_name: str,
        new_name: str
    ) -> bool:
        """Determine if two matters are substantially related"""
        # Same matter type is a strong indicator
        if old_type and new_type and old_type.lower() == new_type.lower():
            return True
        
        # Check for common keywords in matter names
        old_words = set(self._normalize_name(old_name).split())
        new_words = set(self._normalize_name(new_name).split())
        
        # Remove common filler words
        filler = {'the', 'a', 'an', 'of', 'in', 'on', 'for', 'to', 'and', 'or', 'matter', 'case', 'vs', 'v'}
        old_words -= filler
        new_words -= filler
        
        # If significant overlap, matters may be related
        if old_words and new_words:
            overlap = len(old_words & new_words) / min(len(old_words), len(new_words))
            if overlap > 0.3:
                return True
        
        return False
    
    async def _ai_conflict_analysis(
        self,
        conflict_check: ConflictCheck,
        matter_type: str
    ) -> List[ConflictHit]:
        """Use AI to analyze complex conflict relationships"""
        if not self.client:
            return []
        
        hits = []
        
        # Prepare context for AI analysis
        context = {
            "prospective_client": conflict_check.prospective_client.name,
            "adverse_parties": [e.name for e in conflict_check.adverse_parties],
            "related_parties": [e.name for e in conflict_check.related_parties],
            "new_matter": conflict_check.new_matter_name,
            "matter_type": matter_type,
            "existing_clients": [
                {
                    "name": e.name,
                    "corporate_family": e.corporate_family,
                    "aliases": e.aliases
                }
                for e in self.entities.values()
                if e.entity_type == EntityType.CLIENT
            ]
        }
        
        prompt = f"""Analyze these parties for potential conflicts of interest in legal representation:

Context:
{json.dumps(context, indent=2)}

Consider:
1. Corporate relationships (parent/subsidiary, common ownership)
2. Personal relationships (family, business partners)
3. Positional conflicts (adverse legal positions)
4. Industry-specific conflicts
5. Government agency relationships

For each potential conflict found, provide:
- Relationship type
- Parties involved
- Severity (critical/high/moderate/low)
- Applicable ethics rule
- Whether waivable

Respond in JSON format with an array of conflicts found."""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert legal ethics advisor analyzing conflicts of interest."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            for conflict in result.get("conflicts", []):
                severity_map = {
                    "critical": ConflictSeverity.CRITICAL,
                    "high": ConflictSeverity.HIGH,
                    "moderate": ConflictSeverity.MODERATE,
                    "low": ConflictSeverity.LOW
                }
                
                hits.append(ConflictHit(
                    hit_id=str(uuid.uuid4()),
                    conflict_type=ConflictType.RELATED_PARTY,  # AI-detected
                    severity=severity_map.get(conflict.get("severity", "").lower(), ConflictSeverity.LOW),
                    source_entity=conflict.get("party1", "Unknown"),
                    conflicting_entity=conflict.get("party2", "Unknown"),
                    conflicting_matter=None,
                    relationship=conflict.get("relationship_type", "AI-detected relationship"),
                    rule_triggered=conflict.get("ethics_rule", "ABA Model Rules 1.7-1.9"),
                    explanation=conflict.get("explanation", ""),
                    waivable=conflict.get("waivable", True),
                    waiver_requirements=conflict.get("waiver_requirements", []),
                    detected_date=datetime.now().isoformat()
                ))
                
        except Exception as e:
            # Log error but don't fail the check
            pass
        
        return hits
    
    # ========================================
    # WAIVER MANAGEMENT
    # ========================================
    
    def record_waiver(
        self,
        conflict_check_id: str,
        hit_id: str,
        waiving_client_id: str,
        waiver_type: str,
        scope: str,
        obtained_by: str,
        waiver_letter_path: str = "",
        conditions: List[str] = None
    ) -> Waiver:
        """Record a conflict waiver"""
        waiver_id = str(uuid.uuid4())
        
        waiver = Waiver(
            waiver_id=waiver_id,
            conflict_check_id=conflict_check_id,
            hit_id=hit_id,
            waiving_client_id=waiving_client_id,
            waiver_date=datetime.now().isoformat(),
            waiver_type=waiver_type,
            scope=scope,
            obtained_by=obtained_by,
            waiver_letter_path=waiver_letter_path,
            conditions=conditions or []
        )
        
        self.waivers[waiver_id] = waiver
        
        # Update conflict check
        if conflict_check_id in self.conflict_checks:
            self.conflict_checks[conflict_check_id].waiver_obtained = True
            self.conflict_checks[conflict_check_id].waiver_date = waiver.waiver_date
        
        return waiver
    
    def generate_waiver_letter(
        self,
        conflict_check_id: str,
        hit_id: str,
        client_name: str,
        attorney_name: str,
        firm_name: str
    ) -> str:
        """Generate conflict waiver letter template"""
        check = self.conflict_checks.get(conflict_check_id)
        if not check:
            return "Conflict check not found"
        
        hit = next((h for h in check.hits if h.hit_id == hit_id), None)
        if not hit:
            return "Conflict hit not found"
        
        return f"""
CONFLICT OF INTEREST WAIVER AND INFORMED CONSENT

Date: {datetime.now().strftime("%B %d, %Y")}

To: {client_name}
From: {attorney_name}, {firm_name}

Re: Waiver of Conflict of Interest
    Matter: {check.new_matter_name}

Dear {client_name}:

We are writing to request your informed consent to a potential conflict of interest 
that has been identified in connection with our representation of you.

NATURE OF THE CONFLICT:

{hit.explanation}

Rule Implicated: {hit.rule_triggered}

RISKS AND BENEFITS:

If you consent to waive this conflict, you should understand:

1. RISKS:
   - Information learned in one representation could potentially be used in the other
   - Our advice to you might be influenced by our obligations to another client
   - In certain circumstances, we might have to withdraw from representing one or both clients

2. BENEFITS:
   - You may continue to receive representation from attorneys familiar with your matters
   - Continuity of representation
   - Cost efficiency

YOUR OPTIONS:

1. You may consent to waive this conflict, allowing us to continue representation
2. You may decline to waive the conflict, requiring us to decline the new representation
3. You may seek advice from independent counsel before making this decision

REQUIREMENTS FOR VALID WAIVER:

{chr(10).join(f"- {req}" for req in hit.waiver_requirements)}

CONSENT:

If you wish to waive this conflict after careful consideration, please sign below:

I, {client_name}, have read and understand the above disclosure regarding the 
conflict of interest. I have had the opportunity to ask questions and consult with 
independent counsel if desired. I voluntarily consent to waive this conflict.

_____________________________________
Signature

_____________________________________
Date

Please return a signed copy to our office.

Sincerely,

{attorney_name}
{firm_name}
"""
    
    # ========================================
    # REPORTING
    # ========================================
    
    def generate_conflict_report(self, check_id: str) -> Dict[str, Any]:
        """Generate comprehensive conflict report"""
        check = self.conflict_checks.get(check_id)
        if not check:
            return {"error": "Conflict check not found"}
        
        return {
            "report_date": datetime.now().isoformat(),
            "check_id": check.check_id,
            "requested_by": check.requested_by,
            "check_date": check.check_date,
            
            "new_matter": {
                "name": check.new_matter_name,
                "prospective_client": check.prospective_client.name
            },
            
            "parties_checked": {
                "adverse_parties": [e.name for e in check.adverse_parties],
                "related_parties": [e.name for e in check.related_parties]
            },
            
            "results": {
                "overall_status": check.overall_status.value,
                "recommendation": check.recommendation,
                "total_hits": len(check.hits),
                "critical_hits": sum(1 for h in check.hits if h.severity == ConflictSeverity.CRITICAL),
                "high_hits": sum(1 for h in check.hits if h.severity == ConflictSeverity.HIGH),
                "moderate_hits": sum(1 for h in check.hits if h.severity == ConflictSeverity.MODERATE),
                "low_hits": sum(1 for h in check.hits if h.severity == ConflictSeverity.LOW)
            },
            
            "conflicts_found": [
                {
                    "hit_id": h.hit_id,
                    "type": h.conflict_type.value,
                    "severity": h.severity.value,
                    "source_entity": h.source_entity,
                    "conflicting_entity": h.conflicting_entity,
                    "relationship": h.relationship,
                    "rule": h.rule_triggered,
                    "explanation": h.explanation,
                    "waivable": h.waivable,
                    "waiver_requirements": h.waiver_requirements
                }
                for h in check.hits
            ],
            
            "waiver_status": {
                "waiver_obtained": check.waiver_obtained,
                "waiver_date": check.waiver_date,
                "cleared_by": check.cleared_by,
                "cleared_date": check.cleared_date
            }
        }


# Singleton instance
conflict_service = ConflictCheckService()
