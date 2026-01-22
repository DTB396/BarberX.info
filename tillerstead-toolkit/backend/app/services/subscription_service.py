"""
BarberX Legal Case Management Pro Suite
Subscription & Licensing Service

Manages user subscriptions, feature access, and usage limits.
Supports free tier, professional, and enterprise plans.
"""
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import hashlib
import secrets


class SubscriptionTier(Enum):
    """Subscription tiers"""
    FREE = "free"
    SOLO = "solo"  # Solo practitioner
    PROFESSIONAL = "professional"  # Small firm
    ENTERPRISE = "enterprise"  # Large firm / unlimited


class BillingCycle(Enum):
    """Billing cycle options"""
    MONTHLY = "monthly"
    ANNUAL = "annual"  # 20% discount
    LIFETIME = "lifetime"  # One-time purchase


class FeatureCategory(Enum):
    """Feature categories for access control"""
    CASE_MANAGEMENT = "case_management"
    DOCUMENT_UPLOAD = "document_upload"
    OCR_PROCESSING = "ocr_processing"
    AI_ANALYSIS = "ai_analysis"
    BWC_VIDEO = "bwc_video"
    TRANSCRIPTION = "transcription"
    E_DISCOVERY = "e_discovery"
    DEPOSITIONS = "depositions"
    CASE_STRATEGY = "case_strategy"
    DEADLINE_CALC = "deadline_calc"
    BRADY_TRACKING = "brady_tracking"
    CONFLICT_CHECK = "conflict_check"
    BILLING = "billing"
    LEGAL_RESEARCH = "legal_research"
    PLEADING_GEN = "pleading_gen"
    EXPORTS = "exports"
    API_ACCESS = "api_access"


@dataclass
class PricingPlan:
    """Pricing plan definition"""
    tier: SubscriptionTier
    name: str
    description: str
    
    # Pricing
    monthly_price: float
    annual_price: float  # Total annual (typically 20% off)
    lifetime_price: Optional[float] = None
    
    # Limits
    max_cases: int = 0  # 0 = unlimited
    max_documents_per_month: int = 0
    max_ai_queries_per_month: int = 0
    max_users: int = 1
    storage_gb: float = 0
    
    # Features included
    features: List[FeatureCategory] = field(default_factory=list)
    
    # Support level
    support_level: str = "community"  # community, email, priority, dedicated


@dataclass
class Subscription:
    """User subscription record"""
    subscription_id: str
    user_id: str
    organization_id: Optional[str] = None
    
    # Plan details
    tier: SubscriptionTier = SubscriptionTier.FREE
    billing_cycle: BillingCycle = BillingCycle.MONTHLY
    
    # Dates
    created_date: str = ""
    start_date: str = ""
    end_date: str = ""
    trial_end_date: str = ""
    
    # Status
    status: str = "active"  # active, trial, past_due, cancelled, expired
    auto_renew: bool = True
    
    # Payment
    payment_method_id: str = ""
    last_payment_date: str = ""
    next_payment_date: str = ""
    amount_paid: float = 0
    
    # Usage this period
    cases_used: int = 0
    documents_used: int = 0
    ai_queries_used: int = 0
    storage_used_gb: float = 0


@dataclass 
class UsageRecord:
    """Track usage for billing and limits"""
    record_id: str
    subscription_id: str
    user_id: str
    
    # Period
    period_start: str
    period_end: str
    
    # Usage counts
    cases_created: int = 0
    documents_uploaded: int = 0
    documents_ocr: int = 0
    ai_queries: int = 0
    transcription_minutes: float = 0
    video_hours_processed: float = 0
    
    # Feature usage
    feature_usage: Dict[str, int] = field(default_factory=dict)


@dataclass
class APIKey:
    """API key for programmatic access"""
    key_id: str
    subscription_id: str
    user_id: str
    
    # Key details (only prefix stored, hash for validation)
    key_prefix: str  # First 8 chars for display
    key_hash: str  # SHA-256 hash for validation
    
    name: str = ""
    created_date: str = ""
    last_used: str = ""
    expires_date: str = ""
    
    # Permissions
    scopes: List[str] = field(default_factory=list)
    rate_limit_per_minute: int = 60
    
    # Status
    is_active: bool = True


class SubscriptionService:
    """
    Manages subscriptions, licensing, and feature access.
    
    Pricing Model:
    - FREE: Basic features, limited usage
    - SOLO ($49/mo): Full features for solo practitioners
    - PROFESSIONAL ($199/mo): Team features, higher limits
    - ENTERPRISE (Custom): Unlimited, dedicated support
    """
    
    def __init__(self):
        self.subscriptions: Dict[str, Subscription] = {}
        self.usage_records: Dict[str, UsageRecord] = {}
        self.api_keys: Dict[str, APIKey] = {}
        self.plans: Dict[SubscriptionTier, PricingPlan] = {}
        
        # Initialize pricing plans
        self._initialize_plans()
    
    def _initialize_plans(self):
        """Set up pricing plans"""
        
        # FREE TIER - Get started at no cost
        self.plans[SubscriptionTier.FREE] = PricingPlan(
            tier=SubscriptionTier.FREE,
            name="Free",
            description="Get started with essential case management",
            monthly_price=0,
            annual_price=0,
            max_cases=3,
            max_documents_per_month=50,
            max_ai_queries_per_month=25,
            max_users=1,
            storage_gb=1.0,
            features=[
                FeatureCategory.CASE_MANAGEMENT,
                FeatureCategory.DOCUMENT_UPLOAD,
                FeatureCategory.DEADLINE_CALC,
                FeatureCategory.EXPORTS,
            ],
            support_level="community"
        )
        
        # SOLO - For solo practitioners
        self.plans[SubscriptionTier.SOLO] = PricingPlan(
            tier=SubscriptionTier.SOLO,
            name="Solo Practitioner",
            description="Full-featured for independent attorneys",
            monthly_price=49.00,
            annual_price=470.00,  # ~20% off ($39/mo effective)
            lifetime_price=1499.00,
            max_cases=25,
            max_documents_per_month=500,
            max_ai_queries_per_month=200,
            max_users=1,
            storage_gb=25.0,
            features=[
                FeatureCategory.CASE_MANAGEMENT,
                FeatureCategory.DOCUMENT_UPLOAD,
                FeatureCategory.OCR_PROCESSING,
                FeatureCategory.AI_ANALYSIS,
                FeatureCategory.TRANSCRIPTION,
                FeatureCategory.DEADLINE_CALC,
                FeatureCategory.CONFLICT_CHECK,
                FeatureCategory.PLEADING_GEN,
                FeatureCategory.EXPORTS,
            ],
            support_level="email"
        )
        
        # PROFESSIONAL - For small firms
        self.plans[SubscriptionTier.PROFESSIONAL] = PricingPlan(
            tier=SubscriptionTier.PROFESSIONAL,
            name="Professional",
            description="Team collaboration with advanced features",
            monthly_price=199.00,
            annual_price=1910.00,  # ~20% off ($159/mo effective)
            lifetime_price=5999.00,
            max_cases=100,
            max_documents_per_month=2500,
            max_ai_queries_per_month=1000,
            max_users=10,
            storage_gb=100.0,
            features=[
                FeatureCategory.CASE_MANAGEMENT,
                FeatureCategory.DOCUMENT_UPLOAD,
                FeatureCategory.OCR_PROCESSING,
                FeatureCategory.AI_ANALYSIS,
                FeatureCategory.BWC_VIDEO,
                FeatureCategory.TRANSCRIPTION,
                FeatureCategory.E_DISCOVERY,
                FeatureCategory.DEPOSITIONS,
                FeatureCategory.CASE_STRATEGY,
                FeatureCategory.DEADLINE_CALC,
                FeatureCategory.BRADY_TRACKING,
                FeatureCategory.CONFLICT_CHECK,
                FeatureCategory.BILLING,
                FeatureCategory.LEGAL_RESEARCH,
                FeatureCategory.PLEADING_GEN,
                FeatureCategory.EXPORTS,
                FeatureCategory.API_ACCESS,
            ],
            support_level="priority"
        )
        
        # ENTERPRISE - For large firms
        self.plans[SubscriptionTier.ENTERPRISE] = PricingPlan(
            tier=SubscriptionTier.ENTERPRISE,
            name="Enterprise",
            description="Unlimited access with dedicated support",
            monthly_price=499.00,  # Starting price, custom quotes
            annual_price=4790.00,
            lifetime_price=None,  # Contact for pricing
            max_cases=0,  # Unlimited
            max_documents_per_month=0,  # Unlimited
            max_ai_queries_per_month=0,  # Unlimited
            max_users=0,  # Unlimited
            storage_gb=0,  # Unlimited (fair use)
            features=list(FeatureCategory),  # All features
            support_level="dedicated"
        )
    
    # ========================================
    # SUBSCRIPTION MANAGEMENT
    # ========================================
    
    def create_subscription(
        self,
        user_id: str,
        tier: SubscriptionTier = SubscriptionTier.FREE,
        billing_cycle: BillingCycle = BillingCycle.MONTHLY,
        organization_id: str = None,
        trial_days: int = 0
    ) -> Subscription:
        """Create new subscription"""
        subscription_id = str(uuid.uuid4())
        now = datetime.now()
        
        # Calculate dates
        start_date = now
        if trial_days > 0:
            trial_end = now + timedelta(days=trial_days)
            end_date = trial_end
            status = "trial"
        else:
            trial_end = None
            if billing_cycle == BillingCycle.MONTHLY:
                end_date = now + timedelta(days=30)
            elif billing_cycle == BillingCycle.ANNUAL:
                end_date = now + timedelta(days=365)
            else:  # Lifetime
                end_date = now + timedelta(days=36500)  # 100 years
            status = "active"
        
        subscription = Subscription(
            subscription_id=subscription_id,
            user_id=user_id,
            organization_id=organization_id,
            tier=tier,
            billing_cycle=billing_cycle,
            created_date=now.isoformat(),
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            trial_end_date=trial_end.isoformat() if trial_end else "",
            status=status,
            next_payment_date=end_date.isoformat() if tier != SubscriptionTier.FREE else ""
        )
        
        self.subscriptions[subscription_id] = subscription
        
        # Create initial usage record
        self._create_usage_record(subscription_id, user_id)
        
        return subscription
    
    def start_trial(
        self,
        user_id: str,
        tier: SubscriptionTier = SubscriptionTier.PROFESSIONAL,
        trial_days: int = 14
    ) -> Subscription:
        """Start a free trial of a paid tier"""
        return self.create_subscription(
            user_id=user_id,
            tier=tier,
            trial_days=trial_days
        )
    
    def upgrade_subscription(
        self,
        subscription_id: str,
        new_tier: SubscriptionTier,
        billing_cycle: BillingCycle = None
    ) -> Subscription:
        """Upgrade subscription to higher tier"""
        subscription = self.subscriptions.get(subscription_id)
        if not subscription:
            raise ValueError("Subscription not found")
        
        subscription.tier = new_tier
        if billing_cycle:
            subscription.billing_cycle = billing_cycle
        
        # Recalculate end date
        now = datetime.now()
        if billing_cycle == BillingCycle.MONTHLY:
            subscription.end_date = (now + timedelta(days=30)).isoformat()
        elif billing_cycle == BillingCycle.ANNUAL:
            subscription.end_date = (now + timedelta(days=365)).isoformat()
        
        subscription.status = "active"
        
        return subscription
    
    def cancel_subscription(
        self,
        subscription_id: str,
        at_period_end: bool = True
    ) -> Subscription:
        """Cancel subscription"""
        subscription = self.subscriptions.get(subscription_id)
        if not subscription:
            raise ValueError("Subscription not found")
        
        subscription.auto_renew = False
        
        if not at_period_end:
            subscription.status = "cancelled"
            subscription.end_date = datetime.now().isoformat()
        else:
            # Will cancel at end of current billing period
            subscription.status = "active"  # Remains active until period end
        
        return subscription
    
    def _create_usage_record(
        self,
        subscription_id: str,
        user_id: str
    ) -> UsageRecord:
        """Create usage tracking record for current period"""
        record_id = str(uuid.uuid4())
        now = datetime.now()
        
        # Period is current month
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if now.month == 12:
            period_end = period_start.replace(year=now.year + 1, month=1)
        else:
            period_end = period_start.replace(month=now.month + 1)
        
        record = UsageRecord(
            record_id=record_id,
            subscription_id=subscription_id,
            user_id=user_id,
            period_start=period_start.isoformat(),
            period_end=period_end.isoformat()
        )
        
        self.usage_records[record_id] = record
        return record
    
    # ========================================
    # FEATURE ACCESS CONTROL
    # ========================================
    
    def check_feature_access(
        self,
        subscription_id: str,
        feature: FeatureCategory
    ) -> Dict[str, Any]:
        """Check if subscription has access to a feature"""
        subscription = self.subscriptions.get(subscription_id)
        if not subscription:
            return {
                "allowed": False,
                "reason": "Subscription not found"
            }
        
        # Check subscription status
        if subscription.status not in ["active", "trial"]:
            return {
                "allowed": False,
                "reason": f"Subscription is {subscription.status}"
            }
        
        # Check expiration
        if subscription.end_date:
            end = datetime.fromisoformat(subscription.end_date)
            if datetime.now() > end:
                return {
                    "allowed": False,
                    "reason": "Subscription has expired"
                }
        
        # Check feature access
        plan = self.plans.get(subscription.tier)
        if not plan:
            return {
                "allowed": False,
                "reason": "Invalid subscription tier"
            }
        
        if feature in plan.features:
            return {
                "allowed": True,
                "tier": subscription.tier.value,
                "plan": plan.name
            }
        else:
            return {
                "allowed": False,
                "reason": f"Feature '{feature.value}' requires upgrade",
                "current_tier": subscription.tier.value,
                "required_tier": self._get_minimum_tier_for_feature(feature).value
            }
    
    def _get_minimum_tier_for_feature(
        self,
        feature: FeatureCategory
    ) -> SubscriptionTier:
        """Get minimum tier that includes a feature"""
        for tier in [SubscriptionTier.FREE, SubscriptionTier.SOLO, 
                     SubscriptionTier.PROFESSIONAL, SubscriptionTier.ENTERPRISE]:
            plan = self.plans.get(tier)
            if plan and feature in plan.features:
                return tier
        return SubscriptionTier.ENTERPRISE
    
    def check_usage_limit(
        self,
        subscription_id: str,
        limit_type: str
    ) -> Dict[str, Any]:
        """Check if usage is within limits"""
        subscription = self.subscriptions.get(subscription_id)
        if not subscription:
            return {"within_limit": False, "reason": "Subscription not found"}
        
        plan = self.plans.get(subscription.tier)
        if not plan:
            return {"within_limit": False, "reason": "Invalid tier"}
        
        # Get current usage
        current_usage = self._get_current_usage(subscription_id)
        
        limits = {
            "cases": (subscription.cases_used, plan.max_cases),
            "documents": (current_usage.documents_uploaded, plan.max_documents_per_month),
            "ai_queries": (current_usage.ai_queries, plan.max_ai_queries_per_month),
            "storage": (subscription.storage_used_gb, plan.storage_gb),
        }
        
        if limit_type not in limits:
            return {"within_limit": True}  # Unknown limit type, allow
        
        used, max_allowed = limits[limit_type]
        
        # 0 means unlimited
        if max_allowed == 0:
            return {
                "within_limit": True,
                "used": used,
                "limit": "unlimited",
                "percentage": 0
            }
        
        within = used < max_allowed
        return {
            "within_limit": within,
            "used": used,
            "limit": max_allowed,
            "remaining": max(0, max_allowed - used),
            "percentage": round((used / max_allowed) * 100, 1) if max_allowed > 0 else 0,
            "reason": f"Limit reached for {limit_type}" if not within else None
        }
    
    def _get_current_usage(self, subscription_id: str) -> UsageRecord:
        """Get current period usage record"""
        now = datetime.now()
        
        for record in self.usage_records.values():
            if record.subscription_id == subscription_id:
                period_end = datetime.fromisoformat(record.period_end)
                if now < period_end:
                    return record
        
        # Create new record if none exists
        sub = self.subscriptions.get(subscription_id)
        if sub:
            return self._create_usage_record(subscription_id, sub.user_id)
        
        return UsageRecord(
            record_id="",
            subscription_id=subscription_id,
            user_id="",
            period_start="",
            period_end=""
        )
    
    def record_usage(
        self,
        subscription_id: str,
        usage_type: str,
        amount: int = 1
    ):
        """Record usage for billing/limits"""
        record = self._get_current_usage(subscription_id)
        subscription = self.subscriptions.get(subscription_id)
        
        if usage_type == "document":
            record.documents_uploaded += amount
        elif usage_type == "ocr":
            record.documents_ocr += amount
        elif usage_type == "ai_query":
            record.ai_queries += amount
        elif usage_type == "transcription":
            record.transcription_minutes += amount
        elif usage_type == "video":
            record.video_hours_processed += amount
        elif usage_type == "case":
            if subscription:
                subscription.cases_used += amount
        
        # Track by feature
        record.feature_usage[usage_type] = record.feature_usage.get(usage_type, 0) + amount
    
    # ========================================
    # API KEY MANAGEMENT
    # ========================================
    
    def create_api_key(
        self,
        subscription_id: str,
        user_id: str,
        name: str = "",
        scopes: List[str] = None,
        expires_days: int = 365
    ) -> tuple:
        """
        Create new API key.
        Returns (APIKey object, raw_key) - raw key only shown once!
        """
        subscription = self.subscriptions.get(subscription_id)
        if not subscription:
            raise ValueError("Subscription not found")
        
        # Check if API access is allowed
        access = self.check_feature_access(subscription_id, FeatureCategory.API_ACCESS)
        if not access["allowed"]:
            raise ValueError(f"API access not available: {access['reason']}")
        
        # Generate secure key
        raw_key = f"bx_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        key_prefix = raw_key[:11]  # "bx_" + 8 chars
        
        key_id = str(uuid.uuid4())
        now = datetime.now()
        
        api_key = APIKey(
            key_id=key_id,
            subscription_id=subscription_id,
            user_id=user_id,
            key_prefix=key_prefix,
            key_hash=key_hash,
            name=name or f"API Key {key_id[:8]}",
            created_date=now.isoformat(),
            expires_date=(now + timedelta(days=expires_days)).isoformat(),
            scopes=scopes or ["read", "write"],
            rate_limit_per_minute=self._get_rate_limit(subscription.tier)
        )
        
        self.api_keys[key_id] = api_key
        
        return api_key, raw_key
    
    def validate_api_key(self, raw_key: str) -> Optional[APIKey]:
        """Validate API key and return key info if valid"""
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        for api_key in self.api_keys.values():
            if api_key.key_hash == key_hash:
                # Check if active and not expired
                if not api_key.is_active:
                    return None
                
                if api_key.expires_date:
                    expires = datetime.fromisoformat(api_key.expires_date)
                    if datetime.now() > expires:
                        return None
                
                # Update last used
                api_key.last_used = datetime.now().isoformat()
                return api_key
        
        return None
    
    def revoke_api_key(self, key_id: str):
        """Revoke an API key"""
        if key_id in self.api_keys:
            self.api_keys[key_id].is_active = False
    
    def _get_rate_limit(self, tier: SubscriptionTier) -> int:
        """Get API rate limit per minute based on tier"""
        limits = {
            SubscriptionTier.FREE: 10,
            SubscriptionTier.SOLO: 60,
            SubscriptionTier.PROFESSIONAL: 300,
            SubscriptionTier.ENTERPRISE: 1000,
        }
        return limits.get(tier, 10)
    
    # ========================================
    # PRICING & CHECKOUT
    # ========================================
    
    def get_pricing(self) -> Dict[str, Any]:
        """Get all pricing plans for display"""
        return {
            "plans": [
                {
                    "tier": plan.tier.value,
                    "name": plan.name,
                    "description": plan.description,
                    "pricing": {
                        "monthly": plan.monthly_price,
                        "annual": plan.annual_price,
                        "annual_monthly_equivalent": round(plan.annual_price / 12, 2),
                        "annual_savings": round((plan.monthly_price * 12) - plan.annual_price, 2),
                        "lifetime": plan.lifetime_price
                    },
                    "limits": {
                        "cases": plan.max_cases if plan.max_cases > 0 else "Unlimited",
                        "documents_per_month": plan.max_documents_per_month if plan.max_documents_per_month > 0 else "Unlimited",
                        "ai_queries_per_month": plan.max_ai_queries_per_month if plan.max_ai_queries_per_month > 0 else "Unlimited",
                        "users": plan.max_users if plan.max_users > 0 else "Unlimited",
                        "storage_gb": plan.storage_gb if plan.storage_gb > 0 else "Unlimited"
                    },
                    "features": [f.value for f in plan.features],
                    "support": plan.support_level,
                    "popular": plan.tier == SubscriptionTier.PROFESSIONAL
                }
                for plan in self.plans.values()
            ],
            "trial": {
                "available": True,
                "days": 14,
                "tier": "professional",
                "no_credit_card": True
            },
            "currency": "USD"
        }
    
    def get_subscription_status(
        self,
        subscription_id: str
    ) -> Dict[str, Any]:
        """Get detailed subscription status"""
        subscription = self.subscriptions.get(subscription_id)
        if not subscription:
            return {"error": "Subscription not found"}
        
        plan = self.plans.get(subscription.tier)
        usage = self._get_current_usage(subscription_id)
        
        return {
            "subscription_id": subscription_id,
            "tier": subscription.tier.value,
            "plan_name": plan.name if plan else "Unknown",
            "status": subscription.status,
            "billing_cycle": subscription.billing_cycle.value,
            
            "dates": {
                "created": subscription.created_date,
                "start": subscription.start_date,
                "end": subscription.end_date,
                "trial_end": subscription.trial_end_date,
                "next_payment": subscription.next_payment_date
            },
            
            "usage": {
                "cases": {
                    "used": subscription.cases_used,
                    "limit": plan.max_cases if plan else 0,
                    "percentage": self._calc_percentage(subscription.cases_used, plan.max_cases if plan else 0)
                },
                "documents": {
                    "used": usage.documents_uploaded,
                    "limit": plan.max_documents_per_month if plan else 0,
                    "percentage": self._calc_percentage(usage.documents_uploaded, plan.max_documents_per_month if plan else 0)
                },
                "ai_queries": {
                    "used": usage.ai_queries,
                    "limit": plan.max_ai_queries_per_month if plan else 0,
                    "percentage": self._calc_percentage(usage.ai_queries, plan.max_ai_queries_per_month if plan else 0)
                },
                "storage_gb": {
                    "used": subscription.storage_used_gb,
                    "limit": plan.storage_gb if plan else 0,
                    "percentage": self._calc_percentage(subscription.storage_used_gb, plan.storage_gb if plan else 0)
                }
            },
            
            "features": [f.value for f in (plan.features if plan else [])],
            
            "auto_renew": subscription.auto_renew
        }
    
    def _calc_percentage(self, used: float, limit: float) -> float:
        """Calculate usage percentage (0 limit = unlimited = 0%)"""
        if limit == 0:
            return 0
        return round((used / limit) * 100, 1)


# Singleton instance
subscription_service = SubscriptionService()
