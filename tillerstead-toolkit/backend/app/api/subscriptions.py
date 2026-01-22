"""
BarberX Legal Case Management Pro Suite
Subscription & Billing API

Endpoints for subscription management, pricing, and payments.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Depends, Header, Query
from pydantic import BaseModel, Field

from app.services.subscription_service import (
    subscription_service,
    SubscriptionTier,
    BillingCycle,
    FeatureCategory
)

router = APIRouter()


# ============================================================
# SCHEMAS
# ============================================================

class SubscriptionCreateRequest(BaseModel):
    user_id: str
    tier: str = "free"
    billing_cycle: str = "monthly"
    organization_id: Optional[str] = None


class TrialStartRequest(BaseModel):
    user_id: str
    tier: str = "professional"
    trial_days: int = 14


class SubscriptionUpgradeRequest(BaseModel):
    subscription_id: str
    new_tier: str
    billing_cycle: Optional[str] = None


class APIKeyCreateRequest(BaseModel):
    subscription_id: str
    user_id: str
    name: str = ""
    scopes: List[str] = ["read", "write"]
    expires_days: int = 365


class UsageRecordRequest(BaseModel):
    subscription_id: str
    usage_type: str
    amount: int = 1


class FeatureCheckRequest(BaseModel):
    subscription_id: str
    feature: str


# ============================================================
# PRICING ENDPOINTS (Public)
# ============================================================

@router.get("/pricing")
async def get_pricing():
    """
    Get all pricing plans.
    
    Returns pricing for:
    - FREE: $0/mo - Basic features, 3 cases, 50 docs/mo
    - SOLO: $49/mo - Full features for solo practitioners
    - PROFESSIONAL: $199/mo - Team features, 10 users
    - ENTERPRISE: $499+/mo - Unlimited, dedicated support
    """
    return subscription_service.get_pricing()


@router.get("/pricing/compare")
async def compare_plans():
    """Get detailed feature comparison across all plans"""
    pricing = subscription_service.get_pricing()
    
    # Build feature matrix
    all_features = [f.value for f in FeatureCategory]
    
    comparison = {
        "features": []
    }
    
    feature_names = {
        "case_management": "Case Management",
        "document_upload": "Document Upload & Storage",
        "ocr_processing": "OCR Text Extraction",
        "ai_analysis": "AI-Powered Analysis",
        "bwc_video": "Body-Worn Camera Processing",
        "transcription": "Audio Transcription",
        "e_discovery": "E-Discovery & Bates",
        "depositions": "Deposition Management",
        "case_strategy": "Case Strategy Analytics",
        "deadline_calc": "Deadline Calculator",
        "brady_tracking": "Brady/Giglio Tracking",
        "conflict_check": "Conflict Checking",
        "billing": "Legal Billing & Time",
        "legal_research": "Legal Research Tools",
        "pleading_gen": "Pleading Generation",
        "exports": "Reports & Exports",
        "api_access": "API Access"
    }
    
    for feature in all_features:
        feature_row = {
            "id": feature,
            "name": feature_names.get(feature, feature),
            "plans": {}
        }
        
        for plan in pricing["plans"]:
            feature_row["plans"][plan["tier"]] = feature in plan["features"]
        
        comparison["features"].append(feature_row)
    
    comparison["plans"] = pricing["plans"]
    
    return comparison


# ============================================================
# SUBSCRIPTION MANAGEMENT
# ============================================================

@router.post("/subscribe")
async def create_subscription(request: SubscriptionCreateRequest):
    """
    Create a new subscription.
    
    New users start with FREE tier automatically.
    """
    try:
        tier = SubscriptionTier(request.tier)
        cycle = BillingCycle(request.billing_cycle)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    subscription = subscription_service.create_subscription(
        user_id=request.user_id,
        tier=tier,
        billing_cycle=cycle,
        organization_id=request.organization_id
    )
    
    return {
        "subscription_id": subscription.subscription_id,
        "tier": subscription.tier.value,
        "status": subscription.status,
        "message": "Subscription created successfully"
    }


@router.post("/trial/start")
async def start_trial(request: TrialStartRequest):
    """
    Start a 14-day free trial of Professional tier.
    
    No credit card required!
    """
    try:
        tier = SubscriptionTier(request.tier)
    except ValueError:
        tier = SubscriptionTier.PROFESSIONAL
    
    subscription = subscription_service.start_trial(
        user_id=request.user_id,
        tier=tier,
        trial_days=request.trial_days
    )
    
    return {
        "subscription_id": subscription.subscription_id,
        "tier": subscription.tier.value,
        "status": "trial",
        "trial_ends": subscription.trial_end_date,
        "message": f"Your {request.trial_days}-day free trial has started!"
    }


@router.post("/upgrade")
async def upgrade_subscription(request: SubscriptionUpgradeRequest):
    """Upgrade to a higher tier"""
    try:
        new_tier = SubscriptionTier(request.new_tier)
        cycle = BillingCycle(request.billing_cycle) if request.billing_cycle else None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    try:
        subscription = subscription_service.upgrade_subscription(
            subscription_id=request.subscription_id,
            new_tier=new_tier,
            billing_cycle=cycle
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    return {
        "subscription_id": subscription.subscription_id,
        "tier": subscription.tier.value,
        "status": subscription.status,
        "message": f"Upgraded to {new_tier.value} tier"
    }


@router.post("/cancel/{subscription_id}")
async def cancel_subscription(
    subscription_id: str,
    at_period_end: bool = Query(True, description="Cancel at end of billing period")
):
    """Cancel subscription"""
    try:
        subscription = subscription_service.cancel_subscription(
            subscription_id=subscription_id,
            at_period_end=at_period_end
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    if at_period_end:
        message = f"Subscription will cancel on {subscription.end_date}"
    else:
        message = "Subscription cancelled immediately"
    
    return {
        "subscription_id": subscription_id,
        "status": subscription.status,
        "end_date": subscription.end_date,
        "message": message
    }


@router.get("/subscription/{subscription_id}")
async def get_subscription_status(subscription_id: str):
    """Get detailed subscription status and usage"""
    status = subscription_service.get_subscription_status(subscription_id)
    
    if "error" in status:
        raise HTTPException(status_code=404, detail=status["error"])
    
    return status


# ============================================================
# FEATURE ACCESS & USAGE
# ============================================================

@router.post("/access/check")
async def check_feature_access(request: FeatureCheckRequest):
    """
    Check if subscription has access to a feature.
    
    Use this before gating premium features.
    """
    try:
        feature = FeatureCategory(request.feature)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown feature: {request.feature}")
    
    result = subscription_service.check_feature_access(
        subscription_id=request.subscription_id,
        feature=feature
    )
    
    return result


@router.get("/usage/{subscription_id}/limits")
async def get_usage_limits(subscription_id: str):
    """Get current usage vs limits for all limit types"""
    limit_types = ["cases", "documents", "ai_queries", "storage"]
    
    results = {}
    for limit_type in limit_types:
        results[limit_type] = subscription_service.check_usage_limit(
            subscription_id=subscription_id,
            limit_type=limit_type
        )
    
    return {
        "subscription_id": subscription_id,
        "limits": results
    }


@router.post("/usage/record")
async def record_usage(request: UsageRecordRequest):
    """Record usage for billing/limits tracking"""
    subscription_service.record_usage(
        subscription_id=request.subscription_id,
        usage_type=request.usage_type,
        amount=request.amount
    )
    
    # Return updated limit status
    limit_check = subscription_service.check_usage_limit(
        subscription_id=request.subscription_id,
        limit_type=request.usage_type.rstrip('s')  # documents -> document
    )
    
    return {
        "recorded": True,
        "usage_type": request.usage_type,
        "amount": request.amount,
        "current_status": limit_check
    }


# ============================================================
# API KEY MANAGEMENT
# ============================================================

@router.post("/api-keys")
async def create_api_key(request: APIKeyCreateRequest):
    """
    Create a new API key.
    
    ⚠️ The raw key is only returned once - save it securely!
    """
    try:
        api_key, raw_key = subscription_service.create_api_key(
            subscription_id=request.subscription_id,
            user_id=request.user_id,
            name=request.name,
            scopes=request.scopes,
            expires_days=request.expires_days
        )
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    
    return {
        "key_id": api_key.key_id,
        "key": raw_key,  # Only shown once!
        "key_prefix": api_key.key_prefix,
        "name": api_key.name,
        "scopes": api_key.scopes,
        "rate_limit_per_minute": api_key.rate_limit_per_minute,
        "expires": api_key.expires_date,
        "warning": "Save this key securely - it won't be shown again!"
    }


@router.get("/api-keys/{subscription_id}")
async def list_api_keys(subscription_id: str):
    """List all API keys for a subscription (keys are masked)"""
    keys = [
        {
            "key_id": key.key_id,
            "key_prefix": key.key_prefix + "...",
            "name": key.name,
            "scopes": key.scopes,
            "created": key.created_date,
            "last_used": key.last_used,
            "expires": key.expires_date,
            "is_active": key.is_active
        }
        for key in subscription_service.api_keys.values()
        if key.subscription_id == subscription_id
    ]
    
    return {"keys": keys}


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(key_id: str):
    """Revoke an API key"""
    subscription_service.revoke_api_key(key_id)
    return {"revoked": True, "key_id": key_id}


@router.post("/api-keys/validate")
async def validate_api_key(authorization: str = Header(None)):
    """
    Validate an API key from Authorization header.
    
    Use: Authorization: Bearer bx_xxxxx
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="No authorization header")
    
    # Extract key from "Bearer bx_xxxxx"
    parts = authorization.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization format")
    
    raw_key = parts[1]
    api_key = subscription_service.validate_api_key(raw_key)
    
    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid or expired API key")
    
    return {
        "valid": True,
        "key_id": api_key.key_id,
        "scopes": api_key.scopes,
        "rate_limit": api_key.rate_limit_per_minute
    }


# ============================================================
# CHECKOUT (Simulated - integrate with Stripe in production)
# ============================================================

@router.post("/checkout/session")
async def create_checkout_session(
    user_id: str,
    tier: str,
    billing_cycle: str = "monthly"
):
    """
    Create checkout session for payment.
    
    In production, this would integrate with Stripe/PayPal.
    """
    try:
        plan_tier = SubscriptionTier(tier)
        cycle = BillingCycle(billing_cycle)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    plan = subscription_service.plans.get(plan_tier)
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")
    
    # Calculate price
    if cycle == BillingCycle.MONTHLY:
        amount = plan.monthly_price
    elif cycle == BillingCycle.ANNUAL:
        amount = plan.annual_price
    else:
        amount = plan.lifetime_price or plan.annual_price * 3
    
    # In production, create Stripe checkout session here
    return {
        "checkout_url": f"https://checkout.barberx.info/pay?tier={tier}&cycle={billing_cycle}",
        "plan": {
            "tier": tier,
            "name": plan.name,
            "billing_cycle": billing_cycle,
            "amount": amount,
            "currency": "USD"
        },
        "note": "This is a simulated checkout. Integrate with Stripe for production."
    }


# ============================================================
# ADMIN ENDPOINTS
# ============================================================

@router.get("/admin/stats")
async def get_subscription_stats():
    """Get subscription statistics (admin only)"""
    subs = subscription_service.subscriptions.values()
    
    by_tier = {}
    by_status = {}
    
    for sub in subs:
        tier = sub.tier.value
        status = sub.status
        
        by_tier[tier] = by_tier.get(tier, 0) + 1
        by_status[status] = by_status.get(status, 0) + 1
    
    return {
        "total_subscriptions": len(list(subs)),
        "by_tier": by_tier,
        "by_status": by_status,
        "api_keys_issued": len(subscription_service.api_keys)
    }
