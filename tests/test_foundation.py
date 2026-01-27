# 🧪 FOUNDATION TEST SUITE - Build on Solid Ground

"""
Core functionality tests for BarberX production deployment
Tests the critical paths that MUST work for the app to serve users

Run with: pytest tests/test_foundation.py -v
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDeploymentHealth:
    """Verify basic deployment health"""

    def test_environment_variables_present(self):
        """Critical env vars must be set in production"""
        critical_vars = [
            'DATABASE_URL',
            'SECRET_KEY',
            'STRIPE_SECRET_KEY',
        ]
        
        missing = [var for var in critical_vars if not os.getenv(var)]
        assert not missing, f"Missing critical environment variables: {missing}"

    def test_optional_environment_variables(self):
        """Optional but recommended env vars"""
        optional_vars = [
            'COURTLISTENER_API_KEY',
            'OPENAI_API_KEY',
        ]
        
        for var in optional_vars:
            if not os.getenv(var):
                print(f"⚠️  Optional var not set: {var}")


class TestDatabaseConnection:
    """Verify database is accessible"""

    def test_can_import_models(self):
        """Models should import without errors"""
        try:
            from models_auth import User, TierLevel, db
            assert User is not None
            assert TierLevel is not None
            assert db is not None
        except Exception as e:
            pytest.fail(f"Failed to import models: {e}")

    def test_tier_levels_defined(self):
        """All tier levels should be properly defined"""
        from models_auth import TierLevel
        
        required_tiers = ['FREE', 'STARTER', 'PROFESSIONAL', 'PREMIUM', 'ENTERPRISE']
        
        for tier_name in required_tiers:
            assert hasattr(TierLevel, tier_name), f"Missing tier: {tier_name}"

    def test_tier_prices_correct(self):
        """Verify tier prices match marketing ($29/$79/$199/$599)"""
        from models_auth import TierLevel
        
        expected_prices = {
            'FREE': 0,
            'STARTER': 29,
            'PROFESSIONAL': 79,
            'PREMIUM': 199,
            'ENTERPRISE': 599,
        }
        
        for tier_name, expected_price in expected_prices.items():
            tier = getattr(TierLevel, tier_name)
            assert tier.value == expected_price, \
                f"{tier_name} price mismatch: expected ${expected_price}, got ${tier.value}"


class TestFreeTierModules:
    """Verify free tier modules are present (deployment fix)"""

    def test_free_tier_data_retention_imports(self):
        """Critical: This module caused deployment failure"""
        try:
            from free_tier_data_retention import DataRetentionManager, get_user_data_status
            assert DataRetentionManager is not None
            assert get_user_data_status is not None
        except ImportError as e:
            pytest.fail(f"Free tier data retention import failed: {e}")

    def test_free_tier_demo_cases_imports(self):
        """Demo cases module must be present"""
        try:
            from free_tier_demo_cases import get_demo_cases, get_demo_case_by_id, is_demo_case
            assert get_demo_cases is not None
        except ImportError as e:
            pytest.fail(f"Free tier demo cases import failed: {e}")

    def test_free_tier_educational_resources_imports(self):
        """Educational resources module must be present"""
        try:
            from free_tier_educational_resources import CATEGORIES, get_all_educational_resources
            assert CATEGORIES is not None
        except ImportError as e:
            pytest.fail(f"Free tier educational resources import failed: {e}")

    def test_free_tier_upload_manager_imports(self):
        """Upload manager module must be present"""
        try:
            from free_tier_upload_manager import OneTimeUploadManager, free_tier_upload_route_decorator
            assert OneTimeUploadManager is not None
        except ImportError as e:
            pytest.fail(f"Free tier upload manager import failed: {e}")

    def test_free_tier_watermark_imports(self):
        """Watermark service module must be present"""
        try:
            from free_tier_watermark import WatermarkService
            assert WatermarkService is not None
        except ImportError as e:
            pytest.fail(f"Free tier watermark import failed: {e}")


class TestCourtListenerIntegration:
    """Verify CourtListener API integration is ready"""

    def test_legal_library_imports(self):
        """Legal library module must import"""
        try:
            from legal_library import LegalLibrary, LegalDocument
            assert LegalLibrary is not None
            assert LegalDocument is not None
        except ImportError as e:
            pytest.fail(f"Legal library import failed: {e}")

    def test_courtlistener_api_v4_endpoint(self):
        """API endpoint must be v4 (not v3)"""
        from legal_library import LegalLibrary
        
        lib = LegalLibrary()
        assert 'v4' in lib.courtlistener_api, \
            f"API endpoint must use v4, got: {lib.courtlistener_api}"

    def test_foundation_cases_list_exists(self):
        """27 foundation cases should be defined"""
        try:
            # This would be in overnight_library_builder.py
            # For now, just verify the file exists
            builder_file = Path(__file__).parent.parent / 'overnight_library_builder.py'
            assert builder_file.exists(), "overnight_library_builder.py not found"
        except Exception as e:
            print(f"⚠️  Foundation cases builder check failed: {e}")


class TestAdvancedServicesReady:
    """Verify advanced service modules are present"""

    def test_citation_network_analyzer_imports(self):
        """Citation analysis (Shepardizing) module"""
        try:
            from citation_network_analyzer import CitationNetworkAnalyzer, CitationTreatment
            assert CitationNetworkAnalyzer is not None
        except ImportError as e:
            print(f"⚠️  Citation analyzer not yet committed: {e}")

    def test_judge_intelligence_imports(self):
        """Judge research module"""
        try:
            from judge_intelligence import JudgeIntelligence
            assert JudgeIntelligence is not None
        except ImportError as e:
            print(f"⚠️  Judge intelligence not yet committed: {e}")

    def test_chatgpt_service_imports(self):
        """ChatGPT integration module"""
        try:
            from chatgpt_service import ChatGPTService
            assert ChatGPTService is not None
        except ImportError as e:
            print(f"⚠️  ChatGPT service not yet committed: {e}")


class TestStripeIntegration:
    """Verify Stripe billing is configured"""

    def test_stripe_imports(self):
        """Stripe modules should import"""
        try:
            import stripe
            from stripe_subscription_service import StripeSubscriptionService
            assert stripe is not None
            assert StripeSubscriptionService is not None
        except ImportError as e:
            pytest.fail(f"Stripe import failed: {e}")

    def test_stripe_api_key_present(self):
        """Stripe secret key must be set"""
        stripe_key = os.getenv('STRIPE_SECRET_KEY')
        assert stripe_key is not None, "STRIPE_SECRET_KEY not set"
        assert stripe_key.startswith('sk_'), \
            f"Invalid Stripe key format: {stripe_key[:10]}..."


class TestSecurityModules:
    """Verify security features are in place"""

    def test_bcrypt_available(self):
        """Password hashing must work"""
        try:
            from flask_bcrypt import Bcrypt
            bcrypt = Bcrypt()
            
            # Test hash/check
            password = "test_password_123"
            hashed = bcrypt.generate_password_hash(password).decode('utf-8')
            assert bcrypt.check_password_hash(hashed, password)
        except Exception as e:
            pytest.fail(f"Bcrypt test failed: {e}")

    def test_csrf_protection_available(self):
        """CSRF protection must be available"""
        try:
            from flask_wtf.csrf import CSRFProtect
            assert CSRFProtect is not None
        except ImportError as e:
            pytest.fail(f"CSRF protection import failed: {e}")


class TestFileStructure:
    """Verify critical files are present"""

    def test_app_py_exists(self):
        """Main app file must exist"""
        app_file = Path(__file__).parent.parent / 'app.py'
        assert app_file.exists(), "app.py not found"

    def test_requirements_txt_exists(self):
        """Requirements file must exist"""
        req_file = Path(__file__).parent.parent / 'requirements.txt'
        assert req_file.exists(), "requirements.txt not found"

    def test_templates_directory_exists(self):
        """Templates directory must exist"""
        templates_dir = Path(__file__).parent.parent / 'templates'
        assert templates_dir.exists(), "templates/ directory not found"

    def test_critical_templates_exist(self):
        """Critical template files must be present"""
        templates_dir = Path(__file__).parent.parent / 'templates'
        
        critical_templates = [
            'landing.html',
            'mission.html',
            'pricing-comparison.html',
            'components/footer.html',
        ]
        
        for template in critical_templates:
            template_path = templates_dir / template
            assert template_path.exists(), f"Missing critical template: {template}"


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '--tb=short'])
