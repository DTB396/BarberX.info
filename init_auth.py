"""
BarberX Authentication System Setup
Initialize database and create admin account
"""

from flask import Flask
from models_auth import db, bcrypt, User, TierLevel, UsageTracking, ApiKey
from datetime import datetime, timedelta
import os

# Create Flask app
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'barberx-dev-secret-change-in-production')
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(basedir, "instance", "barberx_auth.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
bcrypt.init_app(app)


def init_database():
    """Initialize database and create tables"""
    with app.app_context():
        # Create all tables
        db.create_all()
        print("✅ Database tables created successfully")
        
        # Check if admin already exists
        admin = User.query.filter_by(email='dTb33@pm.me').first()
        
        if admin:
            print(f"ℹ️  Admin account already exists: {admin.email}")
            return admin
        
        # Create admin account
        admin = User(
            email='dTb33@pm.me',
            full_name='Devon Tyler Barber',
            tier=TierLevel.ADMIN,
            is_active=True,
            is_verified=True,
            is_admin=True,
            subscription_start=datetime.utcnow(),
            subscription_end=datetime.utcnow() + timedelta(days=36500)  # 100 years
        )
        admin.set_password('LoveAll33!')
        
        db.session.add(admin)
        db.session.commit()
        
        print(f"✅ Admin account created successfully!")
        print(f"   Email: {admin.email}")
        print(f"   Tier: {admin.tier_name} (${admin.tier_price}/mo)")
        print(f"   Admin Access: {admin.is_admin}")
        
        # Create initial usage tracking
        usage = UsageTracking.get_or_create_current(admin.id)
        print(f"✅ Usage tracking initialized")
        
        return admin


def create_sample_users():
    """Create sample users for testing different tiers"""
    with app.app_context():
        sample_users = [
            {
                'email': 'free@example.com',
                'password': 'test123',
                'full_name': 'Free Tier User',
                'tier': TierLevel.FREE,
            },
            {
                'email': 'pro@example.com',
                'password': 'test123',
                'full_name': 'Professional User',
                'tier': TierLevel.PROFESSIONAL,
            },
            {
                'email': 'premium@example.com',
                'password': 'test123',
                'full_name': 'Premium User',
                'tier': TierLevel.PREMIUM,
            },
        ]
        
        for user_data in sample_users:
            existing = User.query.filter_by(email=user_data['email']).first()
            if existing:
                print(f"ℹ️  User already exists: {user_data['email']}")
                continue
            
            user = User(
                email=user_data['email'],
                full_name=user_data['full_name'],
                tier=user_data['tier'],
                is_active=True,
                is_verified=True,
                subscription_start=datetime.utcnow(),
                subscription_end=datetime.utcnow() + timedelta(days=30)
            )
            user.set_password(user_data['password'])
            
            db.session.add(user)
            print(f"✅ Created {user_data['tier'].name} user: {user_data['email']}")
        
        db.session.commit()
        print(f"\n✅ Sample users created for testing")


def show_all_users():
    """Display all users in the database"""
    with app.app_context():
        users = User.query.all()
        
        print(f"\n{'='*80}")
        print(f"BarberX User Database")
        print(f"{'='*80}\n")
        
        for user in users:
            print(f"📧 {user.email}")
            print(f"   Name: {user.full_name or 'N/A'}")
            print(f"   Tier: {user.tier_name} (${user.tier_price}/mo)")
            print(f"   Admin: {'Yes' if user.is_admin else 'No'}")
            print(f"   Active: {'Yes' if user.is_active else 'No'}")
            print(f"   Verified: {'Yes' if user.is_verified else 'No'}")
            print(f"   Subscription Active: {'Yes' if user.is_subscription_active else 'No'}")
            print(f"   Created: {user.created_at.strftime('%Y-%m-%d %H:%M')}")
            
            # Show tier limits
            limits = user.get_tier_limits()
            print(f"   Limits:")
            for key, value in limits.items():
                if value == -1:
                    print(f"     • {key}: Unlimited")
                elif isinstance(value, bool):
                    print(f"     • {key}: {value}")
                else:
                    print(f"     • {key}: {value}")
            
            print()


def test_authentication():
    """Test authentication system"""
    with app.app_context():
        print(f"\n{'='*80}")
        print(f"Testing Authentication System")
        print(f"{'='*80}\n")
        
        # Test admin login
        admin = User.query.filter_by(email='dTb33@pm.me').first()
        
        if admin:
            # Test correct password
            if admin.check_password('LoveAll33!'):
                print(f"✅ Admin password verification: SUCCESS")
            else:
                print(f"❌ Admin password verification: FAILED")
            
            # Test wrong password
            if not admin.check_password('wrongpassword'):
                print(f"✅ Wrong password rejection: SUCCESS")
            else:
                print(f"❌ Wrong password rejection: FAILED")
            
            # Test tier limits
            print(f"\n✅ Admin tier limits:")
            limits = admin.get_tier_limits()
            print(f"   • Backend Access: {limits.get('backend_access', False)}")
            print(f"   • Admin Dashboard: {limits.get('admin_dashboard', False)}")
            storage = 'Unlimited' if limits.get('storage_gb') == -1 else f"{limits.get('storage_gb')}GB"
            print(f"   • Storage: {storage}")
            
            # Test feature access
            print(f"\n✅ Feature access checks:")
            print(f"   • Backend Access: {admin.can_access_feature('backend_access')}")
            print(f"   • API Access: {admin.can_access_feature('api_access')}")
            print(f"   • Forensic Analysis: {admin.can_access_feature('forensic_analysis')}")


if __name__ == '__main__':
    print(f"\n{'='*80}")
    print(f"BarberX Authentication System Initializer")
    print(f"{'='*80}\n")
    
    # Create directory for database
    os.makedirs('instance', exist_ok=True)
    
    # Initialize database and create admin
    admin = init_database()
    
    # Create sample users for testing
    create_sample_users()
    
    # Show all users
    show_all_users()
    
    # Test authentication
    test_authentication()
    
    print(f"\n{'='*80}")
    print(f"✅ Setup Complete!")
    print(f"{'='*80}\n")
    print(f"Next steps:")
    print(f"  1. Run the Flask app with authentication enabled")
    print(f"  2. Login with admin credentials:")
    print(f"     Email: dTb33@pm.me")
    print(f"     Password: LoveAll33!")
    print(f"  3. Access backend tools and admin dashboard")
    print(f"\n")
