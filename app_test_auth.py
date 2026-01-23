"""
BarberX Flask App - Lightweight Version for Testing Auth
Excludes heavy AI dependencies
"""

from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_required, current_user
from datetime import datetime, timedelta
import os

# Initialize Flask app
app = Flask(__name__)

# Configuration
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'barberx-dev-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(basedir, "instance", "barberx_auth.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'auth.login'

# Import enhanced auth
from models_auth import User, UsageTracking, ApiKey, TierLevel
from auth_routes import auth_bp

# Register blueprints
app.register_blueprint(auth_bp, url_prefix='/auth')

# User loader
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ========================================
# ROUTES
# ========================================

@app.route('/')
def index():
    """Homepage"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard with usage tracking"""
    usage = UsageTracking.get_or_create_current(current_user.id)
    limits = current_user.get_tier_limits()
    return render_template('auth/dashboard.html',
                         user=current_user,
                         usage=usage,
                         limits=limits)

@app.route('/login')
def login_redirect():
    """Redirect old login to new auth system"""
    return redirect(url_for('auth.login'))

@app.route('/register')
def register_redirect():
    """Redirect old register to new auth system"""
    return redirect(url_for('auth.signup'))

@app.route('/pricing')
def pricing():
    """Pricing page"""
    return render_template('pricing.html')

@app.route('/docs')
def docs():
    """Documentation"""
    return render_template('docs.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'database': 'connected',
        'auth': 'enabled',
        'timestamp': datetime.utcnow().isoformat()
    })

# ========================================
# ERROR HANDLERS
# ========================================

@app.errorhandler(404)
def not_found(error):
    """404 page"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """500 page"""
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500

# ========================================
# DATABASE INITIALIZATION
# ========================================

with app.app_context():
    db.create_all()
    print("✅ Database tables initialized")

# ========================================
# MAIN ENTRY POINT
# ========================================

if __name__ == '__main__':
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║                                                        ║
    ║        BarberX Platform — Auth Test Server            ║
    ║                                                        ║
    ╚════════════════════════════════════════════════════════╝
    
    🌐 Server: http://localhost:5000
    🔐 Auth Routes:
       • /auth/signup — Create account
       • /auth/login — Login
       • /dashboard — User dashboard
    
    👤 Admin Login:
       Email: dTb33@pm.me
       Password: LoveAll33!
    
    💈✂️ Like a fresh NYC fade — clean, ready to test!
    """)
    
    app.run(debug=True, port=5000, host='0.0.0.0')
