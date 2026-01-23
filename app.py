# BarberX Legal Tech Platform
# Professional BWC Forensic Analysis System
# Copyright (c) 2026 BarberX Legal Technologies

from flask import Flask, request, jsonify, send_file, render_template, redirect, url_for, session, flash
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
from pathlib import Path
import os
import sys
import json
import threading
import hashlib
import uuid
import logging
from logging.handlers import RotatingFileHandler
import flask

# Enhanced authentication imports
try:
    from models_auth import UsageTracking, ApiKey, TierLevel
    from auth_routes import auth_bp
    ENHANCED_AUTH_AVAILABLE = True
except ImportError as e:
    ENHANCED_AUTH_AVAILABLE = False
    print(f"⚠️  Enhanced auth not available: {e}")

# Import our BWC analyzer (optional - only needed for actual analysis)
try:
    from bwc_forensic_analyzer import BWCForensicAnalyzer
    BWC_ANALYZER_AVAILABLE = True
except ImportError:
    BWC_ANALYZER_AVAILABLE = False
    app_logger = logging.getLogger(__name__)
    app_logger.warning("BWC Forensic Analyzer not available - AI dependencies not installed")

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'barberx-legal-tech-2026-secure-key')

# Use absolute path for database
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
    'DATABASE_URL',
    f'sqlite:///{os.path.join(basedir, "instance", "barberx_auth.db")}'
)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5GB
app.config['UPLOAD_FOLDER'] = Path('./uploads/bwc_videos')
app.config['ANALYSIS_FOLDER'] = Path('./bwc_analysis')
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

# Create directories
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)
app.config['ANALYSIS_FOLDER'].mkdir(parents=True, exist_ok=True)

# Initialize extensions
db = SQLAlchemy(app)
CORS(app)
login_manager = LoginManager(app)
login_manager.login_view = 'auth.login'  # Updated to use auth blueprint

# Register enhanced authentication blueprint
if ENHANCED_AUTH_AVAILABLE:
    app.register_blueprint(auth_bp, url_prefix='/auth')
    print("✅ Enhanced auth routes registered at /auth/*")

# Configure logging
if not app.debug:
    if not os.path.exists('logs'):
        os.mkdir('logs')
    file_handler = RotatingFileHandler('logs/barberx.log', maxBytes=10240000, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('BarberX Legal Tech startup')

# Global analyzer instance (lazy load)
analyzer = None
analysis_tasks = {}  # Track background analysis tasks

# ========================================
# DATABASE MODELS
# ========================================

class User(UserMixin, db.Model):
    """User model for authentication and authorization"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    organization = db.Column(db.String(200))
    role = db.Column(db.String(20), default='user')  # user, pro, admin
    subscription_tier = db.Column(db.String(20), default='free')  # free, professional, enterprise
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    is_verified = db.Column(db.Boolean, default=False)
    
    # Relationships
    analyses = db.relationship('Analysis', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    api_keys = db.relationship('APIKey', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    # Usage limits
    analyses_count = db.Column(db.Integer, default=0)
    storage_used_mb = db.Column(db.Float, default=0.0)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def get_tier_limits(self):
        """Get usage limits based on subscription tier"""
        limits = {
            'free': {
                'max_analyses_per_month': 5,
                'max_file_size_mb': 500,
                'max_storage_gb': 5,
                'batch_processing': False,
                'api_access': False,
                'team_collaboration': False,
                'priority_support': False
            },
            'professional': {
                'max_analyses_per_month': 100,
                'max_file_size_mb': 2000,
                'max_storage_gb': 100,
                'batch_processing': True,
                'api_access': True,
                'team_collaboration': True,
                'priority_support': True
            },
            'enterprise': {
                'max_analyses_per_month': -1,  # unlimited
                'max_file_size_mb': 5000,
                'max_storage_gb': -1,  # unlimited
                'batch_processing': True,
                'api_access': True,
                'team_collaboration': True,
                'priority_support': True
            }
        }
        return limits.get(self.subscription_tier, limits['free'])
    
    def can_analyze(self):
        """Check if user can perform analysis based on tier limits"""
        limits = self.get_tier_limits()
        if limits['max_analyses_per_month'] == -1:
            return True
        
        # Count analyses this month
        month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        month_count = self.analyses.filter(Analysis.created_at >= month_start).count()
        
        return month_count < limits['max_analyses_per_month']
    
    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'full_name': self.full_name,
            'organization': self.organization,
            'role': self.role,
            'subscription_tier': self.subscription_tier,
            'created_at': self.created_at.isoformat(),
            'analyses_count': self.analyses_count,
            'tier_limits': self.get_tier_limits()
        }


class Analysis(db.Model):
    """Analysis record for BWC video processing"""
    __tablename__ = 'analyses'
    
    id = db.Column(db.String(32), primary_key=True)  # UUID
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    # File information
    filename = db.Column(db.String(255), nullable=False)
    file_hash = db.Column(db.String(64), nullable=False, index=True)
    file_size = db.Column(db.BigInteger, nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    
    # Case metadata
    case_number = db.Column(db.String(100), index=True)
    evidence_number = db.Column(db.String(100))
    acquired_by = db.Column(db.String(100))
    source = db.Column(db.String(200))
    known_officers = db.Column(db.JSON)
    
    # Analysis status
    status = db.Column(db.String(20), default='uploaded')  # uploaded, analyzing, completed, failed
    progress = db.Column(db.Integer, default=0)
    current_step = db.Column(db.String(100))
    error_message = db.Column(db.Text)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
    
    # Results summary
    duration = db.Column(db.Float)
    total_speakers = db.Column(db.Integer)
    total_segments = db.Column(db.Integer)
    total_discrepancies = db.Column(db.Integer)
    critical_discrepancies = db.Column(db.Integer)
    
    # Report paths
    report_json_path = db.Column(db.String(500))
    report_txt_path = db.Column(db.String(500))
    report_md_path = db.Column(db.String(500))
    
    # Sharing and collaboration
    is_shared = db.Column(db.Boolean, default=False)
    share_token = db.Column(db.String(64), unique=True)
    
    # Tags for organization
    tags = db.Column(db.JSON)
    notes = db.Column(db.Text)
    
    def generate_id(self):
        """Generate unique analysis ID"""
        self.id = uuid.uuid4().hex
    
    def to_dict(self, include_results=False):
        data = {
            'id': self.id,
            'filename': self.filename,
            'file_hash': self.file_hash,
            'file_size': self.file_size,
            'case_number': self.case_number,
            'evidence_number': self.evidence_number,
            'status': self.status,
            'progress': self.progress,
            'created_at': self.created_at.isoformat(),
            'tags': self.tags or []
        }
        
        if include_results and self.status == 'completed':
            data.update({
                'duration': self.duration,
                'total_speakers': self.total_speakers,
                'total_segments': self.total_segments,
                'total_discrepancies': self.total_discrepancies,
                'critical_discrepancies': self.critical_discrepancies,
                'completed_at': self.completed_at.isoformat() if self.completed_at else None
            })
        
        return data


class APIKey(db.Model):
    """API key for programmatic access"""
    __tablename__ = 'api_keys'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    key = db.Column(db.String(64), unique=True, nullable=False, index=True)
    name = db.Column(db.String(100), nullable=False)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_used_at = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    
    def generate_key(self):
        """Generate secure API key"""
        self.key = f"bx_{uuid.uuid4().hex}{uuid.uuid4().hex[:16]}"
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'key': self.key,
            'created_at': self.created_at.isoformat(),
            'last_used_at': self.last_used_at.isoformat() if self.last_used_at else None,
            'is_active': self.is_active
        }


class AuditLog(db.Model):
    """Audit log for compliance and security"""
    __tablename__ = 'audit_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), index=True)
    
    action = db.Column(db.String(50), nullable=False, index=True)
    resource_type = db.Column(db.String(50))
    resource_id = db.Column(db.String(100))
    details = db.Column(db.JSON)
    
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.String(500))
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    @staticmethod
    def log(action, resource_type=None, resource_id=None, details=None):
        """Create audit log entry"""
        log_entry = AuditLog(
            user_id=current_user.id if current_user.is_authenticated else None,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            ip_address=request.remote_addr,
            user_agent=request.headers.get('User-Agent')
        )
        db.session.add(log_entry)
        db.session.commit()


# ========================================
# AUTHENTICATION
# ========================================

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def api_key_required(f):
    """Decorator to require API key authentication"""
    from functools import wraps
    
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
        
        key_obj = APIKey.query.filter_by(key=api_key, is_active=True).first()
        
        if not key_obj:
            return jsonify({'error': 'Invalid API key'}), 401
        
        # Update last used
        key_obj.last_used_at = datetime.utcnow()
        db.session.commit()
        
        # Set current user
        request.current_api_user = key_obj.user
        
        return f(*args, **kwargs)
    
    return decorated_function


# ========================================
# WEB ROUTES
# ========================================

@app.route('/')
def index():
    """Landing page - modern standalone version"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return send_file('index-standalone.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration - redirect to enhanced signup"""
    if ENHANCED_AUTH_AVAILABLE:
        return redirect(url_for('auth.signup'))
    
    # Fallback to old registration logic
    if request.method == 'GET':
        return send_file('templates/register.html')
    
    data = request.get_json()
    
    # Validate input
    if not all([data.get('email'), data.get('password'), data.get('full_name')]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    # Check if user exists
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already registered'}), 400
    
    # Create user
    user = User(
        email=data['email'],
        full_name=data['full_name'],
        organization=data.get('organization', ''),
        subscription_tier='free'
    )
    user.set_password(data['password'])
    
    db.session.add(user)
    db.session.commit()
    
    # Log audit
    AuditLog.log('user_registered', 'user', str(user.id))
    
    # Auto-login
    login_user(user)
    
    app.logger.info(f'New user registered: {user.email}')
    
    return jsonify({
        'message': 'Registration successful',
        'user': user.to_dict()
    })


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login - redirect to enhanced auth"""
    if ENHANCED_AUTH_AVAILABLE:
        return redirect(url_for('auth.login'))
    
    # Fallback to old login logic
    if request.method == 'GET':
        return send_file('templates/login.html')
    
    data = request.get_json()
    
    user = User.query.filter_by(email=data.get('email')).first()
    
    if not user or not user.check_password(data.get('password')):
        return jsonify({'error': 'Invalid email or password'}), 401
    
    if not user.is_active:
        return jsonify({'error': 'Account is disabled'}), 403
    
    login_user(user, remember=data.get('remember', False))
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.session.commit()
    
    # Log audit
    AuditLog.log('user_login', 'user', str(user.id))
    
    return jsonify({
        'message': 'Login successful',
        'user': user.to_dict()
    })


@app.route('/logout')
@login_required
def logout():
    """User logout"""
    AuditLog.log('user_logout', 'user', str(current_user.id))
    logout_user()
    return redirect(url_for('index'))


@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard with usage tracking"""
    if ENHANCED_AUTH_AVAILABLE:
        try:
            usage = UsageTracking.get_or_create_current(current_user.id)
            limits = current_user.get_tier_limits()
            return render_template('auth/dashboard.html',
                                 user=current_user,
                                 usage=usage,
                                 limits=limits)
        except Exception as e:
            app.logger.error(f"Dashboard error: {e}")
            # Fallback to basic dashboard
            return send_file('templates/dashboard.html')
    else:
        return send_file('templates/dashboard.html')


@app.route('/admin')
@login_required
def admin_panel():
    """Admin panel - requires admin role"""
    if current_user.role != 'admin':
        return jsonify({'error': 'Admin access required'}), 403
    return send_file('templates/admin.html')


@app.route('/analyzer')
@login_required
def analyzer():
    """BWC analyzer interface"""
    return send_file('bwc-analyzer.html')


@app.route('/tools/transcript')
@login_required
def transcript_search():
    """Transcript search tool"""
    return send_file('templates/tools/transcript.html')


@app.route('/tools/entity-extract')
@login_required
def entity_extract():
    """Entity extraction tool"""
    return send_file('templates/tools/entity-extract.html')


@app.route('/tools/timeline')
@login_required
def timeline_builder():
    """Timeline builder tool"""
    return send_file('templates/tools/timeline.html')


@app.route('/tools/discrepancy')
@login_required
def discrepancy_finder():
    """Discrepancy finder tool"""
    return send_file('templates/tools/discrepancy.html')


@app.route('/tools/batch')
@login_required
def batch_processor():
    """Batch processor tool"""
    return send_file('templates/tools/batch.html')


@app.route('/tools/api')
@login_required
def api_console():
    """API console tool"""
    return send_file('templates/tools/api-console.html')


# ========================================
# RESOURCE PAGES
# ========================================

@app.route('/docs')
def docs():
    """Documentation home"""
    return send_file('templates/resources/docs.html')


@app.route('/api')
def api_reference():
    """API reference"""
    return send_file('templates/resources/api-reference.html')


@app.route('/blog')
def blog():
    """Blog home"""
    return send_file('templates/resources/blog.html')


@app.route('/case-studies')
def case_studies():
    """Case studies"""
    return send_file('templates/resources/case-studies.html')


@app.route('/guides')
def guides():
    """User guides"""
    return send_file('templates/resources/guides.html')


@app.route('/faq')
def faq():
    """FAQ page"""
    return send_file('templates/resources/faq.html')


@app.route('/animation-demo')
@app.route('/animation-demo.html')
def animation_demo():
    """Animation system demo page"""
    return send_file('animation-demo.html')


# ========================================
# COMPANY PAGES
# ========================================

@app.route('/about')
def about():
    """About us"""
    return send_file('templates/company/about.html')


@app.route('/careers')
def careers():
    """Careers page"""
    return send_file('templates/company/careers.html')


@app.route('/licenses')
@app.route('/company/licenses')
def licenses():
    """Open source licenses and attributions"""
    return send_file('templates/company/licenses.html')


@app.route('/contact')
def contact():
    """Contact page"""
    return send_file('templates/company/contact.html')


@app.route('/press')
def press():
    """Press/media page"""
    return send_file('templates/company/press.html')


# ========================================
# STATIC ASSETS
# ========================================

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """Serve static assets from assets folder"""
    return send_file(os.path.join('assets', filename))


# ========================================
# API ROUTES - Analysis
# ========================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if AI models are loaded"""
    global analyzer
    
    try:
        if analyzer is None:
            hf_token = os.getenv('HUGGINGFACE_TOKEN')
            analyzer = BWCForensicAnalyzer(
                whisper_model_size='base',
                hf_token=hf_token
            )
        
        return jsonify({
            'status': 'ready',
            'models': {
                'whisper': analyzer.whisper_model is not None,
                'pyannote': analyzer.diarization_pipeline is not None,
                'spacy': analyzer.nlp is not None
            },
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        app.logger.error(f'Health check failed: {str(e)}')
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/upload', methods=['POST'])
@login_required
def upload_file():
    """Handle BWC video file upload"""
    
    # Check if user can analyze
    if not current_user.can_analyze():
        return jsonify({
            'error': 'Monthly analysis limit reached. Please upgrade your plan.'
        }), 403
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Secure filename
    filename = secure_filename(file.filename)
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    unique_filename = f"{current_user.id}_{timestamp}_{filename}"
    filepath = app.config['UPLOAD_FOLDER'] / unique_filename
    
    # Save file
    file.save(filepath)
    
    # Calculate hash
    file_hash = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            file_hash.update(chunk)
    file_hash_hex = file_hash.hexdigest()
    
    file_size = os.path.getsize(filepath)
    
    # Create analysis record
    analysis = Analysis(
        user_id=current_user.id,
        filename=filename,
        file_hash=file_hash_hex,
        file_size=file_size,
        file_path=str(filepath),
        status='uploaded'
    )
    analysis.generate_id()
    
    db.session.add(analysis)
    db.session.commit()
    
    # Update user storage
    current_user.storage_used_mb += file_size / (1024 * 1024)
    db.session.commit()
    
    # Log audit
    AuditLog.log('file_uploaded', 'analysis', analysis.id, {'filename': filename})
    
    app.logger.info(f'File uploaded: {filename} by user {current_user.email}')
    
    return jsonify({
        'upload_id': analysis.id,
        'filename': filename,
        'file_hash': file_hash_hex,
        'file_size': file_size,
        'message': 'File uploaded successfully'
    })


@app.route('/api/analyze', methods=['POST'])
@login_required
def analyze_video():
    """Start BWC video analysis"""
    global analyzer
    
    data = request.get_json()
    upload_id = data.get('upload_id')
    
    analysis = Analysis.query.filter_by(id=upload_id, user_id=current_user.id).first()
    
    if not analysis:
        return jsonify({'error': 'Analysis not found'}), 404
    
    # Update case metadata
    analysis.case_number = data.get('case_number', '')
    analysis.evidence_number = data.get('evidence_number', '')
    analysis.acquired_by = data.get('acquired_by', current_user.full_name)
    analysis.source = data.get('source', 'Web Upload')
    analysis.known_officers = data.get('known_officers', [])
    analysis.status = 'analyzing'
    analysis.started_at = datetime.utcnow()
    analysis.progress = 0
    
    db.session.commit()
    
    # Start analysis in background
    def run_analysis():
        try:
            analysis.current_step = 'Extracting audio...'
            analysis.progress = 10
            db.session.commit()
            
            # Run forensic analysis
            report = analyzer.analyze_bwc_file(
                video_path=analysis.file_path,
                acquired_by=analysis.acquired_by,
                source=analysis.source,
                case_number=analysis.case_number,
                evidence_number=analysis.evidence_number,
                known_officers=analysis.known_officers
            )
            
            analysis.progress = 90
            analysis.current_step = 'Generating reports...'
            db.session.commit()
            
            # Save reports
            output_dir = app.config['ANALYSIS_FOLDER'] / analysis.id
            output_files = analyzer.export_report(
                report,
                output_dir=str(output_dir),
                formats=['json', 'txt', 'md']
            )
            
            # Update analysis record
            analysis.status = 'completed'
            analysis.progress = 100
            analysis.current_step = 'Analysis complete'
            analysis.completed_at = datetime.utcnow()
            analysis.duration = report.duration
            analysis.total_speakers = len(report.speakers)
            analysis.total_segments = len(report.transcript)
            analysis.total_discrepancies = len(report.discrepancies)
            analysis.critical_discrepancies = len([d for d in report.discrepancies if d.severity == 'critical'])
            analysis.report_json_path = str(output_dir / 'report.json')
            analysis.report_txt_path = str(output_dir / 'report.txt')
            analysis.report_md_path = str(output_dir / 'report.md')
            
            # Update user stats
            current_user.analyses_count += 1
            
            db.session.commit()
            
            # Log audit
            AuditLog.log('analysis_completed', 'analysis', analysis.id)
            
            app.logger.info(f'Analysis completed: {analysis.id}')
            
        except Exception as e:
            analysis.status = 'failed'
            analysis.error_message = str(e)
            analysis.progress = 0
            db.session.commit()
            
            app.logger.error(f'Analysis failed: {analysis.id} - {str(e)}')
    
    # Start background thread
    thread = threading.Thread(target=run_analysis)
    thread.start()
    
    analysis_tasks[upload_id] = thread
    
    return jsonify({
        'upload_id': upload_id,
        'message': 'Analysis started',
        'status': 'analyzing'
    })


@app.route('/api/analyses', methods=['GET'])
@login_required
def list_analyses():
    """List user's analyses"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    pagination = current_user.analyses.order_by(Analysis.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return jsonify({
        'total': pagination.total,
        'page': page,
        'per_page': per_page,
        'total_pages': pagination.pages,
        'analyses': [a.to_dict(include_results=True) for a in pagination.items]
    })


@app.route('/api/analysis/<analysis_id>', methods=['GET'])
@login_required
def get_analysis(analysis_id):
    """Get analysis details"""
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
    
    if not analysis:
        return jsonify({'error': 'Analysis not found'}), 404
    
    return jsonify(analysis.to_dict(include_results=True))


@app.route('/api/analysis/<analysis_id>/report/<format>', methods=['GET'])
@login_required
def download_report(analysis_id, format):
    """Download analysis report"""
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
    
    if not analysis:
        return jsonify({'error': 'Analysis not found'}), 404
    
    if analysis.status != 'completed':
        return jsonify({'error': 'Analysis not completed'}), 400
    
    # Get report file path
    if format == 'json':
        report_path = analysis.report_json_path
        mimetype = 'application/json'
    elif format == 'txt':
        report_path = analysis.report_txt_path
        mimetype = 'text/plain'
    elif format == 'md':
        report_path = analysis.report_md_path
        mimetype = 'text/markdown'
    else:
        return jsonify({'error': 'Invalid format'}), 400
    
    if not report_path or not Path(report_path).exists():
        return jsonify({'error': 'Report file not found'}), 404
    
    # Log audit
    AuditLog.log('report_downloaded', 'analysis', analysis_id, {'format': format})
    
    return send_file(
        report_path,
        mimetype=mimetype,
        as_attachment=True,
        download_name=f"BWC_Analysis_{analysis.case_number or analysis_id}.{format}"
    )


# ========================================
# API ROUTES - User Management
# ========================================

@app.route('/api/user/profile', methods=['GET', 'PUT'])
@login_required
def user_profile():
    """Get or update user profile"""
    if request.method == 'GET':
        return jsonify(current_user.to_dict())
    
    data = request.get_json()
    
    if 'full_name' in data:
        current_user.full_name = data['full_name']
    if 'organization' in data:
        current_user.organization = data['organization']
    
    db.session.commit()
    
    return jsonify(current_user.to_dict())


@app.route('/api/user/api-keys', methods=['GET', 'POST'])
@login_required
def manage_api_keys():
    """Manage API keys"""
    if request.method == 'GET':
        keys = current_user.api_keys.filter_by(is_active=True).all()
        return jsonify({
            'api_keys': [k.to_dict() for k in keys]
        })
    
    # Create new API key
    data = request.get_json()
    
    if not data.get('name'):
        return jsonify({'error': 'API key name required'}), 400
    
    # Check tier limits
    if current_user.subscription_tier == 'free':
        return jsonify({'error': 'API access requires Professional or Enterprise plan'}), 403
    
    api_key = APIKey(
        user_id=current_user.id,
        name=data['name']
    )
    api_key.generate_key()
    
    db.session.add(api_key)
    db.session.commit()
    
    # Log audit
    AuditLog.log('api_key_created', 'api_key', str(api_key.id))
    
    return jsonify(api_key.to_dict()), 201


# ========================================
# ADMIN ROUTES
# ========================================

# ========================================
# DASHBOARD API ROUTES
# ========================================

@app.route('/api/dashboard-stats', methods=['GET'])
@login_required
def dashboard_stats():
    """Get dashboard statistics for current user"""
    from datetime import timedelta
    from sqlalchemy import extract, func
    
    # Get current month's analysis count
    now = datetime.utcnow()
    first_day = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    analyses_this_month = Analysis.query.filter(
        Analysis.user_id == current_user.id,
        Analysis.created_at >= first_day
    ).count()
    
    # Get tier limits
    tier_limits = current_user.get_tier_limits()
    
    # Get completed count
    completed_count = Analysis.query.filter_by(
        user_id=current_user.id,
        status='completed'
    ).count()
    
    # Get analyzing count
    analyzing_count = Analysis.query.filter_by(
        user_id=current_user.id,
        status='analyzing'
    ).count()
    
    # Get failed count
    failed_count = Analysis.query.filter_by(
        user_id=current_user.id,
        status='failed'
    ).count()
    
    # Get daily activity for last 7 days
    seven_days_ago = now - timedelta(days=7)
    daily_activity = db.session.query(
        func.date(Analysis.created_at).label('date'),
        func.count(Analysis.id).label('count')
    ).filter(
        Analysis.user_id == current_user.id,
        Analysis.created_at >= seven_days_ago
    ).group_by(func.date(Analysis.created_at)).all()
    
    # Format daily activity
    daily_counts = [0] * 7
    for activity in daily_activity:
        days_diff = (now.date() - activity.date).days
        if 0 <= days_diff < 7:
            daily_counts[6 - days_diff] = activity.count
    
    return jsonify({
        'analyses_this_month': analyses_this_month,
        'storage_used_mb': current_user.storage_used_mb,
        'tier_limits': tier_limits,
        'completed_count': completed_count,
        'analyzing_count': analyzing_count,
        'failed_count': failed_count,
        'daily_activity': daily_counts,
        'subscription_tier': current_user.subscription_tier,
        'role': current_user.role
    })


@app.route('/api/analyses', methods=['GET'])
@login_required
def list_analyses():
    """List user's analyses with pagination"""
    limit = request.args.get('limit', 10, type=int)
    offset = request.args.get('offset', 0, type=int)
    status_filter = request.args.get('status')
    
    query = Analysis.query.filter_by(user_id=current_user.id)
    
    if status_filter:
        query = query.filter_by(status=status_filter)
    
    total = query.count()
    analyses = query.order_by(Analysis.created_at.desc()).limit(limit).offset(offset).all()
    
    return jsonify({
        'total': total,
        'limit': limit,
        'offset': offset,
        'analyses': [a.to_dict() for a in analyses]
    })


@app.route('/api/analysis/<analysis_id>', methods=['GET'])
@login_required
def get_analysis(analysis_id):
    """Get specific analysis details"""
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=current_user.id).first()
    
    if not analysis:
        return jsonify({'error': 'Analysis not found'}), 404
    
    return jsonify(analysis.to_dict())


@app.route('/api/subscription/upgrade', methods=['POST'])
@login_required
def upgrade_subscription():
    """Upgrade user subscription tier"""
    data = request.get_json()
    new_tier = data.get('tier')
    
    valid_tiers = ['free', 'professional', 'enterprise']
    
    if new_tier not in valid_tiers:
        return jsonify({'error': 'Invalid tier'}), 400
    
    # In production, integrate with Stripe here
    current_user.subscription_tier = new_tier
    db.session.commit()
    
    # Log audit
    AuditLog.log('subscription_upgraded', 'user', str(current_user.id), {
        'from': current_user.subscription_tier,
        'to': new_tier
    })
    
    return jsonify({
        'message': 'Subscription upgraded successfully',
        'new_tier': new_tier,
        'tier_limits': current_user.get_tier_limits()
    })


@app.route('/api/user/profile', methods=['GET', 'PUT'])
@login_required
def user_profile():
    """Get or update user profile"""
    if request.method == 'GET':
        return jsonify(current_user.to_dict())
    
    # PUT - update profile
    data = request.get_json()
    
    if 'full_name' in data:
        current_user.full_name = data['full_name']
    if 'organization' in data:
        current_user.organization = data['organization']
    
    db.session.commit()
    
    # Log audit
    AuditLog.log('profile_updated', 'user', str(current_user.id))
    
    return jsonify({
        'message': 'Profile updated successfully',
        'user': current_user.to_dict()
    })


@app.route('/api/user/api-keys', methods=['GET'])
@login_required
def list_api_keys():
    """List user's API keys"""
    if current_user.subscription_tier not in ['professional', 'enterprise']:
        return jsonify({'error': 'API keys require Professional or Enterprise tier'}), 403
    
    keys = APIKey.query.filter_by(user_id=current_user.id).all()
    
    return jsonify({
        'total': len(keys),
        'keys': [k.to_dict() for k in keys]
    })


@app.route('/api/user/api-keys/<key_id>', methods=['DELETE'])
@login_required
def delete_api_key(key_id):
    """Delete an API key"""
    key = APIKey.query.filter_by(id=key_id, user_id=current_user.id).first()
    
    if not key:
        return jsonify({'error': 'API key not found'}), 404
    
    db.session.delete(key)
    db.session.commit()
    
    # Log audit
    AuditLog.log('api_key_deleted', 'api_key', str(key_id))
    
    return jsonify({'message': 'API key deleted successfully'})


@app.route('/api/audit-logs', methods=['GET'])
@login_required
def list_audit_logs():
    """List user's audit logs"""
    limit = request.args.get('limit', 50, type=int)
    
    logs = AuditLog.query.filter_by(user_id=current_user.id)\
        .order_by(AuditLog.created_at.desc())\
        .limit(limit).all()
    
    return jsonify({
        'total': len(logs),
        'logs': [{
            'id': log.id,
            'action': log.action,
            'resource_type': log.resource_type,
            'resource_id': log.resource_id,
            'details': log.details,
            'ip_address': log.ip_address,
            'created_at': log.created_at.isoformat()
        } for log in logs]
    })


# ========================================
# ADMIN ROUTES
# ========================================

@app.route('/admin/users', methods=['GET'])
@login_required
def admin_list_users():
    """Admin: List all users"""
    if current_user.role != 'admin':
        return jsonify({'error': 'Admin access required'}), 403
    
    users = User.query.all()
    return jsonify({
        'total': len(users),
        'users': [u.to_dict() for u in users]
    })


@app.route('/admin/users/<int:user_id>', methods=['GET', 'PUT', 'DELETE'])
@login_required
def admin_manage_user(user_id):
    """Admin: Get, update, or delete specific user"""
    if current_user.role != 'admin':
        return jsonify({'error': 'Admin access required'}), 403
    
    user = User.query.get_or_404(user_id)
    
    if request.method == 'GET':
        return jsonify(user.to_dict())
    
    elif request.method == 'PUT':
        data = request.get_json()
        
        # Update allowed fields
        if 'full_name' in data:
            user.full_name = data['full_name']
        if 'organization' in data:
            user.organization = data['organization']
        if 'subscription_tier' in data:
            if data['subscription_tier'] in ['free', 'professional', 'enterprise']:
                user.subscription_tier = data['subscription_tier']
        if 'role' in data:
            if data['role'] in ['user', 'pro', 'admin']:
                user.role = data['role']
        if 'is_active' in data:
            user.is_active = data['is_active']
        if 'is_verified' in data:
            user.is_verified = data['is_verified']
        
        db.session.commit()
        
        # Log audit
        AuditLog.log('admin_user_updated', 'user', str(user_id), {
            'updated_by': current_user.email
        })
        
        return jsonify({
            'message': 'User updated successfully',
            'user': user.to_dict()
        })
    
    elif request.method == 'DELETE':
        # Don't allow deleting yourself
        if user.id == current_user.id:
            return jsonify({'error': 'Cannot delete your own account'}), 400
        
        db.session.delete(user)
        db.session.commit()
        
        # Log audit
        AuditLog.log('admin_user_deleted', 'user', str(user_id), {
            'deleted_by': current_user.email,
            'deleted_email': user.email
        })
        
        return jsonify({'message': 'User deleted successfully'})


@app.route('/admin/users/<int:user_id>/toggle-status', methods=['POST'])
@login_required
def admin_toggle_user_status(user_id):
    """Admin: Enable/disable user account"""
    if current_user.role != 'admin':
        return jsonify({'error': 'Admin access required'}), 403
    
    user = User.query.get_or_404(user_id)
    
    # Don't allow disabling yourself
    if user.id == current_user.id:
        return jsonify({'error': 'Cannot disable your own account'}), 400
    
    user.is_active = not user.is_active
    db.session.commit()
    
    # Log audit
    AuditLog.log('admin_user_status_toggled', 'user', str(user_id), {
        'new_status': 'active' if user.is_active else 'inactive',
        'changed_by': current_user.email
    })
    
    return jsonify({
        'message': f'User {"activated" if user.is_active else "deactivated"} successfully',
        'is_active': user.is_active
    })


@app.route('/admin/users/<int:user_id>/reset-password', methods=['POST'])
@login_required
def admin_reset_user_password(user_id):
    """Admin: Reset user password"""
    if current_user.role != 'admin':
        return jsonify({'error': 'Admin access required'}), 403
    
    user = User.query.get_or_404(user_id)
    data = request.get_json()
    
    if not data.get('new_password'):
        return jsonify({'error': 'New password required'}), 400
    
    user.set_password(data['new_password'])
    db.session.commit()
    
    # Log audit
    AuditLog.log('admin_password_reset', 'user', str(user_id), {
        'reset_by': current_user.email
    })
    
    return jsonify({'message': 'Password reset successfully'})


@app.route('/admin/analyses', methods=['GET'])
@login_required
def admin_list_analyses():
    """Admin: List all analyses"""
    if current_user.role != 'admin':
        return jsonify({'error': 'Admin access required'}), 403
    
    status_filter = request.args.get('status')
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    
    query = Analysis.query
    
    if status_filter:
        query = query.filter_by(status=status_filter)
    
    total = query.count()
    analyses = query.order_by(Analysis.created_at.desc()).limit(limit).offset(offset).all()
    
    return jsonify({
        'total': total,
        'limit': limit,
        'offset': offset,
        'analyses': [a.to_dict() for a in analyses]
    })


@app.route('/admin/analyses/<analysis_id>', methods=['DELETE'])
@login_required
def admin_delete_analysis(analysis_id):
    """Admin: Delete an analysis"""
    if current_user.role != 'admin':
        return jsonify({'error': 'Admin access required'}), 403
    
    analysis = Analysis.query.get_or_404(analysis_id)
    
    # Delete associated file if exists
    if analysis.file_path and os.path.exists(analysis.file_path):
        try:
            os.remove(analysis.file_path)
        except Exception as e:
            app.logger.error(f'Error deleting file: {e}')
    
    # Update user storage
    if analysis.user:
        analysis.user.storage_used_mb -= analysis.file_size / (1024 * 1024)
        if analysis.user.storage_used_mb < 0:
            analysis.user.storage_used_mb = 0
    
    db.session.delete(analysis)
    db.session.commit()
    
    # Log audit
    AuditLog.log('admin_analysis_deleted', 'analysis', analysis_id, {
        'deleted_by': current_user.email
    })
    
    return jsonify({'message': 'Analysis deleted successfully'})


@app.route('/admin/stats', methods=['GET'])
@login_required
def admin_stats():
    """Admin: Get platform statistics"""
    if current_user.role != 'admin':
        return jsonify({'error': 'Admin access required'}), 403
    
    from sqlalchemy import func
    from datetime import timedelta
    
    total_users = User.query.count()
    total_analyses = Analysis.query.count()
    completed_analyses = Analysis.query.filter_by(status='completed').count()
    analyzing = Analysis.query.filter_by(status='analyzing').count()
    failed = Analysis.query.filter_by(status='failed').count()
    
    # Subscription breakdown
    free_users = User.query.filter_by(subscription_tier='free').count()
    pro_users = User.query.filter_by(subscription_tier='professional').count()
    enterprise_users = User.query.filter_by(subscription_tier='enterprise').count()
    
    # Storage stats
    total_storage = db.session.query(func.sum(User.storage_used_mb)).scalar() or 0
    
    # Active users (logged in last 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    active_users = User.query.filter(User.last_login >= thirty_days_ago).count()
    
    # Daily activity for last 7 days
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    daily_analyses = db.session.query(
        func.date(Analysis.created_at).label('date'),
        func.count(Analysis.id).label('count')
    ).filter(
        Analysis.created_at >= seven_days_ago
    ).group_by(func.date(Analysis.created_at)).all()
    
    # Format daily activity
    now = datetime.utcnow()
    daily_counts = [0] * 7
    for activity in daily_analyses:
        days_diff = (now.date() - activity.date).days
        if 0 <= days_diff < 7:
            daily_counts[6 - days_diff] = activity.count
    
    return jsonify({
        'total_users': total_users,
        'active_users': active_users,
        'total_analyses': total_analyses,
        'completed_analyses': completed_analyses,
        'analyzing_count': analyzing,
        'failed_count': failed,
        'success_rate': (completed_analyses / total_analyses * 100) if total_analyses > 0 else 0,
        'subscription_breakdown': {
            'free': free_users,
            'professional': pro_users,
            'enterprise': enterprise_users
        },
        'total_storage_gb': round(total_storage / 1024, 2),
        'daily_activity': daily_counts
    })


@app.route('/admin/audit-logs', methods=['GET'])
@login_required
def admin_audit_logs():
    """Admin: View all audit logs"""
    if current_user.role != 'admin':
        return jsonify({'error': 'Admin access required'}), 403
    
    action_filter = request.args.get('action')
    limit = request.args.get('limit', 100, type=int)
    offset = request.args.get('offset', 0, type=int)
    
    query = AuditLog.query
    
    if action_filter:
        query = query.filter_by(action=action_filter)
    
    total = query.count()
    logs = query.order_by(AuditLog.created_at.desc()).limit(limit).offset(offset).all()
    
    return jsonify({
        'total': total,
        'limit': limit,
        'offset': offset,
        'logs': [{
            'id': log.id,
            'user_id': log.user_id,
            'action': log.action,
            'resource_type': log.resource_type,
            'resource_id': log.resource_id,
            'details': log.details,
            'ip_address': log.ip_address,
            'user_agent': log.user_agent,
            'created_at': log.created_at.isoformat()
        } for log in logs]
    })


@app.route('/admin/system-info', methods=['GET'])
@login_required
def admin_system_info():
    """Admin: Get system information"""
    if current_user.role != 'admin':
        return jsonify({'error': 'Admin access required'}), 403
    
    import psutil
    
    # Database size
    db_path = os.path.join(app.instance_path, 'barberx_legal.db')
    db_size_mb = os.path.getsize(db_path) / (1024 * 1024) if os.path.exists(db_path) else 0
    
    # Upload folder size
    upload_folder = app.config['UPLOAD_FOLDER']
    upload_size_mb = 0
    if os.path.exists(upload_folder):
        for root, dirs, files in os.walk(upload_folder):
            upload_size_mb += sum(os.path.getsize(os.path.join(root, f)) for f in files)
        upload_size_mb = upload_size_mb / (1024 * 1024)
    
    # System metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return jsonify({
        'database_size_mb': round(db_size_mb, 2),
        'upload_storage_mb': round(upload_size_mb, 2),
        'upload_storage_gb': round(upload_size_mb / 1024, 2),
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'memory_used_gb': round(memory.used / (1024**3), 2),
        'memory_total_gb': round(memory.total / (1024**3), 2),
        'disk_percent': disk.percent,
        'disk_used_gb': round(disk.used / (1024**3), 2),
        'disk_total_gb': round(disk.total / (1024**3), 2),
        'python_version': sys.version,
        'flask_version': flask.__version__
    })


# ========================================
# INITIALIZE DATABASE
# ========================================

with app.app_context():
    db.create_all()
    
    # Create default admin user if doesn't exist
    if not User.query.filter_by(email='admin@barberx.info').first():
        admin = User(
            email='admin@barberx.info',
            full_name='BarberX Administrator',
            role='admin',
            subscription_tier='enterprise',
            is_verified=True
        )
        admin.set_password('admin123')  # Change this!
        db.session.add(admin)
        db.session.commit()
        app.logger.info('Default admin user created')


# ========================================
# RUN APPLICATION
# ========================================

if __name__ == '__main__':
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║        BarberX Legal Technologies                              ║
    ║        Professional BWC Forensic Analysis Platform             ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    
    🌐 Web Application: http://localhost:5000
    🔐 Admin Login: admin@barberx.info / admin123
    📊 Database: SQLite (barberx_legal.db)
    📁 Logs: ./logs/barberx.log
    
    Features:
    ✅ Multi-user authentication
    ✅ Role-based access control
    ✅ Subscription tiers (Free, Professional, Enterprise)
    ✅ API key management
    ✅ Audit logging
    ✅ Database persistence
    ✅ Professional dashboard
    
    Ready for production deployment!
    Press Ctrl+C to stop the server.
    """)
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
