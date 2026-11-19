from pathlib import Path
from datetime import datetime
import secrets, string, os
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, flash, session, abort
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect
from werkzeug.security import generate_password_hash, check_password_hash
import json
import joblib
import tempfile
import hashlib
from collections import Counter

BASE = Path(__file__).resolve().parent
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-this-development-secret-before-deployment')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + str(BASE / 'complaints.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB
db = SQLAlchemy(app)

class User(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    name = db.Column(db.String(100), nullable=False)

    email = db.Column(db.String(150), unique=True, nullable=False)

    password_hash = db.Column(db.String(255), nullable=False)

    role = db.Column(db.String(20), default='customer', nullable=False)

    department = db.Column(db.String(80))

    is_active = db.Column(
        db.Boolean,
        default=True,
        nullable=False
    )

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Complaint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    reference = db.Column(db.String(24), unique=True, nullable=False)
    title = db.Column(db.String(180), nullable=False)
    description = db.Column(db.Text, nullable=False)
    department = db.Column(db.String(80), nullable=False)
    category_prediction = db.Column(db.String(80))
    sentiment_prediction = db.Column(db.String(30))
    final_category = db.Column(db.String(80))
    status = db.Column(db.String(30), default='Submitted')
    response_note = db.Column(db.Text)
    customer_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    assigned_to = db.Column(db.Integer, db.ForeignKey('user.id'))
    source = db.Column(db.String(30), nullable=False, default='citizen')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    customer = db.relationship('User', foreign_keys=[customer_id], backref='complaints')
    staff = db.relationship('User', foreign_keys=[assigned_to])

class StatusHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    complaint_id = db.Column(db.Integer, db.ForeignKey('complaint.id'), nullable=False)
    status = db.Column(db.String(30), nullable=False)
    note = db.Column(db.Text)
    changed_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    complaint = db.relationship('Complaint', backref='history')
    actor = db.relationship('User')

MODEL_DIR = BASE / 'models'

# Department choices are taken directly from the trained CSV category labels.
# This keeps the citizen form, department statistics and CSV taxonomy consistent.
STATUSES = ['Submitted', 'In Progress', 'Resolved']

def load_models():
    category_path = MODEL_DIR / 'category_model.joblib'
    sentiment_path = MODEL_DIR / 'sentiment_model.joblib'

    if not category_path.exists():
        raise FileNotFoundError(
            "Category model is missing. Train it from the admin dashboard."
        )

    if not sentiment_path.exists():
        raise FileNotFoundError(
            "Sentiment model is missing. Restore or train the sentiment model separately."
        )

    return (
        joblib.load(category_path),
        joblib.load(sentiment_path)
    )
# LOAD BOTH MODELS
category_model, sentiment_model = load_models()
# Exact category labels used by the complaints_processed.csv dataset.
# Keep these identifiers unchanged across citizen forms, staff review and analytics.
DEPARTMENTS = [
    'credit_card',
    'credit_reporting',
    'debt_collection',
    'mortgages_and_loans',
    'retail_banking'
]

# Fail early if the loaded classifier does not use the expected CSV taxonomy.
if set(DEPARTMENTS) != set(map(str, category_model.classes_)):
    raise ValueError(
        'The category model labels do not match the CivicCare CSV categories. '
        f'Expected {DEPARTMENTS}, got {list(category_model.classes_)}'
    )

def current_user():
    return db.session.get(User, session.get('user_id')) if session.get('user_id') else None
@app.context_processor
def inject_globals():
    return {
        'current_user': current_user(),
        'departments': DEPARTMENTS,
        'category_labels': list(category_model.classes_)
    }
def login_required(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if not current_user(): flash('Please log in to continue.', 'warning'); return redirect(url_for('login'))
        return fn(*args, **kwargs)
    return wrapped
def roles(*allowed):
    def deco(fn):
        @wraps(fn)
        @login_required
        def wrapped(*args, **kwargs):
            if current_user().role not in allowed: abort(403)
            return fn(*args, **kwargs)
        return wrapped
    return deco
def make_reference():
    alphabet = string.ascii_uppercase + string.digits
    while True:
        ref = 'CMP-' + ''.join(secrets.choice(alphabet) for _ in range(9))
        if not Complaint.query.filter_by(reference=ref).first(): return ref

@app.route('/')
def home():
    counts = {s: Complaint.query.filter_by(status=s).count() for s in STATUSES}
    return render_template('home.html', counts=counts)
@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        name=request.form.get('name','').strip(); email=request.form.get('email','').strip().lower(); password=request.form.get('password','')
        if not name or not email or len(password)<8: flash('Enter your name, a valid email and a password of at least 8 characters.', 'danger')
        elif User.query.filter_by(email=email).first(): flash('That email is already registered.', 'danger')
        else:
            user=User(name=name,email=email,password_hash=generate_password_hash(password),role='customer')
            db.session.add(user); db.session.commit(); flash('Account created. Please log in.', 'success'); return redirect(url_for('login'))
    return render_template('register.html')
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method=='POST':
        user=User.query.filter_by(email=request.form.get('email','').strip().lower()).first()
        if (
            user
            and user.is_active
            and check_password_hash(
                user.password_hash,
                request.form.get('password', '')
    )
):
            session.clear(); session['user_id']=user.id; flash('Welcome back.', 'success')
            return redirect(url_for('admin_dashboard' if user.role=='admin' else 'staff_dashboard' if user.role=='staff' else 'my_complaints'))
        flash('Invalid email or password.', 'danger')
    return render_template('login.html')
@app.route('/logout')
def logout(): session.clear(); flash('You have been logged out.', 'info'); return redirect(url_for('home'))
@app.route('/complaints/new', methods=['GET','POST'])
@roles('customer')
def new_complaint():
    if request.method=='POST':
        title=request.form.get('title','').strip(); desc=request.form.get('description','').strip(); dept=request.form.get('department','')
        if len(title)<5 or len(desc)<20 or dept not in DEPARTMENTS:
            flash('Please provide a title (5+ characters), details (20+ characters), and choose a valid CSV complaint category.', 'danger')
        else:
            category=category_model.predict([desc])[0]; sentiment=sentiment_model.predict([desc])[0]
            c=Complaint(reference=make_reference(),title=title,description=desc,department=dept,
                        category_prediction=category,final_category=category,sentiment_prediction=sentiment,customer_id=current_user().id)
            db.session.add(c); db.session.flush()
            db.session.add(StatusHistory(complaint_id=c.id,status='Submitted',note='Complaint registered.',changed_by=current_user().id))
            db.session.commit(); flash(f'Complaint submitted. Reference: {c.reference}', 'success'); return redirect(url_for('complaint_detail', complaint_id=c.id))
    return render_template('new_complaint.html')
@app.route('/my-complaints')
@roles('customer')
def my_complaints(): return render_template('complaints.html', complaints=Complaint.query.filter_by(customer_id=current_user().id).order_by(Complaint.created_at.desc()).all(), heading='My complaints')
@app.route('/track', methods=['GET','POST'])
def track():
    complaint=None
    if request.method=='POST':
        ref=request.form.get('reference','').strip().upper(); complaint=Complaint.query.filter_by(reference=ref).first()
        if not complaint: flash('No complaint found with that reference number.', 'warning')
    return render_template('track.html', complaint=complaint)
@app.route('/complaints/<int:complaint_id>')
@login_required
def complaint_detail(complaint_id):
    c=Complaint.query.get_or_404(complaint_id); u=current_user()
    if u.role=='customer' and c.customer_id!=u.id: abort(403)
    if u.role=='staff' and c.assigned_to not in (None,u.id): abort(403)
    return render_template('detail.html', c=c, staff_users=User.query.filter_by(role='staff').all())
@app.route('/staff')
@roles('staff')
def staff_dashboard():
    u = current_user()
    query = Complaint.query.filter(
        (Complaint.assigned_to == u.id) | (Complaint.assigned_to.is_(None))
    ).order_by(Complaint.created_at.desc(), Complaint.id.desc())
    pagination = query.paginate(page=request.args.get('page', 1, type=int), per_page=50, error_out=False)
    return render_template(
        'complaints.html', complaints=pagination.items,
        pagination=pagination, heading='Staff work queue'
    )
@app.route('/admin')
@roles('admin')
def admin_dashboard():
    # Live operational data comes only from complaints submitted to CivicCare.
    complaints = Complaint.query.order_by(Complaint.created_at.desc()).limit(100).all()
    total = Complaint.query.count()
    statuses = {status: Complaint.query.filter_by(status=status).count() for status in STATUSES}
    resolved = statuses.get('Resolved', 0)
    resolution_rate = (resolved / total * 100) if total else 0

    # Use the first recorded Resolved event for each complaint to calculate
    # elapsed resolution time, rather than updated_at (which can change later).
    resolution_hours = []
    resolved_complaints_for_timing = Complaint.query.filter_by(status='Resolved').all()
    for complaint in resolved_complaints_for_timing:
        resolved_events = [h.created_at for h in complaint.history if h.status == 'Resolved' and h.created_at]
        if resolved_events and complaint.created_at:
            elapsed = (min(resolved_events) - complaint.created_at).total_seconds() / 3600
            if elapsed >= 0:
                resolution_hours.append(elapsed)
    average_resolution_hours = (sum(resolution_hours) / len(resolution_hours)) if resolution_hours else None

    category_rows = db.session.query(Complaint.final_category, db.func.count(Complaint.id)).group_by(Complaint.final_category).all()
    categories = {str(label or 'Uncategorised'): count for label, count in category_rows}
    sentiment_rows = db.session.query(Complaint.sentiment_prediction, db.func.count(Complaint.id)).group_by(Complaint.sentiment_prediction).all()
    sentiments = {str(label or 'Unclassified'): count for label, count in sentiment_rows}

    # Department performance uses the citizen-selected responsible department.
    department_stats = []
    for department in DEPARTMENTS:
        department_total = Complaint.query.filter_by(department=department).count()
        department_resolved = Complaint.query.filter_by(department=department, status='Resolved').count()
        department_stats.append({
            'name': department,
            'total': department_total,
            'resolved': department_resolved,
            'resolution_rate': (department_resolved / department_total * 100) if department_total else 0
        })

    # Monthly trend is based on live CivicCare submissions (not the historical CSV).
    now = datetime.utcnow()
    month_keys = []
    year, month = now.year, now.month
    for _ in range(5, -1, -1):
        month_keys.append((year, month))
        month -= 1
        if month == 0:
            month, year = 12, year - 1
    monthly_counts = {key: 0 for key in month_keys}
    trend_rows = db.session.query(
        db.func.strftime('%Y', Complaint.created_at),
        db.func.strftime('%m', Complaint.created_at),
        db.func.count(Complaint.id)
    ).group_by(
        db.func.strftime('%Y', Complaint.created_at),
        db.func.strftime('%m', Complaint.created_at)
    ).all()
    for year_text, month_text, count in trend_rows:
        if year_text and month_text:
            key = (int(year_text), int(month_text))
            if key in monthly_counts:
                monthly_counts[key] = count
    monthly_labels = [datetime(y, m, 1).strftime('%b %Y') for y, m in month_keys]
    monthly_values = [monthly_counts[key] for key in month_keys]

    # Historical dataset summary is precomputed from the supplied CSV.
    dataset_path = BASE / 'models' / 'dataset_analysis.json'
    dataset_analysis = None
    if dataset_path.exists():
        try:
            dataset_analysis = json.loads(dataset_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            dataset_analysis = None

    results_path = MODEL_DIR / 'category_training_results.json'
    training_results = None
    if results_path.exists():
        try:
            training_results = json.loads(results_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            training_results = None

    return render_template(
        'admin.html',
        complaints=complaints,
        categories=categories,
        sentiments=sentiments,
        statuses=statuses,
        staff=User.query.filter_by(role='staff').all(),
        training_results=training_results,
        dataset_analysis=dataset_analysis,
        total_complaints=total,
        resolved_complaints=resolved,
        pending_complaints=total - resolved,
        resolution_rate=resolution_rate,
        average_resolution_hours=average_resolution_hours,
        department_stats=department_stats,
        monthly_labels=monthly_labels,
        monthly_values=monthly_values,
        negative_sentiment_count=sentiments.get('Negative', 0),
        sentiment_is_demo=True
    )
@app.route('/admin/train-category', methods=['POST'])
@roles('admin')
def train_category_model():
    """Analyse the uploaded labelled CSV, train the classifier, and enqueue rows for staff."""
    from ml.train_model import train_category_from_csv
    import pandas as pd

    uploaded_file = request.files.get('dataset')
    if not uploaded_file or not uploaded_file.filename:
        flash('Please select a CSV dataset.', 'danger')
        return redirect(url_for('admin_dashboard'))
    if Path(uploaded_file.filename).suffix.lower() != '.csv':
        flash('Only CSV files are allowed.', 'danger')
        return redirect(url_for('admin_dashboard'))

    temp_path = None
    imported_count = 0
    skipped_count = 0
    try:
        data_dir = BASE / 'data'
        data_dir.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', dir=data_dir, delete=False) as temp_file:
            uploaded_file.save(temp_file)
            temp_path = Path(temp_file.name)

        # Training evaluates on a held-out split, then saves the selected model.
        result = train_category_from_csv(temp_path)
        global category_model
        category_model = joblib.load(MODEL_DIR / 'category_model.joblib')

        # A dedicated non-login customer record owns imported dataset tickets.
        import_email = 'dataset-import@civiccare.local'
        import_user = User.query.filter_by(email=import_email).first()
        if import_user is None:
            import_user = User(
                name='Historical Dataset Import', email=import_email,
                password_hash=generate_password_hash(secrets.token_urlsafe(32)),
                role='customer'
            )
            db.session.add(import_user)
            db.session.commit()

        category_counts = Counter()
        sentiment_counts = Counter()
        valid_count = 0
        missing_count = 0
        batch_size = 500
        row_offset = 0

        # Import each labelled narrative as an unassigned Submitted ticket.
        # Chunking prevents loading 160k+ complaint objects into memory at once.
        for frame in pd.read_csv(temp_path, usecols=['product', 'narrative'], chunksize=batch_size):
            missing_count += int(frame['narrative'].isna().sum())
            frame['product'] = frame['product'].fillna('').astype(str).str.strip()
            frame['narrative'] = frame['narrative'].fillna('').astype(str).str.strip()
            frame = frame[(frame['product'] != '') & (frame['narrative'] != '')]
            if frame.empty:
                row_offset += batch_size
                continue

            texts = frame['narrative'].tolist()
            predicted_categories = category_model.predict(texts).tolist()
            predicted_sentiments = sentiment_model.predict(texts).tolist()
            now = datetime.utcnow()
            mappings = []
            refs = []

            for local_index, (_, row) in enumerate(frame.iterrows()):
                narrative = row['narrative']
                actual_category = row['product']
                absolute_index = row_offset + local_index
                digest = hashlib.sha256(
                    f"{absolute_index}\0{actual_category}\0{narrative}".encode('utf-8')
                ).hexdigest()[:12].upper()
                reference = f'CSV-{digest}'
                refs.append(reference)
                category_counts[actual_category] += 1
                sentiment_counts[str(predicted_sentiments[local_index])] += 1
                valid_count += 1
                mappings.append({
                    'reference': reference,
                    'title': f"Imported complaint — {actual_category.replace('_', ' ').title()[:130]}",
                    'description': narrative,
                    # The supplied records are financial-services complaints.
                    'department': actual_category,
                    'category_prediction': str(predicted_categories[local_index]),
                    'sentiment_prediction': str(predicted_sentiments[local_index]),
                    'final_category': actual_category,
                    'status': 'Submitted',
                    'customer_id': import_user.id,
                    'assigned_to': None,
                    'source': 'dataset_import',
                    'created_at': now,
                    'updated_at': now
                })

            existing = {
                value for (value,) in db.session.query(Complaint.reference)
                .filter(Complaint.reference.in_(refs)).all()
            }
            new_mappings = [item for item in mappings if item['reference'] not in existing]
            skipped_count += len(mappings) - len(new_mappings)
            if new_mappings:
                db.session.bulk_insert_mappings(Complaint, new_mappings)
                db.session.commit()
                imported_count += len(new_mappings)
            row_offset += batch_size

        # Save historical analysis, including inferred sentiment (clearly marked experimental).
        source_name = Path(uploaded_file.filename).name
        dataset_summary = {
            'source_file': source_name,
            'record_count': valid_count,
            'valid_narratives': valid_count,
            'missing_narratives': missing_count,
            'category_count': len(category_counts),
            'category_counts': dict(category_counts),
            'sentiment_counts': dict(sentiment_counts),
            'sentiment_note': 'Sentiment counts are model-inferred and experimental; the CSV contains no ground-truth sentiment labels.',
            'imported_to_staff_queue': imported_count,
            'already_imported': skipped_count,
            'date_analysis_available': False,
            'date_note': 'The source CSV has no complaint date column. Imported tickets receive the import timestamp; this is not the original complaint date.'
        }
        (MODEL_DIR / 'dataset_analysis.json').write_text(json.dumps(dataset_summary, indent=2), encoding='utf-8')

        flash(
            f"Dataset analysed. {imported_count:,} complaints added to the staff queue as Submitted; "
            f"{skipped_count:,} matching rows were already imported. Selected category model: {result['selected_model']}.",
            'success'
        )
    except Exception as error:
        db.session.rollback()
        flash(f'Dataset processing stopped after {imported_count:,} imported rows: {error}', 'danger')
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()
    return redirect(url_for('admin_dashboard'))

@app.route('/complaints/<int:complaint_id>/update',methods=['POST'])
@roles('staff','admin')
def update_complaint(complaint_id):
    c=Complaint.query.get_or_404(complaint_id); u=current_user()
    status=request.form.get('status'); category=request.form.get('category'); note=request.form.get('note','').strip()
    if status not in STATUSES or category not in category_model.classes_:abort(400)
    if u.role=='staff' and c.assigned_to not in (None,u.id): abort(403)
    c.status=status; c.final_category=category; c.response_note=note or c.response_note
    if u.role=='staff' and c.assigned_to is None: c.assigned_to=u.id
    if u.role=='admin':
        assignee=request.form.get('assigned_to','').strip()
        if assignee:
            staff_user=User.query.filter_by(id=int(assignee), role='staff').first()
            if not staff_user: abort(400)
            c.assigned_to=staff_user.id
        elif 'assigned_to' in request.form:
            c.assigned_to=None
    db.session.add(StatusHistory(complaint_id=c.id,status=status,note=note,changed_by=u.id)); db.session.commit()
    flash('Complaint updated.', 'success'); return redirect(url_for('complaint_detail',complaint_id=c.id))
@app.route('/complaints/<int:complaint_id>/delete', methods=['POST'])
@roles('admin')
def delete_complaint(complaint_id):
    complaint = Complaint.query.get_or_404(complaint_id)

    # Delete complaint history first
    StatusHistory.query.filter_by(
        complaint_id=complaint.id
    ).delete(synchronize_session=False)

    # Delete the complaint
    db.session.delete(complaint)
    db.session.commit()

    flash('Complaint deleted successfully.', 'success')
    return redirect(url_for('admin_dashboard'))
@app.route('/admin/users',methods=['GET','POST'])
@roles('admin')
def manage_users():
    if request.method=='POST':
        name=request.form.get('name','').strip(); email=request.form.get('email','').strip().lower(); password=request.form.get('password',''); role=request.form.get('role','staff'); dept=request.form.get('department')
        if not name or not email or len(password)<8 or role not in ('staff','admin') or User.query.filter_by(email=email).first(): flash('Check the details. Password must be 8+ characters and email must be unique.', 'danger')
        else:
            db.session.add(User(name=name,email=email,password_hash=generate_password_hash(password),role=role,department=dept)); db.session.commit(); flash('Account created.', 'success'); return redirect(url_for('manage_users'))
    return render_template(
    'users.html',
    users=User.query.order_by(User.created_at.desc()).all()
)
@app.route('/admin/users/<int:user_id>/remove', methods=['POST'])
@roles('admin')
def remove_user(user_id):

    admin = current_user()

    user = db.session.get(User, user_id)

    if not user:
        flash('User not found.', 'danger')
        return redirect(url_for('manage_users'))

    # Prevent administrator from removing their own account
    if user.id == admin.id:
        flash('You cannot remove your own account.', 'danger')
        return redirect(url_for('manage_users'))

    # Prevent removal of the last active administrator
    if user.role == 'admin' and user.is_active:

        active_admins = User.query.filter_by(
            role='admin',
            is_active=True
        ).count()

        if active_admins <= 1:
            flash(
                'You cannot remove the last active administrator.',
                'danger'
            )
            return redirect(url_for('manage_users'))

    # Deactivate the account
    user.is_active = False

    # Remove staff assignments from complaints
    if user.role == 'staff':

        Complaint.query.filter_by(
            assigned_to=user.id
        ).update(
            {'assigned_to': None},
            synchronize_session=False
        )

    db.session.commit()

    flash(
        f'Account belonging to {user.name} has been deactivated.',
        'success'
    )

    return redirect(url_for('manage_users'))
with app.app_context():
    db.create_all()
        # Add is_active column to existing databases
    existing_user_columns = {
        column['name']
        for column in inspect(db.engine).get_columns('user')
    }

    if 'is_active' not in existing_user_columns:

        with db.engine.begin() as connection:

            connection.exec_driver_sql(
                "ALTER TABLE user "
                "ADD COLUMN is_active BOOLEAN NOT NULL DEFAULT 1"
            )
    # Backward-compatible migration for databases created before dataset imports.
    existing_columns = {column['name'] for column in inspect(db.engine).get_columns('complaint')}
    if 'source' not in existing_columns:
        with db.engine.begin() as connection:
            connection.exec_driver_sql("ALTER TABLE complaint ADD COLUMN source VARCHAR(30) NOT NULL DEFAULT 'citizen'")
    if not User.query.filter_by(email='admin@demo.local').first():
        db.session.add(User(name='Demo Administrator',email='admin@demo.local',password_hash=generate_password_hash('Admin@12345'),role='admin'))
        db.session.add(User(name='Demo Staff',email='staff@demo.local',password_hash=generate_password_hash('Staff@12345'),role='staff',department='retail_banking'))
        db.session.commit()

if __name__ == '__main__': app.run(debug=True)
