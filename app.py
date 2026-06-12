# app.py
import os
import pickle
import sqlite3
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'super_secret_session_key_change_this_in_production'
DATABASE = 'database.db'
MODEL_PATH = 'models/document_classifier.pkl'

# Target classes mapping
CLASS_MAP = {0: "Invoice/Financial", 1: "Resume/CV", 2: "Technical Paper"}

# Database Helper Functions
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        # Users Table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        # Classifications History Table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                doc_text TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')
        conn.commit()

# Load the trained Model Pipeline
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

model = load_model()

# --- ROUTES ---

@app.route('/')
def index():
    # If the user is already logged in, bypass landing page and jump to dashboard
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        hashed_pw = generate_password_hash(password)
        
        try:
            db = get_db()
            db.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)', 
                       (username, email, hashed_pw))
            db.commit()
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or Email already exists.', 'danger')
            
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['email'] = user['email']
            flash(f"Welcome back, {user['username']}!", 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'danger')
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session: return redirect(url_for('login'))
    
    db = get_db()
    total = db.execute('SELECT COUNT(*) FROM history WHERE user_id = ?', (session['user_id'],)).fetchone()[0]
    recent = db.execute('SELECT * FROM history WHERE user_id = ? ORDER BY id DESC LIMIT 3', (session['user_id'],)).fetchall()
    
    return render_template('dashboard.html', total=total, recent=recent)

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if 'user_id' not in session: return redirect(url_for('login'))
    
    result = None
    if request.method == 'POST':
        doc_text = request.form['doc_text']
        
        if not model:
            flash('ML Model not trained! Run train_model.py first.', 'danger')
            return render_template('classify.html', result=result)
            
        if doc_text.strip():
            # Get prediction index and calculated probabilities
            pred_class_idx = model.predict([doc_text])[0]
            probabilities = model.predict_proba([doc_text])[0]
            confidence = float(np.max(probabilities)) * 100
            
            prediction_label = CLASS_MAP.get(pred_class_idx, "Unknown")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Save to user session history matrix records
            db = get_db()
            db.execute('''
                INSERT INTO history (user_id, doc_text, prediction, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (session['user_id'], doc_text, prediction_label, round(confidence, 2), timestamp))
            db.commit()
            
            result = {
                'text': doc_text,
                'prediction': prediction_label,
                'confidence': f"{confidence:.2f}%"
            }
            flash('Document processed and saved!', 'success')
            
    return render_template('classify.html', result=result)

@app.route('/history')
def history():
    if 'user_id' not in session: return redirect(url_for('login'))
    
    db = get_db()
    records = db.execute('SELECT * FROM history WHERE user_id = ? ORDER BY id DESC', (session['user_id'],)).fetchall()
    return render_template('history.html', records=records)

@app.route('/analytics')
def analytics():
    if 'user_id' not in session: return redirect(url_for('login'))
    
    db = get_db()
    stats = db.execute('''
        SELECT prediction, COUNT(*) as count 
        FROM history WHERE user_id = ? 
        GROUP BY prediction
    ''', (session['user_id'],)).fetchall()
    
    return render_template('analytics.html', stats=stats)

@app.route('/profile')
def profile():
    if 'user_id' not in session: return redirect(url_for('login'))
    return render_template('profile.html')

if __name__ == '__main__':
    if not os.path.exists(DATABASE):
        init_db()
    app.run(debug=True)