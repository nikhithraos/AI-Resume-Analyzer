import os
import re
import sqlite3
from io import BytesIO
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename
from docx import Document
import PyPDF2
import spacy
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import word_tokenize
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT

nltk.download('punkt', quiet=True)

app = Flask(__name__)
app.secret_key = 'resume-analyzer-secret'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

SKILL_LIST = [
    'python', 'java', 'javascript', 'html', 'css', 'bootstrap', 'flask', 'django',
    'sql', 'sqlite', 'mysql', 'postgresql', 'mongodb', 'pandas', 'numpy', 'nlp',
    'spacy', 'nltk', 'git', 'github', 'rest api', 'machine learning', 'deep learning',
    'data analysis', 'excel', 'power bi', 'tableau', 'aws', 'azure', 'docker', 'linux'
]

try:
    NLP = spacy.load('en_core_web_sm')
except Exception:
    NLP = spacy.blank('en')


def init_db():
    with sqlite3.connect('resume_analyzer.db') as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            name TEXT,
            email TEXT,
            phone TEXT,
            skills TEXT,
            education TEXT,
            ats_score INTEGER,
            job_match INTEGER,
            suggestions TEXT,
            created_at TEXT
        )''')
        conn.commit()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def read_pdf(path):
    text = ''
    with open(path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ''
    return text


def read_docx(path):
    doc = Document(path)
    return '\n'.join(p.text for p in doc.paragraphs)


def extract_text(path):
    if path.lower().endswith('.pdf'):
        return read_pdf(path)
    return read_docx(path)


def extract_email(text):
    m = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return m.group(0) if m else ''


def extract_phone(text):
    patterns = [r'(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}', r'\b\d{10}\b']
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(0)
    return ''


def extract_name(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        first = lines[0]
        if 2 <= len(first.split()) <= 4 and len(first) < 50:
            return first
    doc = NLP(text[:1000])
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            return ent.text
    return ''


def extract_skills(text):
    low = text.lower()
    found = sorted({s for s in SKILL_LIST if s in low})
    missing = sorted(set(SKILL_LIST) - set(found))
    return found, missing


def extract_education(text):
    lines = [l.strip() for l in text.splitlines()]
    kw = ['bachelor', 'master', 'b.tech', 'btech', 'm.tech', 'mba', 'degree', 'university', 'college', 'school', 'diploma']
    hits = [l for l in lines if any(k in l.lower() for k in kw)]
    return hits[:6]


def score_ats(text, skills, education):
    score = 0
    score += min(40, len(skills) * 4)
    if education:
        score += 15
    if re.search(r'project', text, re.I):
        score += 15
    if re.search(r'certificat', text, re.I):
        score += 15
    if re.search(r'intern', text, re.I):
        score += 5
    return min(100, score)


def suggestions(text, skills, education):
    s = []
    if len(skills) < 5:
        s.append('Add more technical skills relevant to the target role.')
    if not re.search(r'project', text, re.I):
        s.append('Add projects with outcomes, tools, and links.')
    if not re.search(r'certificat', text, re.I):
        s.append('Add certifications to strengthen credibility.')
    if not re.search(r'intern', text, re.I):
        s.append('Add internship experience or relevant work exposure.')
    if not education:
        s.append('Add education details with institution and year.')
    return s


def job_match(resume_text, jd_text):
    resume_tokens = set(t.lower() for t in word_tokenize(resume_text) if t.isalpha() and len(t) > 2)
    jd_tokens = set(t.lower() for t in word_tokenize(jd_text) if t.isalpha() and len(t) > 2)
    if not jd_tokens:
        return 0, []
    match = sorted(resume_tokens & jd_tokens)
    pct = round((len(match) / len(jd_tokens)) * 100)
    return min(pct, 100), sorted(jd_tokens - resume_tokens)


def save_analysis(data):
    with sqlite3.connect('resume_analyzer.db') as conn:
        conn.execute('''INSERT INTO analyses
        (filename, name, email, phone, skills, education, ats_score, job_match, suggestions, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
            data['filename'], data['name'], data['email'], data['phone'], ', '.join(data['skills']),
            '\n'.join(data['education']), data['ats_score'], data['job_match'], '\n'.join(data['suggestions']),
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
        conn.commit()


def build_report(data):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=24, leftMargin=24, topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Body2', fontSize=10, leading=14, alignment=TA_LEFT))
    elems = [Paragraph('Resume Analyzer Report', styles['Title']), Spacer(1, 8)]
    rows = [
        ['Name', data['name'] or '-'], ['Email', data['email'] or '-'], ['Phone', data['phone'] or '-'],
        ['ATS Score', str(data['ats_score'])], ['Job Match', f"{data['job_match']}%"],
        ['Extracted Skills', ', '.join(data['skills']) or '-'], ['Education', '<br/>'.join(data['education']) or '-'],
        ['Suggestions', '<br/>'.join(data['suggestions']) or '-']
    ]
    table = Table([[Paragraph(a, styles['Body2']), Paragraph(b, styles['Body2'])] for a, b in rows], colWidths=[120, 360])
    table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.whitesmoke), ('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('VALIGN', (0,0), (-1,-1), 'TOP')]))
    elems.append(table)
    doc.build(elems)
    buffer.seek(0)
    return buffer


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        file = request.files.get('resume')
        jd = request.form.get('job_description', '')
        if not file or file.filename == '' or not allowed_file(file.filename):
            flash('Upload a valid PDF or DOCX file.')
            return redirect(request.url)
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        text = extract_text(path)
        name = extract_name(text)
        email = extract_email(text)
        phone = extract_phone(text)
        skills, missing = extract_skills(text)
        education = extract_education(text)
        ats_score = score_ats(text, skills, education)
        sugg = suggestions(text, skills, education)
        job_pct, missing_kw = job_match(text, jd)
        result = {
            'filename': filename, 'name': name, 'email': email, 'phone': phone, 'skills': skills,
            'missing_skills': missing[:12], 'education': education, 'ats_score': ats_score,
            'suggestions': sugg, 'job_match': job_pct, 'missing_keywords': missing_kw[:20]
        }
        save_analysis(result)
        result['report_url'] = url_for('download_report', filename=filename)
        return render_template('index.html', result=result)
    return render_template('index.html', result=result)


@app.route('/report/<filename>')
def download_report(filename):
    with sqlite3.connect('resume_analyzer.db') as conn:
        cur = conn.execute('SELECT name, email, phone, skills, education, ats_score, job_match, suggestions FROM analyses WHERE filename=? ORDER BY id DESC LIMIT 1', (filename,))
        row = cur.fetchone()
    if not row:
        return redirect(url_for('index'))
    data = {
        'name': row[0], 'email': row[1], 'phone': row[2], 'skills': [s.strip() for s in row[3].split(',') if s.strip()],
        'education': [e for e in row[4].split('\n') if e], 'ats_score': row[5], 'job_match': row[6], 'suggestions': [s for s in row[7].split('\n') if s]
    }
    pdf = build_report(data)
    return send_file(pdf, as_attachment=True, download_name=f'{filename}_report.pdf', mimetype='application/pdf')


if __name__ == '__main__':
    init_db()
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)