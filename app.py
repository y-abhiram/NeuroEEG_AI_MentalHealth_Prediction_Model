
# === Auto-download model from Google Drive if missing ===
import os
import gdown

MODEL_PATH = "models/eeg_models/voting_model.pkl"
GOOGLE_DRIVE_FILE_ID = "1Cd0Bav8E1GF8lN29_WkpLuJO9xYOLZfB"

if not os.path.exists(MODEL_PATH):
    print("Model not found. Downloading from Google Drive...")
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    gdown.download(url, MODEL_PATH, quiet=False)

    #  Ensure download succeeded
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model download failed. Please check your internet connection or the file ID.")

from flask import Flask, render_template, request, session, redirect, url_for, send_file
import time, csv, os

import datetime 
from report_generator import generate_pdf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Prevent TensorFlow from using GPU

import matplotlib
matplotlib.use('Agg')  # Prevent GUI/OpenGL errors
from behavioral import run_behavioral_data_merge
from eeg_utils import run_eeg_merge

# from behavioral import run_behavioral_analysis
from behavioral import run_live_behavioral
app = Flask(__name__)
app.secret_key = 'your_secret_key'
os.makedirs("reports", exist_ok=True)

fs = 256

# === Questions ===
gad7_questions = [
    "Feeling nervous, anxious, or on edge",
    "Not being able to stop or control worrying",
    "Worrying too much about different things",
    "Trouble relaxing",
    "Being so restless that it's hard to sit still",
    "Becoming easily annoyed or irritable",
    "Feeling afraid as if something awful might happen"
]

phq9_questions = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling/staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself",
    "Trouble concentrating",
    "Moving/speaking slowly or fidgety/restless",
    "Thoughts of self-harm or feeling better off dead"
]

pss_questions = [
    ("Upset because something unexpected happened", False),
    ("Felt unable to control important things in life", False),
    ("Felt nervous or stressed", False),
    ("Confident about ability to handle personal problems", True),
    ("Felt things were going your way", True),
    ("Could not cope with all the things you had to do", False),
    ("Could control irritations in your life", True),
    ("Felt on top of things", True),
    ("Angered because things were out of control", False),
    ("Difficulties were piling up", False)
]

adhd_questions = [
    "Trouble wrapping up final details of a project",
    "Difficulty getting things in order",
    "Problems remembering appointments/obligations",
    "Avoiding or delaying things that require thought",
    "Fidgeting or restlessness",
    "Feeling overly active or driven"
]

gad7_labels = ["Not at all", "Several days", "More than half the days", "Nearly every day"]
phq9_labels = gad7_labels
pss_labels = ["Never", "Almost Never", "Sometimes", "Fairly Often", "Very Often"]
adhd_labels = ["Never", "Rarely", "Sometimes", "Often", "Very Often"]

category_names = {
    'gad': 'GAD-7 (Anxiety), In the last 2 weeks',
    'phq': 'PHQ-9 (Depression), Over the last 2 weeks',
    'pss': 'PSS (Stress), Over the last month',
    'adhd': 'ADHD (Over the past 6 months)'
}

all_questions = (
    [('gad', q, gad7_labels) for q in gad7_questions] +
    [('phq', q, phq9_labels) for q in phq9_questions] +
    [('pss', q[0], pss_labels) for q in pss_questions] +
    [('adhd', q, adhd_labels) for q in adhd_questions]
)

# === Routes ===
@app.route('/', methods=['GET', 'POST'])
def personal_info():
    if request.method == 'POST':
        session['personal'] = {
            'name': request.form['name'],
            'age': request.form['age'],
            'email': request.form['email'],
            'gender': request.form['gender']
        }
        return redirect(url_for('dashboard'))

    return render_template('personal_info.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/questionnaire_menu')
def questionnaire_menu():
    return render_template('questionnaire_menu.html')
    



@app.route('/behavioral', methods=['GET', 'POST'])
def run_behavioral():
    if request.method == 'POST':
        personal_info = session.get('personal', {})
        result, path = run_live_behavioral(personal_info)
        return render_template('behavioral_result.html', result=result, pdf_path=path)

    return render_template('behavioral_start.html')

'''
@app.route('/run_all')
def run_all():
    return redirect(url_for('run_behavioral'))
@app.route('/run_all_sequence')
def run_all_sequence():
    start_eeg_collection()  # EEG starts only here
    session['run_all_mode'] = True  # Used later in submit and frontend
    return redirect(url_for('questionnaire_category', category='all', start='true'))


@app.route('/run_all_sequence', methods=['GET', 'POST'])


def run_all_sequence():
    print("✅ Starting EEG and Behavioral in background...")
    start_eeg_collection()
    session['run_all_mode'] = True  # Flag to detect during final report merge

    # Optionally start behavioral analysis (if needed as backend task)
    # run_live_behavioral(session.get('personal'))

    return redirect(url_for('questionnaire_category', category='all', start='true'))

'''
import threading


@app.route('/run_all_sequence', methods=['GET', 'POST'])
def run_all_sequence():
    personal_info = session.get('personal', {'name': 'anonymous'})

    if request.method == 'GET':
        # Show the confirmation/start page
        return render_template("run_all_start.html")

    elif request.method == 'POST':
        print("✅ Starting EEG and Behavioral in background...")

        # Start EEG in background
        start_eeg_collection()

        # Start Behavioral in background thread
        def run_behavioral_bg():
            try:
                summary, csv_path = run_live_behavioral(personal_info)
                print("✅ Behavioral saved:", csv_path)
            except Exception as e:
                print("❌ Behavioral failed:", e)

        threading.Thread(target=run_behavioral_bg, daemon=True).start()

        session['run_all_mode'] = True
        return redirect(url_for('questionnaire_category', category='all', start='true'))


'''
#############################
@app.route('/run_all_sequence', methods=['GET', 'POST'])
def run_all_sequence():
    print("✅ Starting EEG and Behavioral in background...")

    personal_info = session.get('personal', {'name': 'anonymous'})

    # Start EEG in background
    start_eeg_collection()

    # Start Behavioral in background thread
    def run_behavioral_bg():
        try:
            summary, csv_path = run_live_behavioral(personal_info)
            print("✅ Behavioral saved:", csv_path)
        except Exception as e:
            print("❌ Behavioral failed:", e)

    threading.Thread(target=run_behavioral_bg, daemon=True).start()

    # Go to questionnaire
    session['run_all_mode'] = True
    return redirect(url_for('questionnaire_category', category='all', start='true'))
###################################
'''

'''
@app.route('/run_all_sequence', methods=['GET', 'POST'])
def run_all_sequence():
    print("✅ Starting EEG and Behavioral in background...")
    personal_info = session.get('personal', {'name': 'anonymous'})

    # Start EEG collection in background
    start_eeg_collection()

    # Run behavioral analysis and save report
    try:
        behavioral_result, behavioral_pdf_path = run_live_behavioral(personal_info)
        csv_path = f"reports/{personal_info['name'].replace(' ', '_')}_behavioral.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(behavioral_result.keys())
            writer.writerow(behavioral_result.values())
        print("✅ Behavioral report saved.")
    except Exception as e:
        print(f"❌ Behavioral report error: {e}")

    # Run EEG processing and save report
    try:
        eeg_result = stop_and_process_eeg(personal_info)
        filename_base = personal_info['name'].replace(" ", "_") + "_eeg"
        csv_path = f"reports/{filename_base}.csv"
        pdf_path = f"reports/{filename_base}.pdf"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                personal_info['name'], personal_info['age'], personal_info['email'], personal_info['gender'],
                eeg_result['rule_based'], eeg_result['total_samples']
            ] + [f"{label}:{count}" for label, count in eeg_result['ml_summary'].items()])
        generate_pdf(personal_info, eeg_result, pdf_path)
        print("✅ EEG report saved.")
    except Exception as e:
        print(f"❌ EEG report error: {e}")

    # Set flag to merge later
    session['run_all_mode'] = True
    return redirect(url_for('questionnaire_category', category='all', start='true'))
'''

@app.route('/start_behavioral_background')
def start_behavioral_background():
    # Optionally set a flag or trigger a WebRTC-compatible handler
    print("✅ Behavioral started (frontend)")
    return '', 200

@app.route('/stop_behavioral_background')
def stop_behavioral_background():
    # This will be called just before final questionnaire submit
    print("🛑 Behavioral stopped (frontend)")
    return '', 200

@app.route('/questionnaire/<category>', methods=['GET', 'POST'])
def questionnaire_category(category):
    if 'personal' not in session:
        return redirect(url_for('personal_info'))

    # === Load questions based on category ===
    if category == 'all':
        questions = all_questions
    elif category == 'gad':
        questions = [('gad', q, gad7_labels) for q in gad7_questions]
    elif category == 'phq':
        questions = [('phq', q, phq9_labels) for q in phq9_questions]
    elif category == 'pss':
        questions = [('pss', q[0], pss_labels) for q in pss_questions]
    elif category == 'adhd':
        questions = [('adhd', q, adhd_labels) for q in adhd_questions]
    else:
        return "Invalid category", 400

    # === Session keys ===
    step_key = f'step_{category}'
    answer_key = f'answers_{category}'
    time_key = f'times_{category}'
    start_time_key = f'start_time_{category}'

    # === On first GET with start=true → initialize session tracking
    if request.method == 'GET' and request.args.get('start') == 'true':
        session[step_key] = 0
        session[answer_key] = [None] * len(questions)
        session[time_key] = [0] * len(questions)
        session[start_time_key] = time.time()

    # === Load from session
    step = session.get(step_key, 0)
    answers = session.get(answer_key, [None] * len(questions))
    times = session.get(time_key, [0] * len(questions))
    start_time = session.get(start_time_key, time.time())

    # === Handle POST (next/back/submit)
    if request.method == 'POST':
        q_start = session.pop('question_start', time.time())
        q_time = round(time.time() - q_start, 2)
        current_answer = int(request.form['answer'])

        answers[step] = current_answer
        times[step] = q_time

        # === Block-wise save for 'all' category
        if category == 'all':
            block_ranges = {
                'gad': (0, 7),
                'phq': (7, 16),
                'pss': (16, 26),
                'adhd': (26, 32)
            }
            for subcat, (start, end) in block_ranges.items():
                if start <= step < end:
                    sub_step = step - start
                    sub_answers = session.get(f'answers_{subcat}', [None] * (end - start))
                    sub_times = session.get(f'times_{subcat}', [0] * (end - start))
                    sub_answers[sub_step] = current_answer
                    sub_times[sub_step] = q_time
                    session[f'answers_{subcat}'] = sub_answers
                    session[f'times_{subcat}'] = sub_times
                    if f'start_time_{subcat}' not in session:
                        session[f'start_time_{subcat}'] = session[start_time_key]
                    break

        # === Submit → redirect to results
        if 'submit' in request.form:
            session[step_key] = step + 1
            session[answer_key] = answers
            session[time_key] = times
            return redirect(url_for('submit_category', category=category))

        # === Navigation
        if 'next' in request.form:
            step += 1
        elif 'back' in request.form:
            step = max(0, step - 1)

        session[step_key] = step
        session[answer_key] = answers
        session[time_key] = times

        return redirect(url_for('questionnaire_category', category=category))

    # === If step exceeds questions → go to results
    if step >= len(questions):
        return redirect(url_for('submit_category', category=category))

    # === Load question for rendering
    q_type, question_text, labels = questions[step]
    category_name = category_names.get(q_type, q_type.upper())
    session['question_start'] = time.time()

    return render_template('question.html',
                           step=step + 1,
                           total=len(questions),
                           question=question_text,
                           labels=labels,
                           category=category_name,
                           category_slug=category,
                           answer=answers[step],
                           current_time=round(time.time() - start_time, 2),
                           is_last=(step == len(questions) - 1))

'''
@app.route('/submit/<category>', methods=['GET', 'POST'])
def submit_category(category):

    answers = session.pop(f'answers_{category}', [])
    times = session.pop(f'times_{category}', [])
    total_time = round(time.time() - session.pop(f'start_time_{category}', time.time()), 2)
    personal_info = session.get('personal', {})

    result = {}
    if category == 'gad':
        if None in answers:
         return "⚠️ You have unanswered questions. Please complete all questions before submitting.", 400

        score = sum(a for a in answers if a is not None)
        level = "Minimal" if score <= 4 else "Mild" if score <= 9 else "Moderate" if score <= 14 else "Severe"
        result = {'GAD-7 Score': score, 'Anxiety Level': level}
    elif category == 'phq':
     if None in answers:
        return "⚠️ You have unanswered questions. Please complete all questions before submitting.", 400

     score = sum(a for a in answers if a is not None)
     level = "Minimal" if score <= 4 else "Mild" if score <= 9 else "Moderate" if score <= 14 else "Moderately Severe" if score <= 19 else "Severe"
     result = {'PHQ-9 Score': score, 'Depression Level': level}

    elif category == 'pss':
      score = 0
      for i, (_, reverse) in enumerate(pss_questions):
        val = answers[i]
        if val is None:
            return "⚠️ You have unanswered questions in PSS. Please complete all questions.", 400
        if reverse:
            val = 4 - val
        score += val
      level = "Low" if score <= 13 else "Moderate" if score <= 26 else "High"
      result = {'PSS Score': score, 'Stress Level': level}

    elif category == 'adhd':
        count = sum(1 for val in answers if val is not None and val >= 4)
        result = {'ADHD Status': "Suggestive of ADHD" if count >= 4 else "Not Suggestive"}

    result['Total Time (seconds)'] = total_time
    
    
    # Save results to CSV
    csv_path = f"reports/{personal_info['name'].replace(' ', '_')}_{category}.csv"
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([personal_info['name'], personal_info['age'], personal_info['email'], personal_info['gender']] + list(result.values()))

    # Generate PDF
    pdf_path = f"reports/{personal_info['name'].replace(' ', '_')}_{category}.pdf"
    generate_pdf(personal_info, result, pdf_path)

    return render_template('result.html',
                           result=result,
                           personal=personal_info,
                           pdf_path=pdf_path,
                           csv_path=csv_path)
 
  
  
@app.route('/run_eeg')
def run_eeg():
    return redirect(url_for('eeg_start'))

'''
'''
@app.route('/submit/<category>', methods=['GET', 'POST'])
def submit_category(category):
    personal_info = session.get('personal', {})
    result = {}

    # === HANDLE INDIVIDUAL BLOCKS ===
    if category in ['gad', 'phq', 'pss', 'adhd']:
        answers = session.get(f'answers_{category}', [])
        times = session.get(f'times_{category}', [])
        total_time = round(time.time() - session.get(f'start_time_{category}', time.time()), 2)

        # Check if any question is unanswered
        if None in answers or len(answers) == 0:
            return "⚠️ You have unanswered questions. Please complete all questions before submitting.", 400

        # Handle GAD-7
        if category == 'gad':
            score = sum(answers)
            level = "Minimal" if score <= 4 else "Mild" if score <= 9 else "Moderate" if score <= 14 else "Severe"
            result = {'GAD-7 Score': score, 'Anxiety Level': level}

        # Handle PHQ-9
        elif category == 'phq':
            score = sum(answers)
            level = "Minimal" if score <= 4 else "Mild" if score <= 9 else "Moderate" if score <= 14 else "Moderately Severe" if score <= 19 else "Severe"
            result = {'PHQ-9 Score': score, 'Depression Level': level}

        # Handle PSS
        elif category == 'pss':
            score = 0
            for i, (_, reverse) in enumerate(pss_questions):
                val = answers[i]
                if reverse:
                    val = 4 - val
                score += val
            level = "Low" if score <= 13 else "Moderate" if score <= 26 else "High"
            result = {'PSS Score': score, 'Stress Level': level}

        # Handle ADHD
        elif category == 'adhd':
            count = sum(1 for val in answers if val >= 4)
            result = {'ADHD Status': "Suggestive of ADHD" if count >= 4 else "Not Suggestive"}

        # Add Total Time
        result['Total Time (seconds)'] = total_time

        # Save results
        csv_path = f"reports/{personal_info['name'].replace(' ', '_')}_{category}.csv"
        pdf_path = f"reports/{personal_info['name'].replace(' ', '_')}_{category}.pdf"
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                personal_info['name'], personal_info['age'], personal_info['email'], personal_info['gender']
            ] + list(result.values()))
        generate_pdf(personal_info, result, pdf_path)

        # Clear session data after successful result
        session.pop(f'answers_{category}', None)
        session.pop(f'times_{category}', None)
        session.pop(f'start_time_{category}', None)

        return render_template('result.html',
                               result=result,
                               personal=personal_info,
                               pdf_path=pdf_path,
                               csv_path=csv_path)

    # === HANDLE "all" CATEGORY ===
    elif category == 'all':
        result = {}
        total_time = 0

        for subcat, label in zip(['gad', 'phq', 'pss', 'adhd'], ['GAD-7', 'PHQ-9', 'PSS', 'ADHD']):
            sub_answers = session.get(f'answers_{subcat}', [])
            sub_times = session.get(f'times_{subcat}', [])
            sub_total_time = round(time.time() - session.get(f'start_time_{subcat}', time.time()), 2)
            total_time += sub_total_time

            # Check for unanswered questions in each sub-category
            if len(sub_answers) == 0 or any(a is None for a in sub_answers):
                return f"⚠️ You have unanswered questions in {label}.", 400

            # Handle GAD-7
            if subcat == 'gad':
                score = sum(sub_answers)
                level = "Minimal" if score <= 4 else "Mild" if score <= 9 else "Moderate" if score <= 14 else "Severe"
                result.update({'GAD-7 Score': score, 'Anxiety Level': level})

            # Handle PHQ-9
            elif subcat == 'phq':
                score = sum(sub_answers)
                level = "Minimal" if score <= 4 else "Mild" if score <= 9 else "Moderate" if score <= 14 else "Moderately Severe" if score <= 19 else "Severe"
                result.update({'PHQ-9 Score': score, 'Depression Level': level})

            # Handle PSS
            elif subcat == 'pss':
                score = 0
                for i, (_, reverse) in enumerate(pss_questions):
                    val = sub_answers[i]
                    if reverse:
                        val = 4 - val
                    score += val
                level = "Low" if score <= 13 else "Moderate" if score <= 26 else "High"
                result.update({'PSS Score': score, 'Stress Level': level})

            # Handle ADHD
            elif subcat == 'adhd':
                count = sum(1 for val in sub_answers if val >= 4)
                result.update({'ADHD Status': "Suggestive of ADHD" if count >= 4 else "Not Suggestive"})

        result['Total Time (seconds)'] = total_time

        # Save results
        csv_path = f"reports/{personal_info['name'].replace(' ', '_')}_all.csv"
        pdf_path = f"reports/{personal_info['name'].replace(' ', '_')}_all.pdf"
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                personal_info['name'], personal_info['age'], personal_info['email'], personal_info['gender']
            ] + list(result.values()))
        generate_pdf(personal_info, result, pdf_path)

        # Clear session data after successful result
        for subcat in ['gad', 'phq', 'pss', 'adhd']:
            session.pop(f'answers_{subcat}', None)
            session.pop(f'times_{subcat}', None)
            session.pop(f'start_time_{subcat}', None)

        return render_template('result.html',
                               result=result,
                               personal=personal_info,
                               pdf_path=pdf_path,
                               csv_path=csv_path)

    else:
        return "❌ Invalid category submitted", 400

'''
@app.route('/submit/<category>', methods=['GET', 'POST'])
def submit_category(category):
    personal_info = session.get('personal', {})
    result = {}

    # === HANDLE INDIVIDUAL BLOCKS ===
    if category in ['gad', 'phq', 'pss', 'adhd']:
        answers = session.get(f'answers_{category}', [])
        times = session.get(f'times_{category}', [])
        total_time = round(time.time() - session.get(f'start_time_{category}', time.time()), 2)

        # Check for unanswered
        if None in answers or len(answers) == 0:
            return "⚠️ You have unanswered questions. Please complete all questions before submitting.", 400

        # Scoring logic
        if category == 'gad':
            score = sum(answers)
            level = "Minimal" if score <= 4 else "Mild" if score <= 9 else "Moderate" if score <= 14 else "Severe"
            result = {'GAD-7 Score': score, 'Anxiety Level': level}
        elif category == 'phq':
            score = sum(answers)
            level = "Minimal" if score <= 4 else "Mild" if score <= 9 else "Moderate" if score <= 14 else "Moderately Severe" if score <= 19 else "Severe"
            result = {'PHQ-9 Score': score, 'Depression Level': level}
        elif category == 'pss':
            score = 0
            for i, (_, reverse) in enumerate(pss_questions):
                val = answers[i]
                if reverse:
                    val = 4 - val
                score += val
            level = "Low" if score <= 13 else "Moderate" if score <= 26 else "High"
            result = {'PSS Score': score, 'Stress Level': level}
        elif category == 'adhd':
            count = sum(1 for val in answers if val >= 4)
            result = {'ADHD Status': "Suggestive of ADHD" if count >= 4 else "Not Suggestive"}

        result['Total Time (seconds)'] = total_time

        # Save to CSV and PDF
        filename_base = personal_info['name'].replace(' ', '_') + f"_{category}"
        csv_path = f"reports/{filename_base}.csv"
        pdf_path = f"reports/{filename_base}.pdf"
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([personal_info['name'], personal_info['age'], personal_info['email'], personal_info['gender']] + list(result.values()))
        generate_pdf(personal_info, result, pdf_path)

        # Cleanup session
        session.pop(f'answers_{category}', None)
        session.pop(f'times_{category}', None)
        session.pop(f'start_time_{category}', None)

        return render_template('result.html', result=result, personal=personal_info, pdf_path=pdf_path, csv_path=csv_path)

    # === HANDLE "all" CATEGORY ===
    elif category == 'all':
        total_time = 0
        result = {}
        personal_info = session.get('personal', {})

        # === Step 1: Collect Questionnaire Scores
        for subcat, label in zip(['gad', 'phq', 'pss', 'adhd'], ['GAD-7', 'PHQ-9', 'PSS', 'ADHD']):
            sub_answers = session.get(f'answers_{subcat}', [])
            sub_times = session.get(f'times_{subcat}', [])
            sub_total_time = round(time.time() - session.get(f'start_time_{subcat}', time.time()), 2)
            total_time += sub_total_time

            if len(sub_answers) == 0 or any(a is None for a in sub_answers):
                return f"⚠️ You have unanswered questions in {label}.", 400

            if subcat == 'gad':
                score = sum(sub_answers)
                level = "Minimal" if score <= 4 else "Mild" if score <= 9 else "Moderate" if score <= 14 else "Severe"
                result.update({'GAD-7 Score': score, 'Anxiety Level': level})

            elif subcat == 'phq':
                score = sum(sub_answers)
                level = "Minimal" if score <= 4 else "Mild" if score <= 9 else "Moderate" if score <= 14 else "Moderately Severe" if score <= 19 else "Severe"
                result.update({'PHQ-9 Score': score, 'Depression Level': level})

            elif subcat == 'pss':
                score = 0
                for i, (_, reverse) in enumerate(pss_questions):
                    val = sub_answers[i]
                    if reverse:
                        val = 4 - val
                    score += val
                level = "Low" if score <= 13 else "Moderate" if score <= 26 else "High"
                result.update({'PSS Score': score, 'Stress Level': level})

            elif subcat == 'adhd':
                count = sum(1 for val in sub_answers if val >= 4)
                result.update({'ADHD Status': "Suggestive of ADHD" if count >= 4 else "Not Suggestive"})

        result['Total Time (seconds)'] = total_time

        # === Step 2: Stop Behavioral UI (optional)
        stop_behavioral_background()

        # === Step 3: Stop and Process EEG
        try:
            from eeg_utils import stop_and_process_eeg
            eeg_result = stop_and_process_eeg(personal_info)
            result.update({
                'EEG Rule-Based': eeg_result.get('rule_based', 'Unavailable'),
                'EEG ML-Based': str(eeg_result.get('ml_summary', 'Unavailable'))
            })
        except Exception as e:
            result.update({'EEG': f"Error: {str(e)}"})

        # === Step 4: Finalize Behavioral CSV and Report
        try:
            from behavioral import finalize_behavioral_report
            from collections import Counter

            behavioral_result = finalize_behavioral_report(personal_info)

            label_series = behavioral_result.get('Predicted_Label')

            if isinstance(label_series, list):
                label_summary = dict(Counter(label_series))
            elif isinstance(label_series, str):
                label_summary = {label_series: 1}
            else:
                label_summary = {'Unavailable': 0}

            result.update({
               # 'Behavioral Emotion': behavioral_result.get('emotion', 'Unavailable'),
                'Behavioral': label_summary
            })
        except Exception as e:
            result.update({'Behavioral': f"Error: {str(e)}"})

        # === Step 5: Save Combined Report
        os.makedirs("reports", exist_ok=True)
        filename_base = personal_info['name'].replace(" ", "_") + "_all"
        csv_path = f"reports/{filename_base}.csv"
        pdf_path = f"reports/{filename_base}.pdf"

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                personal_info.get('name', ''),
                personal_info.get('age', ''),
                personal_info.get('email', ''),
                personal_info.get('gender', '')
            ] + list(result.values()))

        generate_pdf(personal_info, result, pdf_path)

        # === Step 6: Clear session tracking data
        for subcat in ['gad', 'phq', 'pss', 'adhd']:
            session.pop(f'answers_{subcat}', None)
            session.pop(f'times_{subcat}', None)
            session.pop(f'start_time_{subcat}', None)

        return render_template('result.html',
                               result=result,
                               personal=personal_info,
                               pdf_path=pdf_path,
                               csv_path=csv_path)

    '''
    elif category == 'all':
        total_time = 0
        result = {}

        for subcat, label in zip(['gad', 'phq', 'pss', 'adhd'], ['GAD-7', 'PHQ-9', 'PSS', 'ADHD']):
            sub_answers = session.get(f'answers_{subcat}', [])
            sub_times = session.get(f'times_{subcat}', [])
            sub_total_time = round(time.time() - session.get(f'start_time_{subcat}', time.time()), 2)
            total_time += sub_total_time

            if len(sub_answers) == 0 or any(a is None for a in sub_answers):
                return f"⚠️ You have unanswered questions in {label}.", 400

            if subcat == 'gad':
                score = sum(sub_answers)
                level = "Minimal" if score <= 4 else "Mild" if score <= 9 else "Moderate" if score <= 14 else "Severe"
                result.update({'GAD-7 Score': score, 'Anxiety Level': level})
            elif subcat == 'phq':
                score = sum(sub_answers)
                level = "Minimal" if score <= 4 else "Mild" if score <= 9 else "Moderate" if score <= 14 else "Moderately Severe" if score <= 19 else "Severe"
                result.update({'PHQ-9 Score': score, 'Depression Level': level})
            elif subcat == 'pss':
                score = 0
                for i, (_, reverse) in enumerate(pss_questions):
                    val = sub_answers[i]
                    if reverse:
                        val = 4 - val
                    score += val
                level = "Low" if score <= 13 else "Moderate" if score <= 26 else "High"
                result.update({'PSS Score': score, 'Stress Level': level})
            elif subcat == 'adhd':
                count = sum(1 for val in sub_answers if val >= 4)
                result.update({'ADHD Status': "Suggestive of ADHD" if count >= 4 else "Not Suggestive"})

        result['Total Time (seconds)'] = total_time

        # === Stop behavioral frontend
        stop_behavioral_background()

        # === Stop & save EEG report
        try:
            from eeg_utils import stop_and_process_eeg
            eeg_result = stop_and_process_eeg(personal_info)
            result.update({
                'EEG Rule-Based': eeg_result.get('rule_based', 'Unavailable'),
                'EEG ML-Based': str(eeg_result.get('ml_summary', 'Unavailable'))
            })
        except Exception as e:
            result['EEG'] = f"Error: {str(e)}"

        # === Save behavioral results
        try:
            from behavioral import finalize_behavioral_report
            behavioral_result = finalize_behavioral_report(personal_info)
            result.update({
                'Behavioral Emotion': behavioral_result.get('emotion', 'Unavailable'),
                'Behavioral Label': behavioral_result.get('Predicted_Label', 'Unavailable')
            })
        except Exception as e:
            result['Behavioral'] = f"Error: {str(e)}"

        # === Save all results
        filename_base = personal_info['name'].replace(" ", "_") + "_all"
        csv_path = f"reports/{filename_base}.csv"
        pdf_path = f"reports/{filename_base}.pdf"
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                personal_info['name'], personal_info['age'], personal_info['email'], personal_info['gender']
            ] + list(result.values()))
        generate_pdf(personal_info, result, pdf_path)

        for subcat in ['gad', 'phq', 'pss', 'adhd']:
            session.pop(f'answers_{subcat}', None)
            session.pop(f'times_{subcat}', None)
            session.pop(f'start_time_{subcat}', None)

        return render_template('result.html', result=result, personal=personal_info, pdf_path=pdf_path, csv_path=csv_path)

    else:
        return "❌ Invalid category submitted", 400
'''
'''
@app.route('/submit/<category>', methods=['GET', 'POST'])
def submit_category(category):
    personal_info = session.get('personal', {})
    result = {}

    # === HANDLE INDIVIDUAL BLOCKS ===
    if category in ['gad', 'phq', 'pss', 'adhd']:
        answers = session.get(f'answers_{category}', [])
        times = session.get(f'times_{category}', [])
        total_time = round(time.time() - session.get(f'start_time_{category}', time.time()), 2)

        # Check if any question is unanswered
        if None in answers or len(answers) == 0:
            return "⚠️ You have unanswered questions. Please complete all questions before submitting.", 400

        if category == 'gad':
            score = sum(answers)
            level = "Minimal" if score <= 4 else "Mild" if score <= 9 else "Moderate" if score <= 14 else "Severe"
            result = {'GAD-7 Score': score, 'Anxiety Level': level}
        elif category == 'phq':
            score = sum(answers)
            level = "Minimal" if score <= 4 else "Mild" if score <= 9 else "Moderate" if score <= 14 else "Moderately Severe" if score <= 19 else "Severe"
            result = {'PHQ-9 Score': score, 'Depression Level': level}
        elif category == 'pss':
            score = 0
            for i, (_, reverse) in enumerate(pss_questions):
                val = answers[i]
                if reverse:
                    val = 4 - val
                score += val
            level = "Low" if score <= 13 else "Moderate" if score <= 26 else "High"
            result = {'PSS Score': score, 'Stress Level': level}
        elif category == 'adhd':
            count = sum(1 for val in answers if val >= 4)
            result = {'ADHD Status': "Suggestive of ADHD" if count >= 4 else "Not Suggestive"}

        result['Total Time (seconds)'] = total_time

        # Save results
        csv_path = f"reports/{personal_info['name'].replace(' ', '_')}_{category}.csv"
        pdf_path = f"reports/{personal_info['name'].replace(' ', '_')}_{category}.pdf"
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                personal_info['name'], personal_info['age'], personal_info['email'], personal_info['gender']
            ] + list(result.values()))
        generate_pdf(personal_info, result, pdf_path)

        session.pop(f'answers_{category}', None)
        session.pop(f'times_{category}', None)
        session.pop(f'start_time_{category}', None)

        return render_template('result.html',
                               result=result,
                               personal=personal_info,
                               pdf_path=pdf_path,
                               csv_path=csv_path)



    # === HANDLE "all" CATEGORY ===
    elif category == 'all':
    result = {}
    total_time = 0

    # === Questionnaire Results ===
    for subcat, label in zip(['gad', 'phq', 'pss', 'adhd'], ['GAD-7', 'PHQ-9', 'PSS', 'ADHD']):
        sub_answers = session.get(f'answers_{subcat}', [])
        sub_times = session.get(f'times_{subcat}', [])
        sub_total_time = round(time.time() - session.get(f'start_time_{subcat}', time.time()), 2)
        total_time += sub_total_time

        if len(sub_answers) == 0 or any(a is None for a in sub_answers):
            return f"⚠️ You have unanswered questions in {label}.", 400

        if subcat == 'gad':
            score = sum(sub_answers)
            level = "Minimal" if score <= 4 else "Mild" if score <= 9 else "Moderate" if score <= 14 else "Severe"
            result.update({'GAD-7 Score': score, 'Anxiety Level': level})
        elif subcat == 'phq':
            score = sum(sub_answers)
            level = "Minimal" if score <= 4 else "Mild" if score <= 9 else "Moderate" if score <= 14 else "Moderately Severe" if score <= 19 else "Severe"
            result.update({'PHQ-9 Score': score, 'Depression Level': level})
        elif subcat == 'pss':
            score = 0
            for i, (_, reverse) in enumerate(pss_questions):
                val = sub_answers[i]
                if reverse:
                    val = 4 - val
                score += val
            level = "Low" if score <= 13 else "Moderate" if score <= 26 else "High"
            result.update({'PSS Score': score, 'Stress Level': level})
        elif subcat == 'adhd':
            count = sum(1 for val in sub_answers if val >= 4)
            result.update({'ADHD Status': "Suggestive of ADHD" if count >= 4 else "Not Suggestive"})

    result['Total Time (seconds)'] = total_time

    # === Stop Behavioral Frontend
    stop_behavioral_background()

    # === Stop and Save EEG Report
    try:
        from eeg_utils import stop_and_process_eeg
        eeg_result = stop_and_process_eeg(personal_info)
        result.update({
            'EEG Rule-Based': eeg_result.get('rule_based', 'Unavailable'),
            'EEG ML-Based': str(eeg_result.get('ml_summary', 'Unavailable'))
        })
    except Exception as e:
        result['EEG'] = f"Error: {str(e)}"

    # === Stop and Save Behavioral Report
    try:
        from behavioral import finalize_behavioral_report
        behavioral_result = finalize_behavioral_report(personal_info)
        result.update({
            'Behavioral Emotion': behavioral_result.get('emotion', 'Unavailable'),
            'Behavioral Label': behavioral_result.get('Predicted_Label', 'Unavailable')
        })
    except Exception as e:
        result['Behavioral'] = f"Error: {str(e)}"

    # === Save CSV and PDF
    filename_base = personal_info['name'].replace(' ', '_')
    csv_path = f"reports/{filename_base}_all.csv"
    pdf_path = f"reports/{filename_base}_all.pdf"
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            personal_info['name'], personal_info['age'], personal_info['email'], personal_info['gender']
        ] + list(result.values()))
    generate_pdf(personal_info, result, pdf_path)

    # Clean session
    for subcat in ['gad', 'phq', 'pss', 'adhd']:
        session.pop(f'answers_{subcat}', None)
        session.pop(f'times_{subcat}', None)
        session.pop(f'start_time_{subcat}', None)

    return render_template('result.html',
                           result=result,
                           personal=personal_info,
                           pdf_path=pdf_path,
                           csv_path=csv_path)

    elif category == 'all':
        result = {}
        total_time = 0

        for subcat, label in zip(['gad', 'phq', 'pss', 'adhd'], ['GAD-7', 'PHQ-9', 'PSS', 'ADHD']):
            sub_answers = session.get(f'answers_{subcat}', [])
            sub_times = session.get(f'times_{subcat}', [])
            sub_total_time = round(time.time() - session.get(f'start_time_{subcat}', time.time()), 2)
            total_time += sub_total_time

            if len(sub_answers) == 0 or any(a is None for a in sub_answers):
                return f"⚠️ You have unanswered questions in {label}.", 400

            if subcat == 'gad':
                score = sum(sub_answers)
                level = "Minimal" if score <= 4 else "Mild" if score <= 9 else "Moderate" if score <= 14 else "Severe"
                result.update({'GAD-7 Score': score, 'Anxiety Level': level})
            elif subcat == 'phq':
                score = sum(sub_answers)
                level = "Minimal" if score <= 4 else "Mild" if score <= 9 else "Moderate" if score <= 14 else "Moderately Severe" if score <= 19 else "Severe"
                result.update({'PHQ-9 Score': score, 'Depression Level': level})
            elif subcat == 'pss':
                score = 0
                for i, (_, reverse) in enumerate(pss_questions):
                    val = sub_answers[i]
                    if reverse:
                        val = 4 - val
                    score += val
                level = "Low" if score <= 13 else "Moderate" if score <= 26 else "High"
                result.update({'PSS Score': score, 'Stress Level': level})
            elif subcat == 'adhd':
                count = sum(1 for val in sub_answers if val >= 4)
                result.update({'ADHD Status': "Suggestive of ADHD" if count >= 4 else "Not Suggestive"})

        result['Total Time (seconds)'] = total_time

        # Stop behavioral frontend session
        stop_behavioral_background()

        # Merge behavioral results
        try:
            from behavioral import run_behavioral_data_merge
            result.update(run_behavioral_data_merge())
        except Exception as e:
            result.update({'Behavioral': f"Error: {str(e)}"})

        # Merge EEG results
        try:
            from eeg_utils import run_eeg_merge
            result.update(run_eeg_merge())
        except Exception as e:
            result.update({'EEG': f"Error: {str(e)}"})

        # Save final CSV & PDF
        csv_path = f"reports/{personal_info['name'].replace(' ', '_')}_all.csv"
        pdf_path = f"reports/{personal_info['name'].replace(' ', '_')}_all.pdf"
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                personal_info['name'], personal_info['age'], personal_info['email'], personal_info['gender']
            ] + list(result.values()))
        generate_pdf(personal_info, result, pdf_path)

        for subcat in ['gad', 'phq', 'pss', 'adhd']:
            session.pop(f'answers_{subcat}', None)
            session.pop(f'times_{subcat}', None)
            session.pop(f'start_time_{subcat}', None)

        return render_template('result.html',
                               result=result,
                               personal=personal_info,
                               pdf_path=pdf_path,
                               csv_path=csv_path)

    else:
        return "❌ Invalid category submitted", 400
'''

from flask import jsonify

@app.route('/eeg_live_data')
def eeg_live_data():
    with eeg_collector["lock"]:
        if not eeg_collector["data"]:
            return jsonify({"status": "waiting"})
        latest = eeg_collector["data"][-1]

    eeg_array = np.array(latest)
    filtered = bandpass_filter(eeg_array, 1, 50, fs)

    delta, _ = compute_bandpower(filtered, fs, [0.5, 4])
    theta, _ = compute_bandpower(filtered, fs, [4, 8])
    alpha, _ = compute_bandpower(filtered, fs, [8, 13])
    beta, _ = compute_bandpower(filtered, fs, [13, 30])
    gamma, psd = compute_bandpower(filtered, fs, [30, 50])

    entropy = compute_entropy(psd)
    samp_entropy = sample_entropy(filtered)
    time_now = datetime.datetime.now().strftime('%H:%M:%S')

    return jsonify({
        "status": "ok",
        "timestamp": time_now,
        "waveform": filtered.tolist(),
        "bands": {
            "Alpha": alpha,
            "Beta": beta,
            "Theta": theta,
            "Delta": delta,
            "Gamma": gamma
        },
        "entropy": entropy,
        "samp_entropy": samp_entropy
    })

#from eeg_utils import start_eeg_collection, stop_and_process_eeg
from eeg_utils import (
    eeg_collector,               # ✅ Add this
    bandpass_filter,
    compute_bandpower,
    compute_entropy,
    sample_entropy,
    stop_and_process_eeg,
    start_eeg_collection
)


# Step 1: User clicks "Start EEG" from Dashboard
@app.route('/run_eeg')
def run_eeg():
    return render_template('eeg_start.html')

# Step 2: Start EEG data collection (triggered by button on eeg_start.html)
@app.route('/eeg_collect', methods=['POST'])
def eeg_collect():
    port = request.form.get('port', '/dev/ttyACM0')  # default fallback
    # Pass `port` to the start_eeg_collection function
    start_eeg_collection(port)
    return render_template('eeg_wait.html')

# Step 3: Stop EEG and show result
@app.route('/eeg_submit', methods=['POST'])
def eeg_submit():
    #personal_info = session.get('personal', {})
   # result = stop_and_process_eeg()  # add full-feature CSV logging inside this
    personal_info = session.get('personal', {'name': 'anonymous'})
    result = stop_and_process_eeg(personal_info)

    # Optional: generate waveform plot here and save as PNG
    # to show in eeg_result.html

    filename_base = personal_info['name'].replace(" ", "_") + "_eeg"
    csv_path = f"reports/{filename_base}.csv"
    pdf_path = f"reports/{filename_base}.pdf"

    # Save results CSV
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            personal_info['name'], personal_info['age'], personal_info['email'], personal_info['gender'],
            result['rule_based'], result['total_samples']
        ] + [f"{label}:{count}" for label, count in result['ml_summary'].items()])

    generate_pdf(personal_info, result, pdf_path)

    return render_template('eeg_result.html',
                           personal=personal_info,
                           result=result,
                           csv_path=csv_path,
                           pdf_path=pdf_path)

'''
@app.route('/eeg_start')
def eeg_start():
    start_eeg_collection()
    return render_template('eeg_wait.html')

@app.route('/eeg_submit', methods=['POST'])
def eeg_submit():
    personal_info = session.get('personal', {})
    result = stop_and_process_eeg()

    filename_base = personal_info['name'].replace(" ", "_") + "_eeg"
    csv_path = f"reports/{filename_base}.csv"
    pdf_path = f"reports/{filename_base}.pdf"

    # Save to CSV
    import csv
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            personal_info['name'], personal_info['age'], personal_info['email'], personal_info['gender'],
            result['rule_based'], result['total_samples']
        ] + [f"{label}:{count}" for label, count in result['ml_summary'].items()])

    # PDF
    from report_generator import generate_pdf
    generate_pdf(personal_info, result, pdf_path)

    return render_template('eeg_result.html',
                           personal=personal_info,
                           result=result,
                           csv_path=csv_path,
                           pdf_path=pdf_path)

'''
'''
@app.route('/submit/<category>', methods=['GET', 'POST'])
def submit_category(category):
    personal_info = session.get('personal', {})
    result = {}

    # === HANDLE INDIVIDUAL BLOCKS ===
    if category in ['gad', 'phq', 'pss', 'adhd']:
        answers = session.get(f'answers_{category}', [])
        times = session.get(f'times_{category}', [])
        total_time = round(time.time() - session.get(f'start_time_{category}', time.time()), 2)

        if None in answers:
            return "⚠️ You have unanswered questions. Please complete all questions before submitting.", 400

        if category == 'gad':
            score = sum(answers)
            level = "Minimal" if score <= 4 else "Mild" if score <= 9 else "Moderate" if score <= 14 else "Severe"
            result = {'GAD-7 Score': score, 'Anxiety Level': level}

        elif category == 'phq':
            score = sum(answers)
            level = "Minimal" if score <= 4 else "Mild" if score <= 9 else "Moderate" if score <= 14 else "Moderately Severe" if score <= 19 else "Severe"
            result = {'PHQ-9 Score': score, 'Depression Level': level}

        elif category == 'pss':
            score = 0
            for i, (_, reverse) in enumerate(pss_questions):
                val = answers[i]
                if reverse:
                    val = 4 - val
                score += val
            level = "Low" if score <= 13 else "Moderate" if score <= 26 else "High"
            result = {'PSS Score': score, 'Stress Level': level}

        elif category == 'adhd':
            count = sum(1 for val in answers if val >= 4)
            result = {'ADHD Status': "Suggestive of ADHD" if count >= 4 else "Not Suggestive"}

        result['Total Time (seconds)'] = total_time

        # Save results
        csv_path = f"reports/{personal_info['name'].replace(' ', '_')}_{category}.csv"
        pdf_path = f"reports/{personal_info['name'].replace(' ', '_')}_{category}.pdf"
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                personal_info['name'], personal_info['age'], personal_info['email'], personal_info['gender']
            ] + list(result.values()))
        generate_pdf(personal_info, result, pdf_path)

        # ✅ Now remove session only after successful save
        session.pop(f'answers_{category}', None)
        session.pop(f'times_{category}', None)
        session.pop(f'start_time_{category}', None)

        return render_template('result.html',
                               result=result,
                               personal=personal_info,
                               pdf_path=pdf_path,
                               csv_path=csv_path)

    # === HANDLE "all" CATEGORY ===
    elif category == 'all':
        total_time = 0
        for subcat, label in zip(['gad', 'phq', 'pss', 'adhd'], ['GAD-7', 'PHQ-9', 'PSS', 'ADHD']):
            sub_answers = session.get(f'answers_{subcat}', [])
            sub_times = session.get(f'times_{subcat}', [])
            sub_total_time = round(time.time() - session.get(f'start_time_{subcat}', time.time()), 2)
            total_time += sub_total_time

            if len(sub_answers) == 0 or any(a is None for a in sub_answers):
                return f"⚠️ You have unanswered questions in {label}.", 400

            if subcat == 'gad':
                score = sum(sub_answers)
                level = "Minimal" if score <= 4 else "Mild" if score <= 9 else "Moderate" if score <= 14 else "Severe"
                result.update({'GAD-7 Score': score, 'Anxiety Level': level})

            elif subcat == 'phq':
                score = sum(sub_answers)
                level = "Minimal" if score <= 4 else "Mild" if score <= 9 else "Moderate" if score <= 14 else "Moderately Severe" if score <= 19 else "Severe"
                result.update({'PHQ-9 Score': score, 'Depression Level': level})

            elif subcat == 'pss':
                score = 0
                for i, (_, reverse) in enumerate(pss_questions):
                    val = sub_answers[i]
                    if reverse:
                        val = 4 - val
                    score += val
                level = "Low" if score <= 13 else "Moderate" if score <= 26 else "High"
                result.update({'PSS Score': score, 'Stress Level': level})

            elif subcat == 'adhd':
                count = sum(1 for val in sub_answers if val >= 4)
                result.update({'ADHD Status': "Suggestive of ADHD" if count >= 4 else "Not Suggestive"})

        result['Total Time (seconds)'] = total_time

        # Save results
        csv_path = f"reports/{personal_info['name'].replace(' ', '_')}_all.csv"
        pdf_path = f"reports/{personal_info['name'].replace(' ', '_')}_all.pdf"
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                personal_info['name'], personal_info['age'], personal_info['email'], personal_info['gender']
            ] + list(result.values()))
        generate_pdf(personal_info, result, pdf_path)

        # ✅ Clear session data after success
        for subcat in ['gad', 'phq', 'pss', 'adhd']:
            session.pop(f'answers_{subcat}', None)
            session.pop(f'times_{subcat}', None)
            session.pop(f'start_time_{subcat}', None)

        return render_template('result.html',
                               result=result,
                               personal=personal_info,
                               pdf_path=pdf_path,
                               csv_path=csv_path)

    else:
        return "❌ Invalid category submitted", 400
'''

@app.route('/log_behavior_data', methods=['POST'])
def log_behavior_data():
    try:
        data = request.json
        emotion = data.get("emotion", "unknown")
        blink_rate = data.get("blink_rate", 0)
        smile_ratio = data.get("smile_ratio", 0)
        brow_furrow_score = data.get("brow_furrow_score", 0)

        # You can log or save this in session, CSV, etc.
        print(f"📷 Emotion: {emotion}, Blink Rate: {blink_rate}, Smile: {smile_ratio}, Brow: {brow_furrow_score}")

        # Optional: store in session or buffer
        if 'behavioral_log' not in session:
            session['behavioral_log'] = []
        session['behavioral_log'].append({
            'emotion': emotion,
            'blink_rate': blink_rate,
            'smile_ratio': smile_ratio,
            'brow_furrow_score': brow_furrow_score,
            'timestamp': time.time()
        })

        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/download/<path:filename>')
def download(filename):
    return send_file(filename, as_attachment=True)
@app.route('/reset')
def reset():
    session.clear()

    # Also ensure EEG/Behavioral background cleanup if needed
    try:
        stop_behavioral_background()
    except:
        pass
    try:
        from eeg_utils import stop_eeg_collection
        stop_eeg_collection()
    except:
        pass

    return redirect(url_for('personal_info'))

'''
@app.route('/reset', methods=['GET', 'POST'])
def reset():
    if request.method == 'POST':
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        session['user_info'] = {'name': name, 'age': age, 'gender': gender}

        # Clear previous data
        try:
            shutil.rmtree('reports')  # or clear individual folders
            os.makedirs('reports', exist_ok=True)
        except Exception as e:
            print("Reset error:", e)

        # Reset any additional global/session variables
        session['eeg_data'] = []
        session['behavioral_data'] = []
        session['questionnaire_answers'] = []

        return redirect('/dashboard')  # Back to clean dashboard

    return render_template('personal_info.html')
'''
if __name__ == "__main__":
    print("✅ Starting Flask server at http://127.0.0.1:5000")
    app.run(debug=False, use_reloader=False)


