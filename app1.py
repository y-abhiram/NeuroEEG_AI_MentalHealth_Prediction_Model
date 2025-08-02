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


