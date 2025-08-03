import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU for TensorFlow

import matplotlib
matplotlib.use('Agg')  # Prevent GUI/OpenGL-related issues

import cv2
import numpy as np
import pandas as pd
import threading
import time
from datetime import datetime
from collections import deque, Counter
import joblib
from tensorflow.keras.models import load_model
import mediapipe as mp

# === Load Models ===
cnn_model = load_model("models/cnn_feature_extractor.h5")
rf_model = joblib.load("models/cnn_rf_model.pkl")
scaler = joblib.load("models/cnn_rf_scaler.pkl")
label_encoder = joblib.load("models/cnn_rf_label_encoder.pkl")
emotion_model = load_model('models/emotion_model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def run_live_behavioral(personal_info):
    import cv2
    import time
    import pandas as pd
    import numpy as np
    from collections import deque, Counter
    from datetime import datetime

    frame = None
    lock = threading.Lock()
    running = True
    collected_data = []
    start_time = time.time()
    last_sample_time = time.time()

    latest_emotion = "Neutral"
    latest_confidence = 0.0
    avg_ear = 0.0
    blink_count = 0
    frame_count = 0
    smile_ratio = 0.0
    brow_furrow_score = 0.0
    fatigue_score = 0.0
    movement_code = 1
    tilt_code = 0
    previous_positions = deque(maxlen=10)

    def eye_aspect_ratio(landmarks, indices):
        points = [landmarks[i] for i in indices]
        vertical1 = np.linalg.norm(points[1] - points[5])
        vertical2 = np.linalg.norm(points[2] - points[4])
        horizontal = np.linalg.norm(points[0] - points[3])
        return (vertical1 + vertical2) / (2.0 * horizontal)

    def detect_emotion():
        nonlocal latest_emotion, latest_confidence, running, frame
        try:
            while running:
                with lock:
                    if frame is None:
                        continue
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    face = gray[y:y + h, x:x + w]
                    face = cv2.resize(face, (48, 48))
                    face = np.expand_dims(np.expand_dims(face, -1), 0) / 255.0
                    prediction = emotion_model.predict(face, verbose=0)
                    idx = np.argmax(prediction)
                    with lock:
                        latest_emotion = emotion_labels[idx]
                        latest_confidence = prediction[0][idx] * 100
                time.sleep(0.01)
        except Exception as e:
            print("Emotion Thread Error:", e)

    def detect_features():
        nonlocal avg_ear, blink_count, frame_count, smile_ratio, brow_furrow_score, fatigue_score
        nonlocal tilt_code, movement_code, previous_positions, start_time, running, frame
        try:
            while running:
                with lock:
                    if frame is None:
                        continue
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w = frame.shape[:2]

                results = face_mesh.process(img)
                if results.multi_face_landmarks:
                    mesh = results.multi_face_landmarks[0].landmark
                    landmarks = np.array([[int(p.x * w), int(p.y * h)] for p in mesh])
                    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
                    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
                    ear = (left_ear + right_ear) / 2.0

                    nose = landmarks[1]
                    previous_positions.append(nose)
                    if len(previous_positions) >= 2:
                        dx = previous_positions[-1][0] - previous_positions[0][0]
                        dy = previous_positions[-1][1] - previous_positions[0][1]
                        movement = np.sqrt(dx ** 2 + dy ** 2)
                        movement_code = 1 if movement > 5 else 0

                    left_eye_top = mesh[159].y * h
                    right_eye_top = mesh[386].y * h
                    diff_y = left_eye_top - right_eye_top
                    tilt_code = 0 if abs(diff_y) < 5 else (1 if diff_y > 0 else 2)

                    with lock:
                        avg_ear = ear
                        if ear < 0.23:
                            frame_count += 1
                        else:
                            if 2 < frame_count < 20:
                                blink_count += 1
                            frame_count = 0

                        mouth_left, mouth_right = landmarks[61], landmarks[291]
                        mouth_top, mouth_bottom = landmarks[13], landmarks[14]
                        smile_ratio = np.linalg.norm(mouth_left - mouth_right) / (np.linalg.norm(mouth_top - mouth_bottom) + 1e-5)
                        brow_furrow_score = np.linalg.norm(landmarks[70] - landmarks[300])
                        fatigue_score = (6 * blink_count / max(1, (time.time() - start_time) / 60)) * 0.2 + \
                                        (1 - min(avg_ear / 0.4, 1.0)) * 0.3 + \
                                        (1 if brow_furrow_score < 35 else 0) * 0.2
                time.sleep(0.01)
        except Exception as e:
            print("Feature Thread Error:", e)

    # === Retry camera if first attempt fails ===
    cap = None
    for attempt in range(2):
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Webcam opened successfully")
            break
        else:
            print(f"⚠️ Webcam not opened (attempt {attempt + 1}). Retrying...")
            cap.release()
            time.sleep(1)
    else:
        print("❌ Camera could not be opened. Skipping behavioral.")
        return {"error": "Camera not available"}, None

    threading.Thread(target=detect_emotion, daemon=True).start()
    threading.Thread(target=detect_features, daemon=True).start()

    print("📷 Starting webcam. Press 'q' to quit.")

    try:
        while cap.isOpened():
            ret, new_frame = cap.read()
            if not ret:
                print("⚠️ Frame not captured. Ending capture loop.")
                break

            with lock:
                frame = new_frame.copy()

            current_time = time.time()
            if current_time - last_sample_time >= 10:
                last_sample_time = current_time
                with lock:
                    row = [avg_ear, 6 * blink_count, smile_ratio, brow_furrow_score,
                           movement_code, tilt_code, latest_emotion, fatigue_score]
                    collected_data.append(row)
                    blink_count = 0

            with lock:
                cv2.putText(frame, f"Samples: {len(collected_data)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(frame, f"Emotion: {latest_emotion} ({latest_confidence:.1f}%)", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
                elapsed_minutes = max(1, (time.time() - start_time) / 60)
                blink_rate_per_min = 6 * blink_count / elapsed_minutes
                cv2.putText(frame, f"Blink Rate: {blink_rate_per_min:.2f} BPM", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.imshow("Real-Time Stress Capture", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 Quit key pressed. Exiting webcam loop.")
                break

    finally:
        print("🔚 Cleaning up camera resources")
        running = False
        if cap:
            cap.release()
        cv2.destroyAllWindows()

    # === Save behavioral report ===
    if len(collected_data) == 0:
        print("❌ No behavioral data collected.")
        return {"error": "No samples captured"}, None

    columns = ['ear', 'blink_rate', 'smile_ratio', 'brow_furrow_score',
               'movement_code', 'tilt_code', 'emotion', 'fatigue_score']
    df_live = pd.DataFrame(collected_data, columns=columns)

    if df_live['emotion'].dtype == object:
        df_live['emotion'] = df_live['emotion'].astype('category').cat.codes

    X_scaled = scaler.transform(df_live)
    X_cnn_input = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    cnn_features = cnn_model.predict(X_cnn_input, verbose=0)
    predictions = rf_model.predict(cnn_features)
    predicted_labels = label_encoder.inverse_transform(predictions)

    df_live['Predicted_Label'] = predicted_labels

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("reports", exist_ok=True)
    report_name = f"reports/{personal_info.get('name', 'user').replace(' ', '_')}_behavioral_{timestamp}.csv"
    df_live.to_csv(report_name, index=False)

    summary = dict(Counter(predicted_labels))
    summary['Most Frequent State'] = max(summary, key=summary.get)
    summary['Total Samples'] = len(df_live)

    return summary, report_name


'''
def run_live_behavioral(personal_info):
    frame = None
    lock = threading.Lock()
    running = True
    collected_data = []
    start_time = time.time()
    last_sample_time = time.time()

    latest_emotion = "Neutral"
    latest_confidence = 0.0
    avg_ear = 0.0
    blink_count = 0
    frame_count = 0
    smile_ratio = 0.0
    brow_furrow_score = 0.0
    fatigue_score = 0.0
    movement_code = 1
    tilt_code = 0
    previous_positions = deque(maxlen=10)

    def eye_aspect_ratio(landmarks, indices):
        points = [landmarks[i] for i in indices]
        vertical1 = np.linalg.norm(points[1] - points[5])
        vertical2 = np.linalg.norm(points[2] - points[4])
        horizontal = np.linalg.norm(points[0] - points[3])
        return (vertical1 + vertical2) / (2.0 * horizontal)

    def detect_emotion():
        nonlocal latest_emotion, latest_confidence, running, frame
        try:
            while running:
                with lock:
                    if frame is None:
                        continue
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    face = gray[y:y + h, x:x + w]
                    face = cv2.resize(face, (48, 48))
                    face = np.expand_dims(np.expand_dims(face, -1), 0) / 255.0
                    prediction = emotion_model.predict(face, verbose=0)
                    idx = np.argmax(prediction)
                    with lock:
                        latest_emotion = emotion_labels[idx]
                        latest_confidence = prediction[0][idx] * 100

                time.sleep(0.01)
        except Exception as e:
            print("Emotion Thread Error:", e)

    def detect_features():
        nonlocal avg_ear, blink_count, frame_count, smile_ratio, brow_furrow_score, fatigue_score
        nonlocal tilt_code, movement_code, previous_positions, start_time, running, frame
        try:
            while running:
                with lock:
                    if frame is None:
                        continue
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w = frame.shape[:2]

                results = face_mesh.process(img)
                if results.multi_face_landmarks:
                    mesh = results.multi_face_landmarks[0].landmark
                    landmarks = np.array([[int(p.x * w), int(p.y * h)] for p in mesh])
                    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
                    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
                    ear = (left_ear + right_ear) / 2.0

                    nose = landmarks[1]
                    previous_positions.append(nose)
                    if len(previous_positions) >= 2:
                        dx = previous_positions[-1][0] - previous_positions[0][0]
                        dy = previous_positions[-1][1] - previous_positions[0][1]
                        movement = np.sqrt(dx ** 2 + dy ** 2)
                        movement_code = 1 if movement > 5 else 0

                    left_eye_top = mesh[159].y * h
                    right_eye_top = mesh[386].y * h
                    diff_y = left_eye_top - right_eye_top
                    tilt_code = 0 if abs(diff_y) < 5 else (1 if diff_y > 0 else 2)

                    with lock:
                        avg_ear = ear
                        if ear < 0.23:
                            frame_count += 1
                        else:
                            if 2 < frame_count < 20:
                                blink_count += 1
                            frame_count = 0

                        mouth_left, mouth_right = landmarks[61], landmarks[291]
                        mouth_top, mouth_bottom = landmarks[13], landmarks[14]
                        smile_ratio = np.linalg.norm(mouth_left - mouth_right) / (np.linalg.norm(mouth_top - mouth_bottom) + 1e-5)
                        brow_furrow_score = np.linalg.norm(landmarks[70] - landmarks[300])
                        fatigue_score = (6 * blink_count / max(1, (time.time() - start_time) / 60)) * 0.2 + \
                                        (1 - min(avg_ear / 0.4, 1.0)) * 0.3 + \
                                        (1 if brow_furrow_score < 35 else 0) * 0.2
                time.sleep(0.01)
        except Exception as e:
            print("Feature Thread Error:", e)

    threading.Thread(target=detect_emotion, daemon=True).start()
    threading.Thread(target=detect_features, daemon=True).start()

    cap = cv2.VideoCapture(0)
    print("📷 Starting webcam. Press 'q' to quit.")

    try:
        while cap.isOpened():
            ret, new_frame = cap.read()
            if not ret:
                break

            with lock:
                frame = new_frame.copy()

            current_time = time.time()
            if current_time - last_sample_time >= 10:
                last_sample_time = current_time
                with lock:
                    emotion_code = latest_emotion
                    row = [avg_ear, 6 * blink_count, smile_ratio, brow_furrow_score,
                           movement_code, tilt_code, emotion_code, fatigue_score]
                    collected_data.append(row)
                    blink_count = 0

            with lock:
                cv2.putText(frame, f"Samples: {len(collected_data)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(frame, f"Emotion: {latest_emotion} ({latest_confidence:.1f}%)", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
                elapsed_minutes = max(1, (time.time() - start_time) / 60)
                blink_rate_per_min = 6 * blink_count / elapsed_minutes
                cv2.putText(frame, f"Blink Rate: {blink_rate_per_min:.2f} BPM", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.imshow("Real-Time Stress Capture", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        running = False

    columns = ['ear', 'blink_rate', 'smile_ratio', 'brow_furrow_score',
               'movement_code', 'tilt_code', 'emotion', 'fatigue_score']
    df_live = pd.DataFrame(collected_data, columns=columns)

    if df_live['emotion'].dtype == object:
        df_live['emotion'] = df_live['emotion'].astype('category').cat.codes

    X_scaled = scaler.transform(df_live)
    X_cnn_input = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    cnn_features = cnn_model.predict(X_cnn_input, verbose=0)
    predictions = rf_model.predict(cnn_features)
    predicted_labels = label_encoder.inverse_transform(predictions)

    df_live['Predicted_Label'] = predicted_labels

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("reports", exist_ok=True)
    report_name = f"reports/{personal_info.get('name', 'user').replace(' ', '_')}_behavioral_{timestamp}.csv"
    df_live.to_csv(report_name, index=False)

    summary = dict(Counter(predicted_labels))
    summary['Most Frequent State'] = max(summary, key=summary.get)
    summary['Total Samples'] = len(df_live)

    return summary, report_name
'''
    
import os
import csv
from flask import session


def run_behavioral_data_merge():
    import os, csv
    path = os.path.join("reports", "anonymous_behavioral.csv")  # Or use name from session
    if not os.path.exists(path):
        return {"Behavioral Emotion": "Unavailable", "Behavioral Label": "Unavailable"}

    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        row = next(reader)
        return {
            "Behavioral Emotion": row.get("emotion", "N/A"),
            "Behavioral Label": row.get("Predicted_Label", "N/A")
        }


import os
import pandas as pd

def finalize_behavioral_report(personal_info):
    name = personal_info.get('name', 'user').replace(" ", "_")
    folder = "reports"
    matching_files = [f for f in os.listdir(folder) if f.startswith(f"{name}_behavioral") and f.endswith(".csv")]

    if not matching_files:
        raise FileNotFoundError("No behavioral CSV found.")

    # Get the latest by timestamp
    latest_file = max(matching_files, key=lambda f: os.path.getctime(os.path.join(folder, f)))
    file_path = os.path.join(folder, latest_file)

    df = pd.read_csv(file_path)

    # Handle column checks gracefully
    emotion = df['emotion'].mode()[0] if 'emotion' in df.columns and not df['emotion'].empty else 'Neutral'

    return {
       # "emotion": emotion,
        "Predicted_Label": df['Predicted_Label'].tolist(),  # ✅ Fix applied here
        "source_file": latest_file
    }

'''
def finalize_behavioral_report(personal_info):
    import pandas as pd
    import os
    from report_generator import generate_pdf

    name = personal_info.get('name', 'anonymous').replace(" ", "_")
    path = f"reports/{name}_behavioral.csv"
    if not os.path.exists(path):
        raise FileNotFoundError("No behavioral CSV found.")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("No data found in behavioral report.")

    latest = df.iloc[-1]
    emotion = latest.get("emotion", "Unavailable")
    label = latest.get("Predicted_Label", "Unavailable")

    pdf_path = f"reports/{name}_behavioral.pdf"
    generate_pdf(personal_info, {"Emotion": emotion, "Predicted_Label": label}, pdf_path)

    return {"emotion": emotion, "Predicted_Label": label}

'''
'''
def run_behavioral_data_merge():
    """
    Extracts emotion and final behavioral predicted label from the behavioral CSV.
    """
    personal = session.get("personal", {"name": "anonymous"})
    name = personal['name'].replace(" ", "_")
    filepath = f"reports/{name}_behavioral.csv"

    if not os.path.exists(filepath):
        return {
            "Behavioral Emotion": "Unavailable",
            "Behavioral Label": "Unavailable"
        }

    with open(filepath, 'r') as f:
        reader = list(csv.reader(f))
        if len(reader) <= 1:
            return {
                "Behavioral Emotion": "No Data",
                "Behavioral Label": "No Data"
            }

        header = reader[0]
        last_row = reader[-1]

        try:
            emotion_idx = header.index("emotion")
            label_idx = header.index("Predicted_Label")
            emotion = last_row[emotion_idx]
            label = last_row[label_idx]
        except Exception as e:
            emotion = f"Error: {e}"
            label = f"Error: {e}"

    return {
        "Behavioral Emotion": emotion,
        "Behavioral Label": label
    }
'''
