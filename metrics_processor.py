import cv2
import numpy as np
import pandas as pd
import time
from datetime import datetime
from collections import Counter
import joblib
from tensorflow.keras.models import load_model
import mediapipe as mp
from report_generator import generate_pdf

# === Load Models ===
cnn_model = load_model("/home/abhiram1289/Desktop/IIT TIRUPATHI/eegstress/cnn_feature_extractor.h5")
rf_model = joblib.load("/home/abhiram1289/Desktop/IIT TIRUPATHI/eegstress/cnn_rf_model.pkl")
scaler = joblib.load("/home/abhiram1289/Desktop/IIT TIRUPATHI/eegstress/cnn_rf_scaler.pkl")
label_encoder = joblib.load("/home/abhiram1289/Desktop/IIT TIRUPATHI/eegstress/cnn_rf_label_encoder.pkl")
emotion_model = load_model('/home/abhiram1289/emotion_model.h5')

# === Setup ===
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# === State ===
blink_count = 0
frame_count = 0
start_time = time.time()
last_sample_time = time.time()

# === EAR Helper ===
def eye_aspect_ratio(landmarks, indices):
    points = [landmarks[i] for i in indices]
    vertical1 = np.linalg.norm(points[1] - points[5])
    vertical2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])
    return (vertical1 + vertical2) / (2.0 * horizontal)

def process_frame(frame):
    global blink_count, frame_count, start_time, last_sample_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    # Emotion detection
    latest_emotion = "Neutral"
    latest_confidence = 0.0
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w_face, h_face) in faces:
        face_img = gray[y:y + h_face, x:x + w_face]
        face_img = cv2.resize(face_img, (48, 48))
        face_img = np.expand_dims(np.expand_dims(face_img, -1), 0) / 255.0
        prediction = emotion_model.predict(face_img, verbose=0)
        idx = np.argmax(prediction)
        latest_emotion = emotion_labels[idx]
        latest_confidence = prediction[0][idx] * 100
        break

    # Defaults
    avg_ear = 0.3
    smile_ratio = 0.0
    brow_furrow_score = 0.0
    fatigue_score = 0.0
    movement_code = 1
    tilt_code = 0

    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        mesh = results.multi_face_landmarks[0].landmark
        landmarks = np.array([[int(p.x * w), int(p.y * h)] for p in mesh])
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0

        avg_ear = ear
        if ear < 0.23:
            frame_count += 1
        else:
            if 2 < frame_count < 20:
                blink_count += 1
            frame_count = 0

        mouth_left, mouth_right = landmarks[61], landmarks[291]
        mouth_top, mouth_bottom = landmarks[13], landmarks[14]
        vertical_dist = np.linalg.norm(mouth_top - mouth_bottom)
        if vertical_dist < 1e-2:
            smile_ratio = 0.0
        else:
            smile_ratio = np.linalg.norm(mouth_left - mouth_right) / vertical_dist

        brow_furrow_score = np.linalg.norm(landmarks[70] - landmarks[300])

        fatigue_score = (6 * blink_count / max(1, (time.time() - start_time) / 60)) * 0.2 \
                        + (1 - min(avg_ear / 0.4, 1.0)) * 0.3 \
                        + (1 if brow_furrow_score < 35 else 0) * 0.2

    sample_now = False
    current_time = time.time()
    if current_time - last_sample_time >= 10:
        sample_now = True
        last_sample_time = current_time
        blink_count = 0
        start_time = current_time

    metrics = {
        'ear': round(avg_ear, 3),
        'blink_rate': 6 * blink_count,
        'smile_ratio': round(smile_ratio, 3),
        'brow_furrow_score': round(brow_furrow_score, 2),
        'movement_code': movement_code,
        'tilt_code': tilt_code,
        'emotion': latest_emotion,
        'fatigue_score': round(fatigue_score, 2),
        'sample_ready': sample_now
    }

    return metrics, frame

def finalize_results(metrics_buffer, personal_info):
    if not metrics_buffer:
        print("[Metrics] No metrics collected.")
        return None, None, {'error': 'No metrics collected'}
    df = pd.DataFrame(metrics_buffer)
    if df['emotion'].dtype == object:
        df['emotion'] = df['emotion'].astype('category').cat.codes

    X_scaled = scaler.transform(df)
    X_cnn_input = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    cnn_features = cnn_model.predict(X_cnn_input, verbose=0)
    predictions = rf_model.predict(cnn_features)
    predicted_labels = label_encoder.inverse_transform(predictions)
    df['Predicted_Label'] = predicted_labels

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"reports/{personal_info.get('name', 'user').replace(' ', '_')}_behavioral_{timestamp}"
    csv_path = base + ".csv"
    pdf_path = base + ".pdf"

    df.to_csv(csv_path, index=False)
    summary = dict(Counter(predicted_labels))
    summary['Most Frequent State'] = max(summary, key=summary.get)
    summary['Total Samples'] = len(df)

    generate_pdf(personal_info, summary, pdf_path)
    return csv_path, pdf_path, summary
'''


import cv2
import numpy as np
import pandas as pd
import time
from datetime import datetime
from collections import Counter
import joblib
from tensorflow.keras.models import load_model
import mediapipe as mp
from report_generator import generate_pdf

# === Load Models ===
cnn_model = load_model("models/cnn_feature_extractor.h5")
rf_model = joblib.load("models/cnn_rf_model.pkl")
scaler = joblib.load("models/cnn_rf_scaler.pkl")
label_encoder = joblib.load("models/cnn_rf_label_encoder.pkl")
emotion_model = load_model('models/emotion_model.h5')

# === Setup ===
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# === State ===
blink_count = 0
frame_count = 0
start_time = time.time()
last_sample_time = time.time()

# === EAR Helper ===
def eye_aspect_ratio(landmarks, indices):
    points = [landmarks[i] for i in indices]
    vertical1 = np.linalg.norm(points[1] - points[5])
    vertical2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])
    return (vertical1 + vertical2) / (2.0 * horizontal)

def process_frame(frame):
    global blink_count, frame_count, start_time, last_sample_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    # Emotion detection
    latest_emotion = "Neutral"
    latest_confidence = 0.0
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w_face, h_face) in faces:
        face_img = gray[y:y + h_face, x:x + w_face]
        face_img = cv2.resize(face_img, (48, 48))
        face_img = np.expand_dims(np.expand_dims(face_img, -1), 0) / 255.0
        prediction = emotion_model.predict(face_img, verbose=0)
        idx = np.argmax(prediction)
        latest_emotion = emotion_labels[idx]
        latest_confidence = prediction[0][idx] * 100
        break

    # Defaults
    avg_ear = 0.3
    smile_ratio = 0.0
    brow_furrow_score = 0.0
    fatigue_score = 0.0
    movement_code = 1
    tilt_code = 0

    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        mesh = results.multi_face_landmarks[0].landmark
        landmarks = np.array([[int(p.x * w), int(p.y * h)] for p in mesh])
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0

        avg_ear = ear
        if ear < 0.23:
            frame_count += 1
        else:
            if 2 < frame_count < 20:
                blink_count += 1
            frame_count = 0

        mouth_left, mouth_right = landmarks[61], landmarks[291]
        mouth_top, mouth_bottom = landmarks[13], landmarks[14]
        vertical_dist = np.linalg.norm(mouth_top - mouth_bottom)
        if vertical_dist < 1e-2:
            smile_ratio = 0.0
        else:
            smile_ratio = np.linalg.norm(mouth_left - mouth_right) / vertical_dist

        brow_furrow_score = np.linalg.norm(landmarks[70] - landmarks[300])

        fatigue_score = (6 * blink_count / max(1, (time.time() - start_time) / 60)) * 0.2 \
                        + (1 - min(avg_ear / 0.4, 1.0)) * 0.3 \
                        + (1 if brow_furrow_score < 35 else 0) * 0.2

    sample_now = False
    current_time = time.time()
    if current_time - last_sample_time >= 10:
        sample_now = True
        last_sample_time = current_time
        blink_count = 0
        start_time = current_time

    metrics = {
        'ear': round(avg_ear, 3),
        'blink_rate': 6 * blink_count,
        'smile_ratio': round(smile_ratio, 3),
        'brow_furrow_score': round(brow_furrow_score, 2),
        'movement_code': movement_code,
        'tilt_code': tilt_code,
        'emotion': latest_emotion,
        'fatigue_score': round(fatigue_score, 2),
        'sample_ready': sample_now
    }

    return metrics, frame

def finalize_results(metrics_buffer, personal_info):
    if not metrics_buffer:
        print("[Metrics] No metrics collected.")
        return None, None, {'error': 'No metrics collected'}
    df = pd.DataFrame(metrics_buffer)
    if df['emotion'].dtype == object:
        df['emotion'] = df['emotion'].astype('category').cat.codes

    X_scaled = scaler.transform(df)
    X_cnn_input = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    cnn_features = cnn_model.predict(X_cnn_input, verbose=0)
    predictions = rf_model.predict(cnn_features)
    predicted_labels = label_encoder.inverse_transform(predictions)
    df['Predicted_Label'] = predicted_labels

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"reports/{personal_info.get('name', 'user').replace(' ', '_')}_behavioral_{timestamp}"
    csv_path = base + ".csv"
    pdf_path = base + ".pdf"

    df.to_csv(csv_path, index=False)
    summary = dict(Counter(predicted_labels))
    summary['Most Frequent State'] = max(summary, key=summary.get)
    summary['Total Samples'] = len(df)

    generate_pdf(personal_info, summary, pdf_path)
    return csv_path, pdf_path, summary

'''

'''

import cv2
import numpy as np
import pandas as pd
import time
from datetime import datetime
from collections import Counter
import joblib
from tensorflow.keras.models import load_model
import mediapipe as mp
from report_generator import generate_pdf

# === Load Models ===
cnn_model = load_model("/home/abhiram1289/Desktop/IIT TIRUPATHI/eegstress/cnn_feature_extractor.h5")
rf_model = joblib.load("/home/abhiram1289/Desktop/IIT TIRUPATHI/eegstress/cnn_rf_model.pkl")
scaler = joblib.load("/home/abhiram1289/Desktop/IIT TIRUPATHI/eegstress/cnn_rf_scaler.pkl")
label_encoder = joblib.load("/home/abhiram1289/Desktop/IIT TIRUPATHI/eegstress/cnn_rf_label_encoder.pkl")
emotion_model = load_model('/home/abhiram1289/emotion_model.h5')

# === Setup ===
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# === State ===
blink_count = 0
frame_count = 0
start_time = time.time()

# === EAR Helper ===
def eye_aspect_ratio(landmarks, indices):
    points = [landmarks[i] for i in indices]
    vertical1 = np.linalg.norm(points[1] - points[5])
    vertical2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])
    return (vertical1 + vertical2) / (2.0 * horizontal)

def process_frame(frame):
    global blink_count, frame_count

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    # Emotion
    latest_emotion = "Neutral"
    latest_confidence = 0.0
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w_face, h_face) in faces:
        face_img = gray[y:y + h_face, x:x + w_face]
        face_img = cv2.resize(face_img, (48, 48))
        face_img = np.expand_dims(np.expand_dims(face_img, -1), 0) / 255.0
        prediction = emotion_model.predict(face_img, verbose=0)
        idx = np.argmax(prediction)
        latest_emotion = emotion_labels[idx]
        latest_confidence = prediction[0][idx] * 100
        break

    # Face landmarks
    avg_ear = 0.3
    smile_ratio = 0.0
    brow_furrow_score = 0.0
    fatigue_score = 0.0
    movement_code = 1
    tilt_code = 0

    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        mesh = results.multi_face_landmarks[0].landmark
        landmarks = np.array([[int(p.x * w), int(p.y * h)] for p in mesh])
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0

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

        fatigue_score = (6 * blink_count / max(1, (time.time() - start_time) / 60)) * 0.2 \
                        + (1 - min(avg_ear / 0.4, 1.0)) * 0.3 \
                        + (1 if brow_furrow_score < 35 else 0) * 0.2

    metrics = {
        'ear': round(avg_ear, 3),
        'blink_rate': 6 * blink_count,
        'smile_ratio': round(smile_ratio, 3),
        'brow_furrow_score': round(brow_furrow_score, 2),
        'movement_code': movement_code,
        'tilt_code': tilt_code,
        'emotion': latest_emotion,
        'fatigue_score': round(fatigue_score, 2)
    }

    return metrics, frame

def finalize_results(metrics_buffer, personal_info):
    if not metrics_buffer:
        print("[Metrics] No metrics collected.")
        return None, None, {'error': 'No metrics collected'}
    df = pd.DataFrame(metrics_buffer)
    if df['emotion'].dtype == object:
        df['emotion'] = df['emotion'].astype('category').cat.codes

    X_scaled = scaler.transform(df)
    X_cnn_input = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    cnn_features = cnn_model.predict(X_cnn_input, verbose=0)
    predictions = rf_model.predict(cnn_features)
    predicted_labels = label_encoder.inverse_transform(predictions)
    df['Predicted_Label'] = predicted_labels

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"reports/{personal_info.get('name', 'user').replace(' ', '_')}_behavioral_{timestamp}"
    csv_path = base + ".csv"
    pdf_path = base + ".pdf"

    df.to_csv(csv_path, index=False)
    summary = dict(Counter(predicted_labels))
    summary['Most Frequent State'] = max(summary, key=summary.get)
    summary['Total Samples'] = len(df)

    generate_pdf(personal_info, summary, pdf_path)
    return csv_path, pdf_path, summary
'''
