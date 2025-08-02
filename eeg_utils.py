# eeg_utils.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU for TensorFlow

import joblib
import gdown

# === Auto-download Voting Ensemble model if missing ===
MODEL_PATH = "models/eeg_models/voting_model.pkl"
GOOGLE_DRIVE_FILE_ID = "1Cd0Bav8E1GF8lN29_WkpLuJO9xYOLZfB"

if not os.path.exists(MODEL_PATH):
    print("Voting Ensemble model not found. Downloading from Google Drive...")
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    gdown.download(url, MODEL_PATH, quiet=False)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model download failed. Check your internet or Google Drive file ID.")

import numpy as np
import pandas as pd
import datetime, time, serial, joblib, threading
import scipy.signal as signal
from collections import Counter
DEFAULT_PORT = '/dev/ttyACM0'

# Removed: port = '/dev/ttyACM0'
baud = 115200
fs = 256
buffer_seconds = 6
samples_needed = fs * buffer_seconds

#model = joblib.load("/home/abhiram1289/Desktop/mentalhealth11/models/eeg_models/voting_model.pkl")
#scaler = joblib.load("/home/abhiram1289/Desktop/mentalhealth11/models/eeg_models/scaler.pkl")
#le = joblib.load("/home/abhiram1289/Desktop/mentalhealth11/models/eeg_models/label_encoder.pkl")

# === Load Models ===
model = joblib.load(MODEL_PATH)
scaler = joblib.load("models/eeg_models/scaler.pkl")
le = joblib.load("models/eeg_models/label_encoder.pkl")

eeg_collector = {
    "running": False,
    "data": [],
    "lock": threading.Lock()
}

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return signal.filtfilt(b, a, data)

def compute_bandpower(data, fs, band):
    f, Pxx = signal.welch(data, fs, nperseg=fs*2)
    idx = (f >= band[0]) & (f <= band[1])
    return np.trapz(Pxx[idx], f[idx]), Pxx

def compute_entropy(psd):
    psd_norm = psd / np.sum(psd)
    return -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

def sample_entropy(x, m=2, r=0.2):
    x = np.array(x)
    N = len(x)
    r *= np.std(x)
    def _phi(m):
        x_m = np.array([x[i:i + m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(x_m[:, None] - x_m), axis=2) <= r, axis=0) - 1
        return np.sum(C) / (N - m + 1)
    return -np.log(_phi(m+1) / _phi(m) + 1e-12)

def get_stress_status(alpha, beta, theta, gamma, delta, spectral_entropy, delta_rp, theta_rp, alpha_rp, beta_rp):
    ba = beta / alpha if alpha else 0
    ta = theta / alpha if alpha else 0
    gb = gamma / beta if beta else 0
    tb = theta / beta if beta else 0
    db = delta / beta if beta else 0
    overall = (beta + gamma) / (alpha + theta) if (alpha + theta) else 0

    if ba > 3.0 and tb < 2 and ta < 1.0 and beta_rp > 0.4 and spectral_entropy > 2.5 and overall > 3.8:
        return "Distress"
    elif ba > 2.5 and ta < 1.5 and tb < 3.0:
        return "Acute Stress"
    elif ba > 1.8 and 1.0 < ta < 2.5:
        return "Functional Stress"
    elif 1.2 < ta < 2.5 and ba < 1.5 and tb > 2.8:
        return "Light Relaxation"
    elif ta > 2.8 and db > 2.0 and ba < 1.2:
        return "Deep Meditative"
    elif tb > 3.5 and ta < 1.3 and ba < 1.5 and gb > 0.8:
        return "Drowsy"
    elif ba < 1.2 and ta < 1.2 and overall < 1.4 and tb < 2.5:
        return "Resting"
    elif spectral_entropy < 2.5 and ba > 2.2:
        return "Focused but Stressed"
    elif spectral_entropy < 2.3 and gb > 1.0:
        return "Cognitive Overload"
    elif ta < 1.0 and tb < 2.0 and gb < 1.0 and db < 1.2:
        return "Alert"
    return "Transitional"

def start_eeg_collection(port=None):
    port = port or DEFAULT_PORT  # Use given port, else fallback to default

    def collect():
        try:
            ser = serial.Serial(port, baudrate=baud, timeout=2)
        except Exception as e:
            eeg_collector["error"] = str(e)
            eeg_collector["running"] = False
            return

        eeg_collector["data"] = []
        eeg_collector["running"] = True

        while eeg_collector["running"]:
            eeg_buffer = []
            while len(eeg_buffer) < samples_needed:
                try:
                    line = ser.readline().decode('utf-8').strip()
                    value = float(line)
                    eeg_buffer.append(value)
                except:
                    continue

            with eeg_collector["lock"]:
                eeg_collector["data"].append(eeg_buffer)

    threading.Thread(target=collect, daemon=True).start()

def stop_and_process_eeg(personal):
    eeg_collector["running"] = False
    time.sleep(1)

    all_samples = eeg_collector.get("data", [])
    label_counts = Counter()
    rule_labels = []
    total_samples = 0
    all_logs = []

    for eeg_buffer in all_samples:
        eeg = np.array(eeg_buffer)
        filtered = bandpass_filter(eeg, 1, 50, fs)

        delta, _ = compute_bandpower(filtered, fs, [0.5, 4])
        theta, _ = compute_bandpower(filtered, fs, [4, 8])
        alpha, _ = compute_bandpower(filtered, fs, [8, 13])
        beta, _ = compute_bandpower(filtered, fs, [13, 30])
        gamma, psd = compute_bandpower(filtered, fs, [30, 50])

        total = alpha + beta + theta + delta
        delta_rp = delta / total if total else 0
        theta_rp = theta / total if total else 0
        alpha_rp = alpha / total if total else 0
        beta_rp = beta / total if total else 0

        spectral_entropy = compute_entropy(psd)
        samp_entropy = sample_entropy(filtered)

        rule_label = get_stress_status(alpha, beta, theta, gamma, delta, spectral_entropy,
                                       delta_rp, theta_rp, alpha_rp, beta_rp)

        features = np.array([
            alpha, beta, theta, gamma, beta/alpha if alpha else 0,
            theta/alpha if alpha else 0, gamma/beta if beta else 0,
            theta/beta if beta else 0, delta/beta if beta else 0,
            (beta + gamma) / (alpha + theta + 1e-6),
            spectral_entropy, samp_entropy, delta_rp, theta_rp,
            alpha_rp, beta_rp,
            beta/alpha if alpha else 0,
            theta/alpha if alpha else 0,
            delta / (theta + 1e-6),
            alpha_rp - beta_rp,
            beta / (alpha + theta + 1e-6),
            np.sum(np.diff([delta, theta, alpha, beta, gamma]) > 0),
            compute_entropy(np.array([delta, theta, alpha, beta, gamma]))
        ]).reshape(1, -1)

        pred = model.predict(scaler.transform(features))[0]
        label = le.inverse_transform([pred])[0]

        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        log_row = [
            timestamp, alpha, beta, theta, gamma, delta,
            beta/alpha if alpha else 0, theta/alpha if alpha else 0, gamma/beta if beta else 0,
            theta/beta if beta else 0, delta/beta if beta else 0,
            (beta + gamma) / (alpha + theta + 1e-6),
            spectral_entropy, samp_entropy,
            delta_rp, theta_rp, alpha_rp, beta_rp,
            alpha_rp - beta_rp, beta / (alpha + theta + 1e-6),
            np.sum(np.diff([delta, theta, alpha, beta, gamma]) > 0),
            compute_entropy(np.array([delta, theta, alpha, beta, gamma])),
            rule_label, label
        ]
        all_logs.append(log_row)

        label_counts[label] += 1
        rule_labels.append(rule_label)
        total_samples += 1

    filename_base = personal['name'].replace(' ', '_') + "_eeg"
    csv_path = f"reports/{filename_base}.csv"

    df_log = pd.DataFrame(all_logs, columns=[
        "Time", "Alpha", "Beta", "Theta", "Gamma", "Delta",
        "B/A", "T/A", "G/B", "T/B", "D/B", "(B+G)/(A+T)",
        "Spectral Entropy", "Sample Entropy",
        "Delta RP", "Theta RP", "Alpha RP", "Beta RP",
        "FAA", "Engagement", "ZCR", "Spectral_Entropy_New",
        "Rule_Label", "ML_Label"
    ])
    df_log.to_csv(csv_path, index=False)

    most_common_rule = Counter(rule_labels).most_common(1)[0][0] if rule_labels else "N/A"
    return {
        "rule_based": most_common_rule,
        "ml_summary": dict(label_counts),
        "total_samples": total_samples
    }

def run_eeg_merge():
    path = os.path.join("reports", "anonymous_eeg.csv")
    if not os.path.exists(path):
        return {"EEG Rule-Based": "Parse Error", "EEG ML-Based": "Parse Error"}

    with open(path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        row = next(reader)
        rule = row[4]
        ml_parts = row[5:]
        ml_summary = ', '.join(ml_parts)
        return {
            "EEG Rule-Based": rule,
            "EEG ML-Based": ml_summary
        }

