import os
from datetime import datetime

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "session_logs.txt")

os.makedirs(LOG_DIR, exist_ok=True)

def log_event(text):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    with open(LOG_FILE, "a") as f:
        f.write(f"{timestamp} {text}\n")

def log_prediction(input_text, label, confidence, fallback_used, final_label):
    log_event(f"Input: {input_text}")
    log_event(f"Initial Prediction: {label} | Confidence: {confidence:.2f}")
    if fallback_used:
        log_event(f"Fallback triggered. Final Label after clarification: {final_label}")
    else:
        log_event(f"Prediction accepted without fallback. Final Label: {label}")
    log_event("-" * 50)
