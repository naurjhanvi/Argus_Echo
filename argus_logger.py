import json
import os
from datetime import datetime

LOG_PATH = "data/anomaly_logs.json"
os.makedirs("data", exist_ok=True)

def log_anomaly(
    sensor_id: str,
    facility: str,
    timestamp: int,
    mse_score: float,
    variance: float,
    attack_hint: str = "unknown"
):
    entry = {
        "id": datetime.utcnow().strftime("%Y%m%d%H%M%S%f"),
        "sensor_id": sensor_id,
        "facility": facility,
        "timestamp": timestamp,
        "mse_score": round(float(mse_score), 4),
        "variance": round(float(variance), 6),
        "attack_hint": attack_hint,
        "logged_at": datetime.utcnow().isoformat()
    }

    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []

    logs.append(entry)

    with open(LOG_PATH, "w") as f:
        json.dump(logs, f, indent=2)

    return entry

def get_recent_anomalies(n: int = 10):
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r") as f:
        try:
            logs = json.load(f)
        except json.JSONDecodeError:
            return []
    return logs[-n:]

def get_all_anomalies():
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def infer_attack_hint(variance: float, mse_score: float) -> str:
    if variance == 0.0:
        return "replay_attack"          
    elif variance < 0.01 and mse_score > 1.0:
        return "low_entropy_anomaly"   
    elif mse_score > 3.0:
        return "high_deviation_anomaly" 
    else:
        return "subtle_anomaly"