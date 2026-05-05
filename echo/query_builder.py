from argus_logger import get_recent_anomalies, get_all_anomalies

ATTACK_DESCRIPTIONS = {
    "replay_attack": (
        "a Replay Attack — the sensor variance dropped to exactly 0.0, "
        "indicating the sensor output has been frozen and is broadcasting "
        "a static baseline while physical conditions may have changed."
    ),
    "low_entropy_anomaly": (
        "a Low-Entropy Anomaly — near-zero variance combined with an elevated "
        "MSE score, suggesting the sensor is artificially stable in a system "
        "where natural thermodynamic noise is expected."
    ),
    "high_deviation_anomaly": (
        "a High-Deviation Anomaly — the LSTM-AE reconstruction error is "
        "significantly above the normal baseline, indicating sensor readings "
        "that deviate sharply from learned normal behaviour."
    ),
    "subtle_anomaly": (
        "a Subtle Anomaly — MAE score is above the detection threshold but "
        "variance remains present, suggesting early-stage deviation or "
        "gradual sensor drift."
    ),
    "unknown": (
        "an anomaly of unclassified type — insufficient signal characteristics "
        "to determine attack vector."
    )
}

def build_query_from_anomaly(anomaly: dict) -> str:
    attack_desc = ATTACK_DESCRIPTIONS.get(
        anomaly.get("attack_hint", "unknown"),
        ATTACK_DESCRIPTIONS["unknown"]
    )

    query = f"""
An anomaly has been detected by the Argus Edge AI system in a safety-critical 
industrial control system. Here are the technical details:

- Sensor ID: {anomaly['sensor_id']}
- Facility: {anomaly['facility']}
- Timestamp: {anomaly['timestamp']}
- MAE Anomaly Score: {anomaly['mse_score']} (detection threshold: 0.13)
- Sensor Variance: {anomaly['variance']}
- Detected Pattern: {attack_desc}

Based on ICS security literature and known cyber-physical attack patterns:
1. What attack technique does this pattern most closely match?
2. What is the recommended immediate operator response?
3. What downstream systems or sensors should be cross-checked?
4. What is the potential physical consequence if this goes unaddressed?
""".strip()

    return query

def build_query_from_latest() -> tuple[dict, str]:
    recent = get_recent_anomalies(n=1)
    if not recent:
        return None, None
    anomaly = recent[-1]
    return anomaly, build_query_from_anomaly(anomaly)

def build_query_from_id(anomaly_id: str) -> tuple[dict, str]:
    all_logs = get_all_anomalies()
    match = next((a for a in all_logs if a["id"] == anomaly_id), None)
    if not match:
        return None, None
    return match, build_query_from_anomaly(match)

def build_summary_query(n: int = 5) -> str:
    recent = get_recent_anomalies(n=n)
    if not recent:
        return None

    lines = []
    for a in recent:
        lines.append(
            f"- Sensor {a['sensor_id']} | Facility: {a['facility']} | "
            f"MAE: {a['mse_score']} | Pattern: {a['attack_hint']} | "
            f"Time: {a['timestamp']}"
        )

    query = f"""
The Argus Edge AI system has detected {len(recent)} anomalies recently 
in safety-critical ICS infrastructure. Summary:

{chr(10).join(lines)}

Based on this pattern of detections:
1. Is there evidence of a coordinated multi-sensor attack?
2. Which facility appears most at risk?
3. What attack campaign does this pattern suggest?
4. What is the recommended security posture for the operator?
""".strip()

    return query