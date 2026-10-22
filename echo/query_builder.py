from argus_logger import get_recent_anomalies, get_all_anomalies

def build_query_from_anomaly(anomaly: dict) -> str:
    telemetry = anomaly.get("telemetry") or {}
    threshold = telemetry.get("anomaly_threshold")
    threshold_detail = (
        f"The calibrated threshold for this model is {threshold}; "
        f"this score is {telemetry.get('score_above_threshold', 'not available')} times that threshold."
        if threshold is not None
        else "No calibrated threshold is recorded for this event."
    )
    query = f"""
Argus detected a telemetry deviation from the trained baseline. Explain this
recorded event for an industrial operator using the provided evidence. Do not
diagnose a physical fault or cyberattack from a heuristic pattern label.

- Sensor ID: {anomaly['sensor_id']}
- Source file / facility label: {anomaly['facility']}
- Row / timestamp index: {anomaly['timestamp']}
- Window reconstruction error: {anomaly['mse_score']} ({threshold_detail})
- Sensor Variance: {anomaly['variance']}
- Descriptive pattern heuristic: {anomaly.get('attack_hint', 'unknown')}

Explain what Argus measured and its limits, where the event was recorded, and
which safe verification and escalation checks are supported by the cited
documents. Treat reconstruction error as a deviation score, not confidence or
proof of an anomaly. Distinguish general ICS guidance from approved site procedures.
Never prescribe equipment control changes or claim a diagnosis.
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
            f"- Sensor {a['sensor_id']} | File/facility label: {a['facility']} | "
            f"Window reconstruction error: {a['mse_score']} | "
            f"Pattern heuristic: {a['attack_hint']} | Row: {a['timestamp']}"
        )

    query = f"""
Argus detected {len(recent)} telemetry deviations from trained baselines.
Summarize the logged evidence for an industrial operator without diagnosing
faults or cyberattacks. Summary:

{chr(10).join(lines)}

Explain which sensors and file labels were involved, whether multiple events
occurred close together, what location or time metadata is missing, and which
safe verification/escalation checks the cited documents support. Do not infer
an attack campaign, physical cause, or physical consequence from these scores.
""".strip()

    return query
