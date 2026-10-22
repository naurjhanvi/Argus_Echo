import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path


STORAGE_ROOT = Path(os.getenv("ARGUS_STORAGE_DIR", "ml"))
DB_PATH = Path(os.getenv("ARGUS_DB_PATH", STORAGE_ROOT / "argus_echo.db"))
SQLITE_TIMEOUT_SECONDS = int(os.getenv("ARGUS_SQLITE_TIMEOUT_SECONDS", "30"))


def utc_now() -> str:
    return datetime.utcnow().isoformat()


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT_SECONDS)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db():
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                raw_sensor_count INTEGER NOT NULL,
                expected_feature_count INTEGER NOT NULL,
                epochs INTEGER NOT NULL,
                batch_size INTEGER NOT NULL,
                training_rows INTEGER NOT NULL,
                training_file_path TEXT,
                model_path TEXT NOT NULL,
                scaler_path TEXT NOT NULL,
                config_path TEXT NOT NULL,
                config_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS diagnostic_runs (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                uploaded_file_name TEXT NOT NULL,
                uploaded_file_path TEXT,
                result_path TEXT,
                anomaly_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (model_id) REFERENCES models(id)
            );

            CREATE TABLE IF NOT EXISTS anomalies (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                sensor_id TEXT NOT NULL,
                facility TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                mse_score REAL NOT NULL,
                variance REAL NOT NULL,
                attack_hint TEXT NOT NULL,
                telemetry_context_json TEXT,
                echo_answer TEXT,
                logged_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (run_id) REFERENCES diagnostic_runs(id),
                FOREIGN KEY (model_id) REFERENCES models(id)
            );

            CREATE TABLE IF NOT EXISTS voice_incidents (
                id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                transcript TEXT,
                answer TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                model_version TEXT NOT NULL,
                interaction_id TEXT,
                anomaly_context_json TEXT,
                status TEXT NOT NULL DEFAULT 'open',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS voice_interactions (
                id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                transcript TEXT,
                answer TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                model_version TEXT NOT NULL,
                interaction_id TEXT,
                escalated INTEGER NOT NULL,
                anomaly_context_json TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS bolna_executions (
                execution_id TEXT PRIMARY KEY,
                agent_id TEXT,
                status TEXT NOT NULL,
                transcript TEXT,
                summary TEXT,
                conversation_duration REAL,
                total_cost REAL,
                recording_url TEXT,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS bolna_session_grants (
                nonce TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                expires_at INTEGER NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_bolna_session_grants_user_created
                ON bolna_session_grants(user_id, created_at);
            """
        )
        for table in ("voice_incidents", "voice_interactions"):
            columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            if "anomaly_context_json" not in columns:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN anomaly_context_json TEXT")
        anomaly_columns = {row["name"] for row in conn.execute("PRAGMA table_info(anomalies)")}
        if "telemetry_context_json" not in anomaly_columns:
            conn.execute("ALTER TABLE anomalies ADD COLUMN telemetry_context_json TEXT")


def save_voice_interaction(
    question: str,
    transcript: str | None,
    result: dict,
    interaction_id: str | None = None,
    anomaly_context: dict | str | None = None,
):
    interaction_pk = uuid.uuid4().hex
    now = utc_now()
    anomaly_context_json = json.dumps(anomaly_context, ensure_ascii=False) if anomaly_context else None
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO voice_interactions
            (id, question, transcript, answer, sources_json, confidence, model_version,
             interaction_id, escalated, anomaly_context_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (interaction_pk, question, transcript, result["answer"],
             json.dumps(result.get("sources", [])), float(result.get("confidence", 0)),
             str(result.get("model_version", "unknown")), interaction_id,
             int(bool(result.get("escalate"))), anomaly_context_json, now),
        )
        if result.get("escalate"):
            conn.execute(
                """INSERT INTO voice_incidents
                (id, question, transcript, answer, sources_json, confidence, model_version,
                 interaction_id, anomaly_context_json, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)""",
                (uuid.uuid4().hex, question, transcript, result["answer"],
                 json.dumps(result.get("sources", [])), float(result.get("confidence", 0)),
                 str(result.get("model_version", "unknown")), interaction_id, anomaly_context_json, now),
            )
    return interaction_pk


def list_voice_interactions(n: int = 50):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM voice_interactions ORDER BY created_at DESC LIMIT ?", (n,)
        ).fetchall()
    results = [dict(row) for row in rows]
    for row in results:
        row["sources"] = json.loads(row.pop("sources_json"))
        row["anomaly_context"] = json.loads(row.pop("anomaly_context_json")) if row.get("anomaly_context_json") else None
    return results


def list_voice_incidents(n: int = 50):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM voice_incidents ORDER BY created_at DESC LIMIT ?", (n,)
        ).fetchall()
    results = [dict(row) for row in rows]
    for row in results:
        row["sources"] = json.loads(row.pop("sources_json"))
        row["anomaly_context"] = json.loads(row.pop("anomaly_context_json")) if row.get("anomaly_context_json") else None
    return results


def update_voice_incident_status(incident_id: str, status: str):
    if status not in {"open", "acknowledged", "resolved"}:
        raise ValueError("Invalid incident status.")
    with get_connection() as conn:
        return conn.execute(
            "UPDATE voice_incidents SET status = ? WHERE id = ?", (status, incident_id)
        ).rowcount > 0


def save_bolna_execution(payload: dict):
    execution_id = str(payload.get("execution_id") or payload.get("id") or "").strip()
    if not execution_id:
        raise ValueError("Bolna execution payload is missing its execution ID.")
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO bolna_executions
            (execution_id, agent_id, status, transcript, summary, conversation_duration,
             total_cost, recording_url, payload_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(execution_id) DO UPDATE SET
              agent_id=excluded.agent_id, status=excluded.status,
              transcript=COALESCE(excluded.transcript, bolna_executions.transcript),
              summary=COALESCE(excluded.summary, bolna_executions.summary),
              conversation_duration=COALESCE(excluded.conversation_duration, bolna_executions.conversation_duration),
              total_cost=COALESCE(excluded.total_cost, bolna_executions.total_cost),
              recording_url=COALESCE(excluded.recording_url, bolna_executions.recording_url),
              payload_json=excluded.payload_json, updated_at=excluded.updated_at""",
            (execution_id, payload.get("agent_id"), str(payload.get("status", "unknown")),
             payload.get("transcript"), payload.get("summary"), payload.get("conversation_duration"),
             payload.get("total_cost"), (payload.get("telephony_data") or {}).get("recording_url"),
             json.dumps(payload), utc_now()),
        )
    return execution_id


def consume_bolna_session_grant(user_id: str, nonce: str, expires_at: int) -> str:
    """Consume a one-time UI grant, with a small per-user session-start limit."""
    now = int(datetime.utcnow().timestamp())
    with get_connection() as conn:
        conn.execute("DELETE FROM bolna_session_grants WHERE expires_at < ?", (now,))
        count = conn.execute(
            "SELECT COUNT(*) FROM bolna_session_grants WHERE user_id = ? AND created_at > ?",
            (user_id, now - 600),
        ).fetchone()[0]
        if count >= 5:
            return "rate_limited"
        try:
            conn.execute(
                "INSERT INTO bolna_session_grants (nonce, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
                (nonce, user_id, int(expires_at), now),
            )
        except sqlite3.IntegrityError:
            return "replayed"
    return "consumed"


def list_bolna_executions(n: int = 50):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM bolna_executions ORDER BY updated_at DESC LIMIT ?", (n,)
        ).fetchall()
    results = [dict(row) for row in rows]
    for row in results:
        row["payload"] = json.loads(row.pop("payload_json"))
    return results


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        120_000,
    ).hex()
    return digest, salt


def create_user(email: str, password: str):
    email = email.strip().lower()
    password_hash, password_salt = hash_password(password)
    user_id = uuid.uuid4().hex
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO users (id, email, password_hash, password_salt, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, email, password_hash, password_salt, utc_now()),
        )
    return {"id": user_id, "email": email}


def authenticate_user(email: str, password: str):
    email = email.strip().lower()
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if row is None:
        return None

    password_hash, _ = hash_password(password, row["password_salt"])
    if password_hash != row["password_hash"]:
        return None
    return {"id": row["id"], "email": row["email"]}


def user_storage_dir(user_id: str) -> Path:
    path = STORAGE_ROOT / "users" / user_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_storage_dir(user_id: str, model_id: str) -> Path:
    path = user_storage_dir(user_id) / "models" / model_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_storage_dir(user_id: str, run_id: str) -> Path:
    path = user_storage_dir(user_id) / "diagnostics" / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_uploaded_file(uploaded_file, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(uploaded_file.getvalue())
    return destination


def create_model_record(
    user_id: str,
    name: str,
    config: dict,
    training_file_path: Path,
    model_path: Path,
    scaler_path: Path,
    config_path: Path,
):
    model_id = model_path.parent.name
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO models (
                id, user_id, name, raw_sensor_count, expected_feature_count, epochs,
                batch_size, training_rows, training_file_path, model_path, scaler_path,
                config_path, config_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_id,
                user_id,
                name,
                config["raw_sensor_count"],
                config["num_features"],
                config["epochs"],
                config["batch_size"],
                config["training_rows"],
                str(training_file_path) if training_file_path else None,
                str(model_path),
                str(scaler_path),
                str(config_path),
                json.dumps(config),
                utc_now(),
            ),
        )
    return get_model(model_id, user_id)


def list_user_models(user_id: str):
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM models
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_model(model_id: str, user_id: str):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM models WHERE id = ? AND user_id = ?",
            (model_id, user_id),
        ).fetchone()
    return dict(row) if row else None


def create_diagnostic_run(
    user_id: str,
    model_id: str,
    uploaded_file_name: str,
    uploaded_file_path: Path,
    result_path: Path,
):
    run_id = result_path.parent.name
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO diagnostic_runs (
                id, user_id, model_id, uploaded_file_name, uploaded_file_path,
                result_path, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                user_id,
                model_id,
                uploaded_file_name,
                str(uploaded_file_path),
                str(result_path),
                utc_now(),
            ),
        )
    return run_id


def complete_diagnostic_run(run_id: str, user_id: str, anomaly_count: int):
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE diagnostic_runs
            SET anomaly_count = ?
            WHERE id = ? AND user_id = ?
            """,
            (anomaly_count, run_id, user_id),
        )


def log_anomaly_record(
    user_id: str,
    run_id: str,
    model_id: str,
    sensor_id: str,
    facility: str,
    timestamp: int,
    mse_score: float,
    variance: float,
    attack_hint: str,
    telemetry_context: dict | None = None,
):
    anomaly_id = uuid.uuid4().hex
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO anomalies (
                id, user_id, run_id, model_id, sensor_id, facility, timestamp,
                mse_score, variance, attack_hint, telemetry_context_json, logged_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                anomaly_id,
                user_id,
                run_id,
                model_id,
                sensor_id,
                facility,
                int(timestamp),
                round(float(mse_score), 4),
                round(float(variance), 6),
                attack_hint,
                json.dumps(telemetry_context, ensure_ascii=False) if telemetry_context else None,
                utc_now(),
            ),
        )
    return get_anomaly(anomaly_id, user_id)


def list_recent_anomalies(user_id: str, n: int = 5):
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM anomalies
            WHERE user_id = ?
            ORDER BY logged_at DESC
            LIMIT ?
            """,
            (user_id, n),
        ).fetchall()
    return [_decode_anomaly_row(row) for row in rows]


def get_latest_anomaly(user_id: str):
    rows = list_recent_anomalies(user_id, n=1)
    return rows[0] if rows else None


def list_latest_anomaly_per_sensor(user_id: str, n: int = 12):
    """Return the most recent detected event per sensor for voice-session context."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM (
                SELECT anomalies.*,
                       ROW_NUMBER() OVER (
                           PARTITION BY sensor_id ORDER BY logged_at DESC, timestamp DESC
                       ) AS sensor_rank
                FROM anomalies
                WHERE user_id = ?
            )
            WHERE sensor_rank = 1
            ORDER BY logged_at DESC
            LIMIT ?
            """,
            (user_id, n),
        ).fetchall()
    results = [_decode_anomaly_row(row) for row in rows]
    for row in results:
        row.pop("sensor_rank", None)
    return results


def get_anomaly(anomaly_id: str, user_id: str):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM anomalies WHERE id = ? AND user_id = ?",
            (anomaly_id, user_id),
        ).fetchone()
    return _decode_anomaly_row(row) if row else None


def _decode_anomaly_row(row):
    result = dict(row)
    raw = result.pop("telemetry_context_json", None)
    try:
        result["telemetry"] = json.loads(raw) if raw else None
    except (TypeError, json.JSONDecodeError):
        result["telemetry"] = None
    return result


def save_echo_answer(anomaly_id: str, user_id: str, answer: str):
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE anomalies
            SET echo_answer = ?
            WHERE id = ? AND user_id = ?
            """,
            (answer, anomaly_id, user_id),
        )


def list_diagnostic_runs(user_id: str, n: int = 10):
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT diagnostic_runs.*, models.name AS model_name
            FROM diagnostic_runs
            JOIN models ON models.id = diagnostic_runs.model_id
            WHERE diagnostic_runs.user_id = ?
            ORDER BY diagnostic_runs.created_at DESC
            LIMIT ?
            """,
            (user_id, n),
        ).fetchall()
    return [dict(row) for row in rows]
