"""HTTP integration surface for Bolna VoiceOps agents."""

import hmac
import logging
import os
from typing import Optional

import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from echo.rag import echo_is_ready, query_echo
from persistence import (
    consume_bolna_session_grant,
    init_db,
    save_bolna_execution,
    save_voice_interaction,
)
from voice_security import verify_bolna_session_grant


app = FastAPI(title="Argus VoiceOps", version="1.0.0")
logger = logging.getLogger(__name__)
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ARGUS_UI_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Argus-Webhook-Token"],
    max_age=600,
)


@app.on_event("startup")
def startup():
    init_db()
    if os.getenv("ARGUS_BUILD_INDEX_ON_STARTUP", "true").lower() in {"1", "true", "yes"} and not echo_is_ready():
        from echo.ingest import ingest_data_folder

        ingest_data_folder(os.getenv("ARGUS_DATA_DIR", "data"))


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    transcript: Optional[str] = Field(default=None, max_length=2000)
    interaction_id: Optional[str] = Field(default=None, max_length=256)
    latest_anomaly: Optional[str] = Field(default=None, max_length=12000)


class SessionRequest(BaseModel):
    user_data: dict = Field(default_factory=dict)


@app.get("/health")
def health():
    return {"status": "ok", "knowledge_base_ready": echo_is_ready()}


@app.post("/api/voiceops/answer")
def answer_question(
    payload: QuestionRequest,
    authorization: Optional[str] = Header(default=None),
):
    expected = os.getenv("ARGUS_VOICEOPS_API_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=503, detail="VoiceOps API is not configured.")
    supplied = (authorization or "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")
    init_db()
    interaction_id = payload.interaction_id
    try:
        if not echo_is_ready():
            raise RuntimeError("Knowledge base is not ready.")
        result = query_echo(
            payload.question,
            anomaly_context=payload.latest_anomaly,
            voice_mode=True,
        )
    except Exception:
        # Return a spoken, structured escalation instead of a non-200 response.
        # Bolna can then close the current turn honestly instead of promising
        # an asynchronous follow-up that this prototype cannot deliver.
        logger.exception("VoiceOps guidance lookup failed")
        result = {
            "answer": (
                "I could not retrieve documented guidance during this call. I cannot "
                "provide sensor-specific steps without that evidence. Please follow "
                "your approved site procedure and contact a qualified human operator now."
            ),
            "sources": [],
            "confidence": 0.0,
            "confidence_label": "retrieval confidence (not answer correctness)",
            "escalate": True,
            "model_version": "voiceops-fallback",
        }
    save_voice_interaction(
        payload.question,
        payload.transcript or payload.question,
        result,
        interaction_id,
        anomaly_context=payload.latest_anomaly,
    )
    source_names = list(dict.fromkeys(
        str(source.get("source"))
        for source in result.get("sources", [])
        if source.get("source")
    ))
    spoken_answer = result["answer"]
    if source_names:
        spoken_answer += " Source: " + "; ".join(source_names[:3]) + "."
    spoken_answer += (
        f" Retrieval relevance: {round(float(result['confidence']) * 100)} percent; "
        "this is not a measure of answer correctness."
    )
    if result.get("escalate"):
        spoken_answer += " Human operator review is required."
    return {
        "answer": result["answer"],
        "spoken_answer": spoken_answer,
        "sources": result["sources"],
        "confidence": result["confidence"],
        "confidence_label": result["confidence_label"],
        "escalate": result["escalate"],
        "model_version": result["model_version"],
        "matched_sensor": result.get("matched_sensor"),
        "sensor_event": result.get("sensor_event"),
        "interaction_id": interaction_id,
    }


@app.post("/api/bolna-session")
def create_bolna_session(payload: SessionRequest, origin: Optional[str] = Header(default=None)):
    # This endpoint creates live Bolna sessions using a server-side API key.
    # Only allow it to be called from the configured Argus UI origins.
    if not origin or origin not in allowed_origins:
        raise HTTPException(status_code=403, detail="Bolna sessions must be started from the Argus UI.")
    user_data = dict(payload.user_data)
    session_grant = str(user_data.pop("_argus_session_grant", ""))
    try:
        grant = verify_bolna_session_grant(session_grant)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired voice-session grant.") from exc
    init_db()
    grant_status = consume_bolna_session_grant(
        grant["user_id"], grant["nonce"], grant["expires_at"]
    )
    if grant_status == "replayed":
        raise HTTPException(status_code=409, detail="This voice-session grant has already been used. Refresh Argus and try again.")
    if grant_status == "rate_limited":
        raise HTTPException(status_code=429, detail="Too many voice sessions. Wait a few minutes and try again.")
    api_key = os.getenv("BOLNA_API_KEY", "")
    agent_id = os.getenv("BOLNA_AGENT_ID", "")
    if not api_key or not agent_id:
        raise HTTPException(status_code=503, detail="Bolna browser calling is not configured.")
    if len(str(user_data).encode("utf-8")) > 50_000:
        raise HTTPException(status_code=413, detail="Bolna user_data exceeds 50 KB.")
    try:
        response = requests.post(
            "https://api.bolna.ai/web-call/session",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"agent_id": agent_id, "user_data": user_data},
            timeout=15,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="Could not reach Bolna session service.") from exc
    try:
        data = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Bolna returned an invalid session response.") from exc
    if not response.ok:
        raise HTTPException(status_code=response.status_code, detail=data)
    return data


@app.post("/api/voiceops/bolna-webhook")
def bolna_webhook(payload: dict, x_argus_webhook_token: Optional[str] = Header(default=None)):
    expected = os.getenv("ARGUS_BOLNA_WEBHOOK_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=503, detail="Webhook is not configured.")
    if not hmac.compare_digest(x_argus_webhook_token or "", expected):
        raise HTTPException(status_code=401, detail="Unauthorized")
    # Bolna sends repeated updates. Upsert by execution ID, preserving final
    # fields when earlier status events omit them.
    try:
        execution_id = save_bolna_execution(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"received": True, "execution_id": execution_id}
