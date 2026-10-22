"""Short-lived grants for browser-initiated Bolna voice sessions."""

import base64
import hashlib
import hmac
import json
import os
import secrets
import time


def _secret() -> bytes:
    secret = os.getenv("ARGUS_VOICEOPS_API_TOKEN", "")
    if not secret:
        raise RuntimeError("ARGUS_VOICEOPS_API_TOKEN is required for voice-session grants.")
    return secret.encode("utf-8")


def issue_bolna_session_grant(user_id: str, ttl_seconds: int = 300) -> str:
    payload = {
        "user_id": str(user_id),
        "expires_at": int(time.time()) + ttl_seconds,
        "nonce": secrets.token_urlsafe(24),
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).rstrip(b"=").decode("ascii")
    signature = hmac.new(_secret(), encoded.encode("ascii"), hashlib.sha256).digest()
    signed = base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")
    return f"{encoded}.{signed}"


def verify_bolna_session_grant(token: str) -> dict:
    try:
        encoded, supplied_signature = token.split(".", 1)
        expected_signature = base64.urlsafe_b64encode(
            hmac.new(_secret(), encoded.encode("ascii"), hashlib.sha256).digest()
        ).rstrip(b"=").decode("ascii")
        if not hmac.compare_digest(supplied_signature, expected_signature):
            raise ValueError("Invalid grant signature.")
        payload = json.loads(
            base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        )
    except (ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Invalid voice-session grant.") from exc

    now = int(time.time())
    expires_at = payload.get("expires_at")
    if (
        not isinstance(payload.get("user_id"), str)
        or not payload["user_id"]
        or not isinstance(payload.get("nonce"), str)
        or not payload["nonce"]
        or not isinstance(expires_at, int)
        or expires_at <= now
        or expires_at > now + 600
    ):
        raise ValueError("Expired or invalid voice-session grant.")
    return payload
