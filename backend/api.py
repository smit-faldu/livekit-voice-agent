"""FastAPI backend: issues LiveKit access tokens and serves the built frontend.

A self-hosted LiveKit server has no token service, so this API signs JWTs with
LIVEKIT_API_KEY / LIVEKIT_API_SECRET. The response follows LiveKit's standard
token-endpoint format, so the React SDK's `TokenSource.endpoint` works as-is.

Security (the token is the key to the room, so this endpoint is a trust boundary):
  - caller must send `Authorization: Bearer <APP_ACCESS_KEY>` (from .env.local)
  - identity and room are chosen by the server, never by the client
  - clients may only dispatch our own agent
  - tokens expire after 10 minutes (self-hosted LiveKit can't revoke tokens)

Started by server.py together with the agent worker.
"""

import hmac
import os
import uuid
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from livekit import api
from pydantic import BaseModel, Field

load_dotenv(".env.local")

ROOT = Path(__file__).parent.parent
AGENT_NAME = "assistant"  # must match @server.rtc_session(agent_name=...) in agent/main.py
TOKEN_TTL = timedelta(minutes=10)

app = FastAPI(title="Local Voice Assistant")


def require_access_key(authorization: str = Header(default="")):
    expected = os.getenv("APP_ACCESS_KEY", "")
    if not expected:
        raise HTTPException(500, "APP_ACCESS_KEY is not set in .env.local")
    given = authorization.removeprefix("Bearer ").strip()
    # compare_digest: constant-time comparison, so response timing can't leak the key.
    if not hmac.compare_digest(given.encode(), expected.encode()):
        raise HTTPException(401, "invalid access key")


class TokenRequest(BaseModel):
    participant_name: str | None = Field(default=None, max_length=64)
    room_config: dict | None = None  # the SDK sends {agents: [{agentName: "assistant"}]}


@app.post("/api/token", status_code=201, dependencies=[Depends(require_access_key)])
def get_token(req: TokenRequest):
    key, secret, url = (os.getenv(k) for k in ("LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "LIVEKIT_URL"))
    if not (key and secret and url):
        raise HTTPException(500, "LIVEKIT_* env vars missing")

    # Only allow dispatching our own agent; ignore any other room settings from the client.
    requested = (req.room_config or {}).get("agents") or [{}]
    for a in requested:
        name = a.get("agentName") or a.get("agent_name") or AGENT_NAME
        if name != AGENT_NAME:
            raise HTTPException(403, f"agent {name!r} not allowed")

    # Fresh room per session: token-based dispatch only fires when the room is created.
    room = f"room-{uuid.uuid4().hex[:8]}"
    token = (
        api.AccessToken(key, secret)
        .with_identity(f"user-{uuid.uuid4().hex[:6]}")
        .with_name(req.participant_name or "User")
        .with_ttl(TOKEN_TTL)
        .with_grants(api.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=True))
        .with_room_config(api.RoomConfiguration(agents=[api.RoomAgentDispatch(agent_name=AGENT_NAME)]))
    )
    # The browser may need a different address than the agent: on Colab the agent uses
    # ws://127.0.0.1:7880 while the browser comes in through a tunnel (wss://...).
    public_url = os.getenv("LIVEKIT_PUBLIC_URL") or url
    return {"server_url": public_url, "participant_token": token.to_jwt()}


@app.get("/api/health")
def health():
    return {"ok": True}


# Phase 1 test page, kept for learning.
@app.get("/phase1")
def phase1_page():
    return FileResponse(ROOT / "steps" / "01_client.html")


# The React app (built into frontend/dist) is served last, so /api/* routes win.
DIST = ROOT / "frontend" / "dist"
if DIST.is_dir():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
