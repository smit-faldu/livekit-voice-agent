# Local Voice Assistant

A voice assistant built on LiveKit Agents that runs on your own machine. Speech recognition (faster-whisper), turn detection, and speech synthesis (Piper) all run locally. The only outside call is Gemini, and only text is sent.

## Run (2 terminals)

```
# 1. LiveKit media server (keys from .env.local)
livekit-server.exe --dev --keys "<LIVEKIT_API_KEY>: <LIVEKIT_API_SECRET>"

# 2. Everything else: web app + token API + agent, in one process
uv run server.py
```

Open http://localhost:3000 and enter the access key (`APP_ACCESS_KEY` in `.env.local`).

On the first run, `server.py` builds the React app (this needs Node.js). It rebuilds automatically whenever `frontend/src` changes.

## `.env.local`

```
LIVEKIT_URL=ws://127.0.0.1:7880
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
GOOGLE_API_KEY=...        # Gemini
APP_ACCESS_KEY=...        # the browser must send this to get a room token
```

## Layout

```
server.py      single entry point: FastAPI + built frontend + agent worker
agent/         main.py (AgentSession), graph.py (LangGraph + tools), stt_local.py (Whisper), tts_local.py (Piper)
backend/       api.py (token endpoint, auth, static files)
frontend/      React + Vite app (livekit-client)
steps/         learning scripts for phases 1 to 5
models/        downloaded model weights
path.md        learning journal
```
