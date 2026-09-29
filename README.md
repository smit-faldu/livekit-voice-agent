# Local Voice Assistant

A voice assistant you talk to in the browser, built on **LiveKit Agents** and running on your own machine:

| Part | What it uses | Where it runs |
|---|---|---|
| Media transport (WebRTC) | self-hosted `livekit-server` | local |
| Voice activity detection | Silero VAD | local (CPU) |
| Turn detection | LiveKit turn detector `v1-mini` | local (CPU) |
| Speech-to-text | faster-whisper `base.en` | local (CPU) |
| Brain | LangGraph + LangChain → **Gemini** (with tools) | Google API, **text only** |
| Text-to-speech | Piper `en_US-lessac-low` | local (CPU) |
| Web app + token API | React (Vite) + FastAPI | local |

Your audio never leaves your machine. The only outside connection is Gemini, which receives the transcribed text.

```
Browser (React, livekit-client)
  │ 1. POST /api/token (Bearer APP_ACCESS_KEY) ──> server.py :3000  (FastAPI + frontend + agent worker)
  │ 2. WebSocket + WebRTC audio ────────────────> livekit-server :7880
                                                      │ dispatches the job to the agent
                                                      ▼
                       agent: Silero VAD → turn detector → Whisper → LangGraph/Gemini → Piper
```

---

## 1. Prerequisites

Tested on Windows 11 (CPU only, 8 GB RAM). macOS and Linux work with the equivalent commands.

| Tool | Version tested | Install |
|---|---|---|
| **uv** (Python manager) | 0.11 | `powershell -c "irm https://astral.sh/uv/install.ps1 \| iex"` ([docs](https://docs.astral.sh/uv/)) |
| **Python** | 3.14 | installed by uv automatically (`.python-version`) |
| **Node.js** | 24 | https://nodejs.org (needed to build the frontend) |
| **LiveKit server** | 1.13.7 | see step 2 |
| **LiveKit CLI** `lk` (optional) | 2.18 | `winget install LiveKit.LiveKitCLI` |
| **Gemini API key** | – | https://aistudio.google.com/apikey |

Plan for about 2 GB of disk (models plus dependencies) and about 0.7 GB of free RAM while the app runs.

---

## 2. Install the LiveKit server (self-hosted, no LiveKit Cloud)

**Windows:** download the latest `livekit_<version>_windows_amd64.zip` from
https://github.com/livekit/livekit/releases/latest and put `livekit-server.exe` in the project root (it's gitignored) or anywhere on your `PATH`.

**macOS:** `brew install livekit`  **Linux:** `curl -sSL https://get.livekit.io | bash`

Check it: `livekit-server --version`

---

## 3. Get the project and install dependencies

```
git clone <this repo> "livekit voice agent"
cd "livekit voice agent"

uv sync                          # Python deps into .venv
cd frontend && npm install && cd ..   # frontend deps
```

---

## 4. Create `.env.local`

Create `.env.local` in the project root. It's gitignored, so never commit it.

```
# LiveKit server connection. Pick your own key and secret (the secret should be long and random);
# the server is started with the same pair in step 6.
LIVEKIT_URL=ws://127.0.0.1:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=replace-with-a-long-random-secret

# Gemini (the only external service)
GOOGLE_API_KEY=your-gemini-api-key

# Password the browser must send to get a room token
APP_ACCESS_KEY=replace-with-a-random-key
```

Generate random values with:
```
uv run python -c "import secrets; print(secrets.token_urlsafe(32))"
```

---

## 5. Download the local models (once)

`server.py` runs with Hugging Face **offline** (`HF_HUB_OFFLINE=1`) for privacy, so download the models once before the first start:

```
# Speech-to-text: faster-whisper base.en (~145 MB) → models/whisper
uv run python -c "from faster_whisper import WhisperModel; WhisperModel('base.en', download_root='models/whisper')"

# Text-to-speech: Piper voice (~60 MB) → models/piper
uv run python -m piper.download_voices --download-dir models/piper en_US-lessac-low
```

Silero VAD and the turn detector come bundled with the Python packages, so they need no download.

Optional check that both work:
```
uv run agent/stt_local.py some_speech.wav   # prints the transcript + speed
uv run agent/tts_local.py "Hello there"      # writes models/tts_test.wav
```

---

## 6. Run (2 terminals)

**Terminal 1: LiveKit server**, using the key and secret from `.env.local`:
```
livekit-server --dev --keys "devkey: replace-with-a-long-random-secret"
```
(On Windows with the exe in the project root: `.\livekit-server.exe --dev --keys "..."`)

**Terminal 2: the app** (web page + token API + agent, in one process):
```
uv run server.py
```

Open **http://localhost:3000**, enter your `APP_ACCESS_KEY`, click **Start conversation**, and allow microphone access. Talk, or type in the chat box.

- The first run builds the React app (`frontend/dist`). It rebuilds automatically when `frontend/src` changes.
- The first conversation takes a few seconds longer while the models load into RAM.

Things to try: "What time is it?", "What is 17.5 percent of 2480?", then follow-up questions (it remembers the conversation).

---

## Other ways to run

| Goal | Command |
|---|---|
| Talk in the terminal (no browser, no LiveKit server) | `uv run agent/main.py console` |
| Text chat with the brain only (LangGraph + Gemini) | `uv run agent/graph.py` |
| Frontend hot reload while editing UI | `uv run server.py` + `cd frontend && npm run dev` → http://localhost:5173 |
| Learning scripts, one per phase | `uv run steps/0N_*.py dev` (see `path.md`) |

---

## Configuration

| What | Where | Default |
|---|---|---|
| Whisper model (`tiny.en` faster, `small.en` more accurate) | `agent/main.py` → `WhisperSTT(model=...)` | `base.en` |
| Piper voice ([voice list](https://huggingface.co/rhasspy/piper-voices)) | `agent/main.py` → `PiperTTS(voice=...)` | `en_US-lessac-low` |
| Gemini model and system prompt | `agent/graph.py` → `MODEL`, `SYSTEM_PROMPT` | `gemini-3.5-flash-lite` |
| Tools | `agent/graph.py` → `TOOLS` | date/time, calculator |
| Turn timing | `agent/main.py` → `endpointing`, VAD `min_silence_duration` | 0.3–2.5 s, 0.3 s |
| CPU threads per model | `WhisperSTT(cpu_threads=)`, `PiperTTS(threads=)` | 4, 2 |
| Port | `server.py` → `PORT` | 3000 |

To switch Whisper model or Piper voice, download the new model first (step 5), because the server runs offline.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `port 3000 is already in use` | An old server is still running. Stop it (Task Manager → `python.exe`) and start again. |
| Agent never joins / status "failed" | Is `livekit-server` running, and does `--keys` match `.env.local`? Is more than one agent worker running? |
| `401` / "Wrong access key" | The browser key must equal `APP_ACCESS_KEY` in `.env.local`. |
| Mic button crossed out | Allow microphone access for `localhost:3000` in the browser. |
| Start does nothing | Click again or reload the page (a prefetched token can be stale after a server restart). |
| Slow replies | Plug in the laptop charger (the CPU is throttled on battery), close heavy apps, try `tiny.en`. The per-turn latency is printed in the server log. |
| `LocalEntryNotFoundError` / model not found | Run the downloads in step 5. |
| Gemini `503 Service Unavailable` | Google is overloaded; the client retries automatically. |

---

## Project layout

```
server.py          single entry point: FastAPI + built frontend + agent worker (one process)
agent/
  main.py          AgentSession: VAD, turn detector, STT, LLM, TTS, latency logs
  graph.py         LangGraph brain: Gemini + tools (spoken text via get_stream_writer)
  stt_local.py     faster-whisper as a LiveKit STT plugin
  tts_local.py     Piper as a LiveKit TTS plugin
backend/api.py     /api/token (auth, 10-min tokens, agent dispatch), serves frontend/dist
frontend/          React + Vite app (livekit-client, @livekit/components-react)
steps/             learning scripts for phases 1–5
models/            downloaded weights (gitignored)
path.md            learning journal: every phase explained
CLAUDE.md          project rules for Claude Code
```
