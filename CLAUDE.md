# CLAUDE.md

Project context and rules for Claude. The user's learning journal is `path.md` (theirs, update it at the end of every phase).

## Goal
Fully local voice assistant (general Q&A) built on LiveKit Agents, done phase by phase so the user learns each piece.

## Hard constraints
- **No LiveKit Cloud.** Self-hosted `livekit-server --dev` at `ws://127.0.0.1:7880`, started with the custom key/secret from `.env.local`.
- **No Docker, no speech APIs.** STT and TTS run in-process via official Python libraries:
  - STT (used): `sherpa-onnx` streaming (`agent/stt_streaming.py`), chosen by env `STT_MODEL` (default `nemo80` = NeMo FastConformer 80 ms int8, endpoint 0.5 s; also `nemo480`, `zipformer20m`, `whisper`). Streams primed with 2.5 s silence; reset ONLY after a non-empty endpoint (resets during silence dropped first words). Zipformer 20M was too inaccurate on the user's real voice (audiobook-only training).
  - STT testing: `SAVE_UTTERANCES=<dir>` saves each utterance; `agent/compare_stt.py <dir>` compares all models. Robot/LibriSpeech clips can't reveal accent problems; test on the user's voice.
  - STT fallback: `faster-whisper` base.en batch (`agent/stt_local.py`).
  - TTS: `piper-tts` (`en_US-lessac-low`, ONNX, bundled espeak-ng), wrapped as a custom `livekit.agents.tts.TTS`. Kokoro was tried and dropped: too slow/heavy for this laptop.
  - Model files are downloaded once to `models/` (gitignored).
- **Only external call = Gemini** (text only). LLM logic lives in LangChain + LangGraph, plugged in via `livekit.plugins.langchain.LLMAdapter`.
- **Never use** LiveKit Inference (`inference.STT/LLM/TTS`), BVC noise cancellation, or TurnDetector `v1`. Pin `inference.TurnDetector(version="v1-mini")`.
- Hardware: i5-1235U laptop (2 P + 8 E cores, 15 W), 7.7 GB RAM with ~1 GB free, often on battery (throttled). Python 3.14 via `uv`.
- Low-resource rules: benchmarks swing 10x with free RAM/battery (measure twice); cap threads on every model (sherpa 2, Whisper `cpu_threads=4`, Piper ORT `intra_op=2`, no spinning); smallest viable models; load models once in `prewarm`; measure before choosing.

## Layout
```
server.py  single entry: FastAPI (token API + built frontend) + agent worker in one process
agent/     graph.py (LangGraph + tools), stt_streaming.py (used), stt_local.py (Whisper), compare_stt.py, tts_local.py, main.py (AgentSession)
backend/   api.py (FastAPI: POST /api/token in LiveKit standard format)
frontend/  React + Vite, livekit-client (no hand-written audio websockets)
steps/     01..05 learning scripts, one per phase, kept for reference
models/    downloaded model weights (gitignored)
```

## Phases
All phases 0–8 done (8 = streaming STT + preemptive generation). Current agent config (`agent/main.py`):
- `TurnDetector(version="v1-mini")` (local), VAD `min_silence_duration=0.3`, endpointing 0.3–2.5 s
- `interruption.mode="vad"` (adaptive is cloud-only); preemptive generation ON (Gemini starts on the streamed transcript before turn commit)
- `LLMAdapter(stream_mode="custom")`: only text passed to `get_stream_writer()` in the graph is spoken (tool output never reaches TTS)
- Tools: `get_current_datetime`, `calculator` (AST-based, no eval)
- `AgentServer(num_idle_processes=1, load_threshold=inf)`; models loaded once via `load_models()` (cached), jobs run as threads
- Token endpoint requires `Authorization: Bearer APP_ACCESS_KEY`; server picks identity/room; only agent "assistant"; 10 min TTL
- `HF_HUB_OFFLINE=1` in server.py. Verified: only outbound connection is Google (Gemini).
Full plan: `~/.claude/plans/hey-i-want-to-glimmering-bee.md`.

## Working rules
- One phase at a time. Explain what and why, give a runnable checkpoint, then stop and wait for the user.
- After each phase, append an entry to `path.md`: what was built, key concepts, why it matters, commands, gotchas.
- Verify LiveKit APIs with the livekit-docs MCP before writing code; the SDK changes fast.
- Keep code minimal and readable; this is a learning project.

## Commands
- LiveKit: `livekit-server.exe --dev --keys "<KEY>: <SECRET>"` (binary in repo root, gitignored)
- App: `uv run server.py` → http://localhost:3000 (auto-builds frontend; fails fast if port 3000 busy)
- Agent alone in terminal: `uv run agent/main.py console`
- Frontend hot-reload dev: `cd frontend && npm run dev` (:5173, proxies /api to :3000)
- Testing gotcha: only one worker named "assistant" at a time; kill leftover uvicorn/agent processes (their child python survives killing the parent).
- Debug: `$env:AGENT_LOG_LEVEL="DEBUG"; uv run server.py` shows turn-detector probabilities.
- Env: `.env.local` (gitignored) holds `LIVEKIT_*`, `GOOGLE_API_KEY`, `APP_ACCESS_KEY`.
