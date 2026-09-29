# My learning path: local LiveKit voice assistant

My journal. One entry per phase: what I built, the concepts, why they matter, and the commands.

## Big picture

```
Browser (React, livekit-client)
  │ 1. POST /api/token ─────> FastAPI (:3000)          signs a JWT so I may join a room
  │ 2. WebSocket + WebRTC ──> livekit-server (:7880)    routes audio between participants
                                   │ dispatches a job
                                   ▼
                        Agent worker (Python)
                          ├─ VAD:  Silero               "is someone speaking?"
                          ├─ STT:  faster-whisper        speech -> text   (local)
                          ├─ LLM:  LangGraph + Gemini    text -> answer   (only cloud call)
                          ├─ TTS:  Kokoro (onnx)         answer -> speech (local)
                          └─ Turn detector v1-mini       "has the user finished talking?"
```

Voice loop: **hear → detect speech → transcribe → think → speak**.

---

## Phase 0: Setup ✅

**Built**
- `pyproject.toml` deps: `livekit-agents[openai,silero,langchain]`, `langgraph`, `langchain-google-genai`, `fastapi`, `uvicorn`, `livekit-api`, `python-dotenv`.
- `.env.local` holds secrets (gitignored).
- Installed `livekit-server` and got a Gemini key.

**Concepts**
- **livekit-server**: a media router (an SFU). It doesn't understand speech; it just moves audio and video between participants in a *room*.
- **Participant**: anyone in a room. My browser is one participant and the agent is another.
- **Track**: one media stream (for example, my mic audio). Participants *publish* tracks and others *subscribe* to them.
- **API key + secret → JWT token**: the server trusts anyone holding a token signed with the secret. `--dev` mode uses `devkey` / `secret`.
- **Agent worker**: a Python process that registers with the server and gets *dispatched* into rooms to talk to users.

**Why it matters**
- Everything stays on my machine. Only the transcript text goes to Gemini.

**Decisions**
- No Docker. STT and TTS run inside the Python agent using the official libraries: `faster-whisper` and `kokoro-onnx`.
- LangGraph owns the "brain", so I can add tools and memory later without touching the voice code.

**Commands**
```
livekit-server --dev
uv sync
```

---

## Phase 1: Transport + token API ✅

**Built**
- `backend/api.py`: FastAPI `POST /api/token` returns `{server_url, participant_token}` (LiveKit's standard format). `GET /` serves the test page.
- `steps/01_join_room.py`: an agent that joins the room and logs participants and tracks. No AI yet.
- `steps/01_client.html`: a browser test client using `livekit-client` from a CDN.

**Flow (what happens when I click Join)**
1. The browser sends `POST /api/token`. FastAPI signs a JWT with my API secret. The token contains:
   - `video.room` + `roomJoin`: which room I may join.
   - `canPublish` / `canSubscribe`: whether I may send and receive tracks.
   - `roomConfig.agents = [{agentName: "assistant"}]`: an **agent dispatch** request.
   - `exp`: the expiry time. Self-hosted LiveKit can't revoke tokens, so short TTLs matter.
2. The browser runs `room.connect(url, token)`. The SDK opens a **WebSocket** (signaling: join, offer/answer, events) and then **WebRTC** (the actual audio, over UDP). I never write this code.
3. The server creates the room and sees the dispatch request, so it sends a **job** to a worker registered as `"assistant"`.
4. My worker's `entrypoint(ctx)` runs. `ctx.connect()` joins the room as participant `agent-XXXX`.
5. The browser publishes its mic, so the agent gets `track_subscribed` (`KIND_AUDIO`, `SOURCE_MICROPHONE`).

**Concepts**
- **Worker vs job**: the worker is the long-running process that registers with the server (`registered worker` in the log). A job is one room session. One worker can handle many jobs.
- **Explicit dispatch** (`agent_name="assistant"`): the agent joins only rooms that ask for it. Without a name, it would join *every* room.
- **Dispatch happens only when the room is created**, so the API makes a fresh room name per session (`room-xxxxxxxx`).
- **Events**: `participant_connected`, `participant_disconnected`, `track_subscribed`. Everything in LiveKit is event-driven.
- **Modes**: `console` (terminal mic, no server), `dev` (connects to the server, hot reload), `start` (production).
- **Why FastAPI**: the API secret must never reach the browser, so a backend signs the tokens. That is the only job it has here.

**Gotchas**
- My `.env.local` uses custom keys, not `devkey`/`secret`. The server must be started with the same key and secret, or the worker gets an "unauthorized" error.
- The token endpoint has no authentication yet (TODO for Phase 7).

**Commands**
```
livekit-server --dev                                 # terminal 1
uv run uvicorn backend.api:app --port 3000 --reload  # terminal 2
uv run steps/01_join_room.py dev                     # terminal 3
# open http://localhost:3000 → Join + enable mic
```

---

## Phase 2: VAD (Voice Activity Detection) ✅

**Built**
- `steps/02_vad.py`: the agent subscribes to my mic and runs Silero VAD on it. It prints START / END of speech and a live probability meter.

**Pipeline**
```
mic (48 kHz WebRTC) ─> rtc.AudioStream(sample_rate=16000, mono) ─> ~10 ms frames
   ─> vad.stream().push_frame(frame) ─> events: START_OF_SPEECH / INFERENCE_DONE / END_OF_SPEECH
```

**Concepts**
- **Audio frame**: a small chunk of raw PCM samples (16-bit ints). At 16 kHz, 10 ms = 160 samples.
- **Sample rate**: WebRTC uses 48 kHz. Silero only accepts 8 or 16 kHz, so `AudioStream` resamples for me.
- **VAD**: a tiny neural net (Silero, about 2 MB ONNX, runs on CPU) that outputs P(speech) for each window. It knows *that* I speak, not *what* I say.
- **Why VAD comes before STT**:
  1. Whisper is expensive on CPU, so run it only on real speech segments.
  2. VAD marks where each utterance starts and ends, so the STT gets one clean clip per sentence.
  3. Later, VAD also detects **interruptions** (I start talking while the agent speaks).
- **`END_OF_SPEECH.frames`**: the whole utterance audio. This is exactly what Phase 3 sends to Whisper.
- **prewarm (`setup_fnc`)**: loads the model once per worker process, so the first user doesn't wait.

**Knobs (in `prewarm`)**
| Param | Default | Effect |
|---|---|---|
| `activation_threshold` | 0.5 | Probability needed to count as speech. Raise it for noisy rooms. |
| `min_speech_duration` | 0.05 s | Ignores clicks and blips. |
| `min_silence_duration` | 0.55 s | Pause needed to end a segment. Lower = snappier but cuts mid-sentence pauses. |
| `prefix_padding_duration` | 0.5 s | Audio kept before the start, so the first word isn't clipped. |

**Experiment**: set `min_silence_duration=0.2` and pause mid-sentence. One sentence splits into two segments. That problem is why Phase 7 adds the **turn detector** (it understands whether I'm *done*, not just silent).

**Gotchas**
- Only one worker should be registered as `"assistant"` at a time. If an old step script is still running, the server may send the job to it instead. Stop old workers with Ctrl+C.
- The log says `dev` mode is deprecated in favour of `lk agent dev`, and that `agent_name` should move to `livekit.toml`. Both still work. We'll migrate later.

**Commands**
```
uv run steps/02_vad.py dev    # server + API running as in Phase 1
# http://localhost:3000 → Join → talk, pause, talk
```

---

## Phase 3: STT (Speech-to-Text) with faster-whisper ✅

**Built**
- `agent/stt_local.py`: a `WhisperSTT` class, my own LiveKit STT plugin wrapping `faster-whisper`. It runs as a benchmark too: `uv run agent/stt_local.py file.wav tiny.en base.en`.
- `steps/03_stt.py`: VAD (Phase 2) cuts utterances, and each one goes to Whisper. It prints the text and the latency.
- Models downloaded to `models/whisper/` (gitignored).

**Pipeline**
```
mic ─> AudioStream 16 kHz ─> Silero VAD ─> END_OF_SPEECH.frames (one utterance)
    ─> WhisperSTT.recognize(frames) ─> combine frames ─> int16 → float32 [-1,1]
    ─> faster-whisper (CPU, int8, in a thread) ─> "Hello there."
```

**Concepts**
- **Whisper**: OpenAI's open-weights speech recognition model. **faster-whisper** runs the same weights on CTranslate2, about 4x faster on CPU.
- **int8 quantization**: weights stored as 8-bit ints instead of 32-bit floats. That makes them 4x smaller and faster on CPU, with almost no accuracy loss.
- **`.en` models**: English-only. Smaller and more accurate for English than the multilingual version of the same size.
- **Batch vs streaming STT**: Whisper needs the *finished* clip (batch), so it can't show words while I talk. That's why VAD matters: it decides when the clip is finished. Cloud STTs like Deepgram stream partial words instead.
- **LiveKit plugin contract**: subclass `stt.STT`, declare `STTCapabilities(streaming=False)`, implement `_recognize_impl(buffer) -> SpeechEvent(FINAL_TRANSCRIPT)`. That's all `AgentSession` needs (used in Phase 5).
- **`asyncio.to_thread`**: Whisper blocks the CPU for about 0.8 s. Running it in a thread keeps the event loop (WebRTC audio) alive meanwhile.
- **RTF (real-time factor)** = processing time / audio length. It must be well below 1 for conversation.

**Benchmark on my CPU (4.8 s clip)**
| Model | Size | STT time | RTF | Verdict |
|---|---|---|---|---|
| `tiny.en` | 75 MB | 0.48 s | 0.10 | fastest, weaker on accents and noise |
| `base.en` | 145 MB | 0.81 s | 0.17 | **chosen**: good balance |
| `small.en` | 480 MB | 2.41 s | 0.50 | too slow for live conversation on CPU |

**Decoding settings (why)**
- `beam_size=1`: greedy decoding. Fastest, and good enough for short sentences.
- `vad_filter=False`: Silero already removed the silence.
- `condition_on_previous_text=False`: each utterance is separate, which avoids Whisper's repetition loops.
- `language="en"`: skips language detection, which is faster and more reliable on short clips.

**Observed**
- "Hello there. [pause] This is a test…" became **two** transcripts, because VAD split at the pause. Phase 7's turn detector will merge those into one *turn*.
- Latency is about 0.75 s even for short clips, which is fixed overhead. Total "stopped talking → text" time = VAD silence (0.55 s) + STT (about 0.75 s).

**Gotchas**
- The first run downloads the model, which is slow. `prewarm` loads it, so `initialize_process_timeout=60`.
- The Windows "symlinks" warning from huggingface_hub is harmless. It copies files instead of linking them.
- Emoji logs crash only if output is redirected to a file on Windows. Use `PYTHONIOENCODING=utf-8` in that case.
- The `small` (multilingual) model is in `models/whisper` from benchmarking. Delete it to free about 480 MB.

**Commands**
```
uv run steps/03_stt.py dev
uv run agent/stt_local.py some.wav tiny.en base.en    # benchmark models
```

**Q&A: is our STT streaming?**
No, it's **batch**. While I talk, VAD only buffers frames. On `END_OF_SPEECH`, the whole clip goes to Whisper. Wait after I stop ≈ VAD silence (0.55 s) + Whisper (about 0.75 s).
Whisper can't truly stream: it reads a complete clip. "Streaming Whisper" tools just re-run it on a growing window, which costs a lot of CPU.

**Future upgrade: real streaming STT (open source, local)**
| Model | Streaming | CPU | Notes |
|---|---|---|---|
| sherpa-onnx streaming Zipformer | true | excellent | pip `sherpa-onnx`. Best fit for this project. |
| NVIDIA Parakeet / FastConformer streaming | true (cache-aware) | medium/heavy | Top accuracy. Can run through sherpa-onnx. |
| Vosk (Kaldi) | true | very light | Older, lower accuracy. |
| Moonshine | low-latency | good | Edge-focused, fast on short clips. |
| Kyutai STT | true | GPU-leaning | Newer, strong quality. |

To upgrade: write a new `stt.STT` with `streaming=True` that emits `INTERIM_TRANSCRIPT` / `FINAL_TRANSCRIPT` events. The rest of the agent stays the same.

---

## Phase 4: LLM (LangChain + LangGraph + Gemini) ✅

**Built**
- `agent/graph.py`: `build_graph()`, a LangGraph `StateGraph` with one `assistant` node that calls Gemini through LangChain. It also runs as a text chat: `uv run agent/graph.py`.
- `steps/04_llm.py`: speak → VAD → Whisper → graph → the streamed answer is printed, with memory across turns.

**Pipeline**
```
utterance ─> WhisperSTT ─> "What is the capital of Japan?"
   ─> history += HumanMessage
   ─> graph.astream({"messages": history}, stream_mode="messages")
        START ─> [assistant node: SystemMessage + history ─> ChatGoogleGenerativeAI] ─> END
   ─> tokens stream out ─> history += AIMessage
```

**Concepts**
- **LangChain**: a common interface to LLMs (`ChatGoogleGenerativeAI`) plus message types: `SystemMessage` (rules), `HumanMessage` (user), `AIMessage` (bot).
- **LangGraph**: builds the agent as a *graph* of steps.
  - **State** = `TypedDict` with `messages`.
  - **Reducer** `add_messages` = a node's output is *appended* to the message list, not replacing it.
  - **Node** = an async function `state → update`.
  - **Edges** = the order steps run in. Now: `START → assistant`. Later: `assistant ⇄ tools`.
- **Why a graph for a single LLM call?** In Phase 7, tools and multi-step logic plug in as new nodes and edges, and the voice code never changes.
- **Streaming (`stream_mode="messages"`)**: tokens arrive while Gemini generates, so TTS can start speaking the first sentence early. `streaming=True` on the model was **required**; without it, the whole answer came as one chunk.
- **Stateless graph + memory outside**: the graph gets the full history every turn. In Phase 5, LiveKit's `AgentSession` holds the history (chat context), and `LLMAdapter` converts it to LangChain messages.
- **Voice system prompt**: short answers; no markdown, emojis or lists (TTS would read the symbols); numbers written the way they're said ("fourteen million").
- **Thinking level**: Gemini 3 models "think" before answering, which adds delay. `thinking_level="low"` keeps it short.

**Model benchmark (first token)**
| Model | First token | Verdict |
|---|---|---|
| `gemini-3.5-flash-lite` | about 0.9 s | **chosen** |
| `gemini-3.8-flash` (thinking low) | 6.3 s | too slow for voice |
| `gemini-3.5-flash` | 49 s (overloaded that day?) | too slow |
| `gemini-2.5-flash` | error: `thinking_level` not supported (uses `thinking_budget`) | n/a |

**Observed latency**: STT about 0.8 s + LLM about 0.9 s. Follow-up "how many people live there?" correctly understood "there" = Tokyo (memory works).

**Gotchas**
- `thinking_level="minimal"` → 400 error on 3.8-flash. Supported levels differ per model.
- Flash-lite ignores `temperature` (fixed sampling), so I removed it.
- Only *text* goes to Google. Audio never leaves my machine.

**Commands**
```
uv run agent/graph.py            # text chat with the brain, shows latency
uv run steps/04_llm.py dev       # voice in, text answers out
```

---

## Phase 5: TTS + full voice loop ✅

**Built**
- `agent/tts_local.py`: `PiperTTS`, my own LiveKit TTS plugin wrapping **Piper** (`en_US-lessac-low`). Benchmark: `uv run agent/tts_local.py "text"` writes `models/tts_test.wav`.
- `steps/05_tts.py` (**5a, by hand**): VAD → STT → graph → split the reply into sentences → Piper → my own published audio track.
- `agent/main.py` (**5b, the real agent**): the same thing with `AgentSession` in about 30 lines, plus interruptions, transcripts and agent state for free.

**Pipeline (AgentSession)**
```
mic ─> Silero VAD ─> WhisperSTT (batch, per utterance)
    ─> LLMAdapter(LangGraph) ─> Gemini tokens stream
    ─> sentence splitter ─> PiperTTS.synthesize(sentence) ─> 16 kHz PCM ─> agent audio track ─> my speaker
```

**Concepts**
- **TTS plugin contract**: subclass `tts.TTS(streaming=False)`, return a `ChunkedStream` from `synthesize()`, and in `_run()` call `output_emitter.initialize(sample_rate, mime_type="audio/pcm")` then `push(pcm_bytes)`. The framework slices the bytes into frames.
- **Sentence streaming**: TTS doesn't wait for the full LLM reply. Each sentence is spoken as soon as it's complete, so the first sentence plays while Gemini is still writing the rest. (In 5a I do this with a regex; `AgentSession` does it for me.)
- **Publishing audio (5a)**: `rtc.AudioSource` = my "mouth". `LocalAudioTrack` wraps it and `publish_track` sends it to the room. `capture_frame()` paces frames in real time.
- **What `AgentSession` adds over 5a**:
  - turn handling and interruptions: I can talk over the agent and it stops.
  - chat history (`chat_ctx`) and the LangChain conversion.
  - transcripts sent to the frontend.
  - agent state (listening / thinking / speaking).
  - a greeting via `session.say()`.
- **`LLMAdapter`**: turns my LangGraph graph into a LiveKit LLM. `Agent(instructions="")` because the system prompt lives in the graph.
- **Explicit turn settings**: `turn_detection="vad"`, `interruption.mode="vad"`, `preemptive_generation` off. The *defaults* would try LiveKit Cloud models (TurnDetector `v1`, adaptive interruption).

**Low-resource lessons (my laptop: i5-1235U 15 W, 7.7 GB RAM, about 1 GB free)**
| Finding | Numbers | Fix |
|---|---|---|
| Kokoro too heavy | fp32 RTF 1.8–4, int8 RTF 3.5–12 (int8 is *slower* on CPU) | switched to **Piper** |
| Piper low vs medium | low RTF 0.54, medium RTF 1.0 (under load) | `lessac-low` (16 kHz) |
| **Thread cap = biggest win** | Piper RTF 1.05 → **0.05** with `intra_op_num_threads=2` | cap threads on every model |
| Models fighting for cores | VAD fell 9 s behind real time, Whisper 0.8 s → 9 s | Whisper `cpu_threads=4`, Piper 2 threads, no spinning |
| On battery = throttled | CPU at 1.3 GHz, the same Whisper job took 0.9 s vs 2–3 s | plug in the charger + Windows "Best performance" mode |
| RAM pressure | about 270k page faults per Whisper run | close Chrome tabs / other apps while testing |

**RAM budget of the agent process**: Whisper base.en about 300 MB + Piper about 100 MB + Silero about 30 MB + runtime ≈ 0.5–0.6 GB.

**Observed (fake-mic test, on battery)**: from transcript to first spoken word ≈ 1.2 s (Gemini + Piper). The main delay is Whisper when the CPU is throttled.

**Gotchas**
- Register `track_subscribed` **before** `ctx.connect()`. Otherwise tracks that already exist get subscribed during connect and the handler misses them. (Bug I hit in 5a.)
- Only one `"assistant"` worker at a time.
- Start the server with my custom keys: `livekit-server.exe --dev --keys "<KEY>: <SECRET>"`.

**Commands**
```
uv run agent/main.py console      # talk in the terminal (no browser / server needed)
uv run agent/main.py dev          # browser: http://localhost:3000
uv run steps/05_tts.py dev        # the by-hand version, to compare
uv run agent/tts_local.py "Hello there"   # hear/benchmark the voice
```

---

## Phase 6: Custom frontend (React + Vite) ✅

**Built**
- `frontend/`: a Vite + React + TypeScript app using `livekit-client` + `@livekit/components-react`.
  - `src/App.tsx`:
    - Start button.
    - Agent state badge and audio visualizer.
    - Live transcript of both speakers.
    - Typed chat.
    - Mic toggle and End button.
  - `src/index.css`: light and dark theme.
  - `vite.config.ts`: proxies `/api` to FastAPI `:3000`, so the browser sees one origin (no CORS problems).

**Flow when I click "Start conversation"**
```
useSession(TokenSource.endpoint('/api/token'), {agentName: 'assistant'})
 └ session.start()
     1. POST /api/token  {room_config: {agents: [{agentName: "assistant"}]}}   (the SDK adds this)
        → FastAPI parses room_config and signs the JWT → {server_url, participant_token}
     2. connect to livekit-server (WebSocket signaling + WebRTC audio)
     3. publish the mic (microphone: {enabled: true})
     4. the server dispatches "assistant" → the agent joins → greets me
```

**Concepts (the hooks)**
| Hook / component | What it gives me | Where the data comes from |
|---|---|---|
| `useSession` | start/end, connection state | token + room lifecycle |
| `SessionProvider` | room + session for every child (React context) | wraps the conversation UI |
| `useAgent()` | `state`: listening / thinking / speaking, plus the agent's audio track | the agent's `lk.agent.state` participant attribute |
| `useSessionMessages()` | `messages` (spoken transcripts + typed chat) and `send()` | LiveKit **text streams** (`lk.transcription`, `lk.chat`) |
| `RoomAudioRenderer` | plays the agent's voice | subscribed audio tracks |
| `BarVisualizer` | bars that move with the agent's voice and state | the agent's audio track |
| `TrackToggle` | mic on/off | local mic track |

**Key insights**
- **No custom WebSockets needed.** Audio, transcripts, chat and agent state all flow through LiveKit. FastAPI only issues tokens.
- **Typed chat works automatically.** `AgentSession` listens on the `lk.chat` text stream, treats typed text like a spoken turn, and still *speaks* the answer.
- **Autoplay rule**: browsers only allow audio after a user click. That's why the session starts from a button, not on page load.
- **Message types**: `userTranscript` (my speech), `agentTranscript` (agent speech), `chatMessage` (typed; `from.isLocal` tells me whether I sent it).

**Observed**
- Greeting → spoken question transcribed → answer. Typed "What is two plus two?" → "Two plus two is four." Memory across voice and text worked (I named it John and it remembered).
- Gemini once returned `503 Service Unavailable` (Google overloaded). The client retried automatically.
- No browser console errors.

**Gotchas**
- If the mic shows crossed out, allow mic permission for `localhost:5173` in the browser.
- `/api/token` is still unauthenticated (Phase 7).

**Commands (4 terminals)**
```
livekit-server.exe --dev --keys "<KEY>: <SECRET>"
uv run uvicorn backend.api:app --port 3000
uv run agent/main.py dev
cd frontend && npm run dev          # → http://localhost:5173
```

---

## Phase 7: Improvements + one-server setup ✅

### 7.0 Everything in one backend (`server.py`)
**Now only 2 commands:**
```
livekit-server.exe --dev --keys "<KEY>: <SECRET>"
uv run server.py                                   # → http://localhost:3000
```
One Python process holds:
- **Frontend**: `server.py` runs `npm run build` if `frontend/dist` is missing or older than `src/`. FastAPI serves it as static files at `/`.
- **Token API**: `/api/token`, plus `/api/health` and the old `/phase1` test page.
- **Agent worker**: `AgentServer.run()` runs as a background task inside FastAPI's **lifespan**, in the same event loop.

**Concepts**
- **Lifespan**: FastAPI's startup/shutdown hook. It starts the agent worker on startup and closes it cleanly on Ctrl+C.
- **Jobs as threads**: each conversation runs as a thread inside this process, so the models load **once** (`load_models()` with `functools.cache`) and are shared. RAM with everything loaded and a session running: about **700 MB**.
- `num_idle_processes=1`: production mode would keep 12 warm runners, too many for my RAM.
- `load_threshold=inf`: production mode refuses jobs above 70% CPU, and my laptop is often that busy.
- **Fail fast**: `server.py` checks port 3000 first. (I hit a trap twice: an old uvicorn kept port 3000, the new server quietly failed to start, and the *old* API kept answering.)

### 7.1 Turn detector (`v1-mini`, local)
```python
turn_detection=inference.TurnDetector(version="v1-mini")   # pinned → runs locally on CPU
silero.VAD.load(min_silence_duration=0.3)                  # was 0.55
endpointing={"min_delay": 0.3, "max_delay": 2.5}
```
- **VAD vs turn detector**: VAD knows *silence*. The turn detector (an audio model) hears *how* I speak (intonation, rhythm) and predicts "finished?".
  - Sure I'm done → wait only `min_delay` (0.3 s).
  - Unsure → wait up to `max_delay` (2.5 s).
- That's why the VAD silence could drop to 0.3 s. A mid-sentence pause no longer ends my turn, because the model sees I'm not finished.
- Log proof: `eot prediction probability 0.56 > threshold 0.36 → endpointing_delay 0.3`.
- The robotic Windows test voice sometimes scored "unfinished" and waited the full 2.5 s. Real voices score better.
- ⚠️ Without `version="v1-mini"`, dev mode picks cloud `v1`.

### 7.2 Tools in LangGraph
- Tools: `get_current_datetime()` and `calculator(expression)`.
- **Graph**: `START → assistant ⇄ tools → … → END`.
  - `tools_condition` routes to `tools` if Gemini asked for a tool call, otherwise to END.
  - `ToolNode` runs the tools, and the results go back to `assistant` so it can phrase the answer.
- `@tool` + docstring = the schema Gemini sees. The docstring is the model's instruction manual.
- **Calculator safety**: it parses with `ast` and allows only numbers and arithmetic. **Never `eval()` model output**: it would run arbitrary Python.
- **Bug found and fixed: tool results were going to be spoken raw.** `LLMAdapter` in `messages` mode would forward ToolMessages ("2335550.667") to TTS.
  - A `nostream` tag didn't help.
  - Fix: make speech **explicit**. The assistant node streams tokens through `get_stream_writer()`, and `LLMAdapter(stream_mode="custom")` speaks only that. Tool data can never leak into speech.
- Cost: a tool turn = 2 Gemini calls (about 1 s more).
- A "let me check…" filler isn't needed: these tools take milliseconds. The pattern for slow tools: call `say("Let me check.")` from inside the tool node.

### 7.3 Latency tuning + measurement
Every turn is logged with LiveKit's per-message metrics (`ChatMessage.metrics`):
```
🧑 'What is the capital of Japan?' | end-of-turn 1.29s, STT 1.29s
🤖 '...' | LLM first token 3.11s, TTS first audio 0.37s, total (you stop -> it speaks) 6.12s
```
| Stage | What it measures | Typical here | Lever |
|---|---|---|---|
| end-of-turn | I stop → turn committed | 0.3–2.5 s | turn detector confidence, `min/max_delay` |
| STT | I stop → transcript | 1.2–1.5 s (on battery) | whisper `tiny.en`, charger plugged in |
| LLM first token | turn → first Gemini word | 0.9–3.5 s (varies, tools add a call) | flash-lite, short prompt, fewer tool calls |
| TTS first audio | first sentence → sound | 0.4–1.0 s | Piper low, thread cap |
| **e2e** | I stop → agent speaks | **about 6 s on battery** | all of the above |
- STT and end-of-turn overlap: with batch Whisper the turn can't commit before the transcript arrives.
- Biggest remaining wins: plug in the charger (the CPU is throttled on battery), `tiny.en`, and later a streaming STT (see the Phase 3 notes).

### 7.4 Token endpoint security
Rules for `/api/token` (a **trust boundary**: a token = access to a room):
1. **Access key**: `Authorization: Bearer <APP_ACCESS_KEY>` is required.
   - The key is stored in `.env.local`.
   - It's compared with `hmac.compare_digest`, which takes constant time so timing can't leak the key.
   - Missing or wrong key → **401**.
2. **The server decides** identity and room name. The client's `participant_identity: "admin"` is ignored (no impersonation).
3. **Only our agent**: requesting any other agent name → **403**. Other client `room_config` fields are dropped.
4. **TTL 10 minutes**: self-hosted LiveKit can't revoke tokens, so they must expire soon.
5. CORS removed: the frontend is same-origin now.
- Frontend: key field (password input), stored in `localStorage` (wrapped in try/catch).
  - It uses `TokenSource.custom(...)`, which stays stable and reads the key at fetch time.
- **Bug found and fixed**: `useSession` *prefetches* a token on page load. The key variable was still empty then, giving "Wrong access key" and a dead first click. Fixed by loading the key from storage at startup.

### 7.5 Privacy check
- `server.py` sets `HF_HUB_OFFLINE=1`: models are already downloaded, so there are no Hugging Face lookups.
- **Measured**: I watched all TCP connections of the server process during a full voice conversation. Exactly one remote endpoint: `2001:4860:…:443` = **Google** (Gemini).
  - No LiveKit Cloud, no telemetry, no Hugging Face.
  - WebRTC audio only goes to `127.0.0.1` (my own livekit-server).
- How to re-check: `Get-NetTCPConnection -OwningProcess <server.py PID>` while talking.

**Gotchas**
- Only one worker named `"assistant"` at a time. Killing `uvicorn`/`uv run` can leave a child `python.exe` alive, so check the port.
- The first run builds the frontend (needs Node.js). Later runs rebuild only when `frontend/src` changes.
- If Start ever does nothing, click again or reload. The prefetched token may be stale after a server restart.

**Next ideas**
- Streaming STT (sherpa-onnx) to cut the about-1.3 s transcript wait.
- Ollama instead of Gemini for 100% offline.
- More tools, such as notes or reminders.
