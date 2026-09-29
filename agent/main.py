"""The voice assistant: LiveKit AgentSession wiring all local pieces together.

  mic -> Silero VAD -> turn detector -> WhisperSTT -> LangGraph (Gemini) -> PiperTTS -> speaker

Everything runs on this machine except the Gemini call (text only).
Normally started by `server.py` (one process for API + frontend + agent).
Standalone: uv run agent/main.py console   (terminal mic/speaker, no browser)
"""

import functools
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ConversationItemAddedEvent,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    inference,
)
from livekit.agents.llm import ChatMessage
from livekit.plugins import langchain, silero

sys.path.insert(0, str(Path(__file__).parent.parent))  # `agent.` imports when run as a script
from agent.graph import build_graph
from agent.stt_local import WhisperSTT
from agent.tts_local import PiperTTS

load_dotenv(".env.local")
logger = logging.getLogger("assistant")
AGENT_NAME = "assistant"  # the token endpoint dispatches this name


@functools.cache
def load_models() -> dict:
    """Load every local model once per process (~0.6 GB RAM) and share it across sessions."""
    return {
        # Turn detector needs VAD silence >= 0.25 s. It decides "finished?" from how you
        # sound, so VAD can stop waiting early and the model confirms the end of turn.
        "vad": silero.VAD.load(min_silence_duration=0.3),
        "stt": WhisperSTT(model="base.en"),
        "tts": PiperTTS(voice="en_US-lessac-low"),
    }


def prewarm(proc: JobProcess):
    proc.userdata.update(load_models())


server = AgentServer(
    setup_fnc=prewarm,
    initialize_process_timeout=120,
    num_idle_processes=1,  # jobs run as threads in this process; keep only one warm runner (RAM)
    # Production default refuses new jobs above 70% CPU; a busy laptop hits that easily.
    # Single-user local app, so always accept.
    load_threshold=float("inf"),
)


def log_turn_latency(ev: ConversationItemAddedEvent):
    """Per-turn latency breakdown, from the metrics LiveKit attaches to each chat message."""
    if not isinstance(ev.item, ChatMessage):
        return
    m, text = ev.item.metrics, (ev.item.text_content or "")[:60]
    if ev.item.role == "user":
        logger.info(
            "🧑 %r | end-of-turn %.2fs, STT %.2fs",
            text, m.get("end_of_turn_delay", 0), m.get("transcription_delay", 0),
        )
    elif ev.item.role == "assistant" and "e2e_latency" in m:
        logger.info(
            "🤖 %r | LLM first token %.2fs, TTS first audio %.2fs, total (you stop -> it speaks) %.2fs",
            text, m.get("llm_node_ttft", 0), m.get("tts_node_ttfb", 0), m["e2e_latency"],
        )


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=ctx.proc.userdata["stt"],
        tts=ctx.proc.userdata["tts"],
        turn_handling=TurnHandlingOptions(
            # Pinned to v1-mini: runs on local CPU. Unpinned, dev mode would pick cloud "v1".
            turn_detection=inference.TurnDetector(version="v1-mini"),
            endpointing={"min_delay": 0.3, "max_delay": 2.5},  # wait 0.3 s if sure you're done, up to 2.5 s if unsure
            interruption={"mode": "vad"},  # barge-in via VAD ("adaptive" is cloud-only)
            preemptive_generation={"enabled": False},  # one Gemini call per turn
        ),
    )
    session.on("conversation_item_added", log_turn_latency)

    # System prompt, tools and model live in the LangGraph graph, so Agent instructions stay empty.
    # stream_mode="custom": speak only what the graph explicitly writes (never raw tool output).
    agent = Agent(instructions="", llm=langchain.LLMAdapter(graph=build_graph(), stream_mode="custom"))

    await session.start(agent=agent, room=ctx.room)
    await session.say("Hi! I'm your local voice assistant. Ask me anything.")


if __name__ == "__main__":
    agents.cli.run_app(server)
