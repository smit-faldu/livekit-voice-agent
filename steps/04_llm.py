"""Phase 4: VAD + STT + LLM (LangGraph + Gemini).

Speak -> VAD cuts utterance -> Whisper transcribes -> LangGraph/Gemini answers.
The answer streams into the terminal (no voice yet; TTS is Phase 5).
Run: uv run steps/04_llm.py dev   (then join from http://localhost:3000)
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from livekit import agents, rtc
from livekit.agents import AgentServer, JobContext, JobProcess, vad
from livekit.plugins import silero

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.graph import build_graph
from agent.stt_local import WhisperSTT

load_dotenv(".env.local")
logger = logging.getLogger("phase4")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    proc.userdata["stt"] = WhisperSTT()


server = AgentServer(setup_fnc=prewarm, initialize_process_timeout=60)


@server.rtc_session(agent_name="assistant")
async def entrypoint(ctx: JobContext):
    vad_model, stt = ctx.proc.userdata["vad"], ctx.proc.userdata["stt"]
    graph = build_graph()
    history: list[BaseMessage] = []  # conversation memory for this room (AgentSession does this in Phase 5)
    turn_lock = asyncio.Lock()  # answer one utterance at a time

    async def answer(ev: vad.VADEvent):
        t0 = time.perf_counter()
        text = (await stt.recognize(ev.frames)).alternatives[0].text
        t_stt = time.perf_counter() - t0
        if not text:
            return
        async with turn_lock:
            logger.info("🧑 %s   (STT %.2fs)", text, t_stt)
            history.append(HumanMessage(text))
            t1, first, reply = time.perf_counter(), None, ""
            async for chunk, _ in graph.astream({"messages": history}, stream_mode="messages"):
                first = first or time.perf_counter() - t1
                reply += chunk.text
            history.append(AIMessage(reply))
            logger.info("🤖 %s   (LLM first token %.2fs, total %.2fs)", reply, first or 0, time.perf_counter() - t1)

    async def listen(track: rtc.Track):
        audio = rtc.AudioStream(track, sample_rate=16000, num_channels=1)
        stream = vad_model.stream()

        async def feed():
            async for e in audio:
                stream.push_frame(e.frame)

        feeder = asyncio.create_task(feed())
        async for ev in stream:
            if ev.type == vad.VADEventType.END_OF_SPEECH:
                asyncio.create_task(answer(ev))
        feeder.cancel()

    @ctx.room.on("track_subscribed")
    def on_track(track: rtc.Track, pub: rtc.RemoteTrackPublication, p: rtc.RemoteParticipant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(listen(track))

    await ctx.connect()


if __name__ == "__main__":
    agents.cli.run_app(server)
