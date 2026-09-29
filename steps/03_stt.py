"""Phase 3: VAD + local STT (faster-whisper).

VAD cuts the mic audio into utterances (Phase 2); each finished utterance goes
to Whisper and the transcript is printed with its latency.
Run: uv run steps/03_stt.py dev   (then join from http://localhost:3000)
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import AgentServer, JobContext, JobProcess, vad
from livekit.plugins import silero

sys.path.insert(0, str(Path(__file__).parent.parent))  # so `agent.` imports work from steps/
from agent.stt_local import WhisperSTT

load_dotenv(".env.local")
logger = logging.getLogger("phase3")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    proc.userdata["stt"] = WhisperSTT(model="base.en")  # try "tiny.en" (faster) or "small.en" (more accurate)


# Whisper model loading is slow; give the process more than the default 10 s to start.
server = AgentServer(setup_fnc=prewarm, initialize_process_timeout=60)


async def transcribe(stt: WhisperSTT, ev: vad.VADEvent, who: str):
    t = time.perf_counter()
    result = await stt.recognize(ev.frames)
    took = time.perf_counter() - t
    text = result.alternatives[0].text
    logger.info("📝 %s: %r  (%.1fs audio -> %.2fs STT)", who, text, ev.speech_duration, took)


async def listen(track: rtc.Track, ctx: JobContext, who: str):
    vad_model, stt = ctx.proc.userdata["vad"], ctx.proc.userdata["stt"]
    audio = rtc.AudioStream(track, sample_rate=16000, num_channels=1)
    stream = vad_model.stream()

    async def feed():
        async for event in audio:
            stream.push_frame(event.frame)

    feeder = asyncio.create_task(feed())
    async for ev in stream:
        if ev.type == vad.VADEventType.START_OF_SPEECH:
            logger.info("🎤 %s speaking...", who)
        elif ev.type == vad.VADEventType.END_OF_SPEECH:
            # Don't await here: keep reading VAD events while Whisper works.
            asyncio.create_task(transcribe(stt, ev, who))
    feeder.cancel()


@server.rtc_session(agent_name="assistant")
async def entrypoint(ctx: JobContext):
    @ctx.room.on("track_subscribed")
    def on_track(track: rtc.Track, pub: rtc.RemoteTrackPublication, p: rtc.RemoteParticipant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(listen(track, ctx, p.identity))

    await ctx.connect()


if __name__ == "__main__":
    agents.cli.run_app(server)
