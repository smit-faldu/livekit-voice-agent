"""Phase 5a: the full voice loop BY HAND: VAD -> STT -> LLM -> TTS -> speaker.

Same as Phase 4, plus: the agent publishes its own audio track and speaks the
reply sentence by sentence. Compare with agent/main.py (Phase 5b), where
AgentSession does all of this (and interruptions, transcripts, state) for us.
Run: uv run steps/05_tts.py dev   (then join from http://localhost:3000)
"""

import asyncio
import logging
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from livekit import agents, rtc
from livekit.agents import AgentServer, JobContext, JobProcess, vad
from livekit.plugins import silero

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.graph import build_graph
from agent.stt_local import WhisperSTT
from agent.tts_local import PiperTTS

load_dotenv(".env.local")
logger = logging.getLogger("phase5a")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    proc.userdata["stt"] = WhisperSTT()
    proc.userdata["tts"] = PiperTTS()


server = AgentServer(setup_fnc=prewarm, initialize_process_timeout=120)


@server.rtc_session(agent_name="assistant")
async def entrypoint(ctx: JobContext):
    vad_model, stt, tts = (ctx.proc.userdata[k] for k in ("vad", "stt", "tts"))
    graph = build_graph()
    history: list[BaseMessage] = []
    turn_lock = asyncio.Lock()

    # Our "mouth": an audio source we push PCM frames into, published as a track after connect.
    source = rtc.AudioSource(tts.sample_rate, 1)

    async def speak(text: str):
        # tts.synthesize() yields ready-made rtc.AudioFrames; capture_frame paces them in real time.
        async for audio in tts.synthesize(text):
            await source.capture_frame(audio.frame)

    async def answer(ev: vad.VADEvent):
        text = (await stt.recognize(ev.frames)).alternatives[0].text
        if not text:
            return
        async with turn_lock:
            logger.info("🧑 %s", text)
            history.append(HumanMessage(text))
            reply, buffer = "", ""
            async for chunk, _ in graph.astream({"messages": history}, stream_mode="messages"):
                reply += chunk.text
                buffer += chunk.text
                # Speak each complete sentence as soon as it arrives, don't wait for the whole reply.
                *done, buffer = SENTENCE_END.split(buffer)
                for sentence in done:
                    logger.info("🔊 %s", sentence)
                    await speak(sentence)
            if buffer.strip():
                logger.info("🔊 %s", buffer)
                await speak(buffer)
            history.append(AIMessage(reply))

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

    # Register handlers BEFORE connect: tracks already in the room get subscribed during connect().
    @ctx.room.on("track_subscribed")
    def on_track(track: rtc.Track, pub: rtc.RemoteTrackPublication, p: rtc.RemoteParticipant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(listen(track))

    await ctx.connect()
    track = rtc.LocalAudioTrack.create_audio_track("agent-voice", source)
    await ctx.room.local_participant.publish_track(track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))


if __name__ == "__main__":
    agents.cli.run_app(server)
