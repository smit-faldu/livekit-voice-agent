"""Phase 2: Voice Activity Detection (VAD) with Silero.

Agent listens to the user's mic and prints when speech starts and ends.
Run: uv run steps/02_vad.py dev   (then join from http://localhost:3000)
"""

import asyncio
import logging

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import AgentServer, JobContext, JobProcess, vad
from livekit.plugins import silero

load_dotenv(".env.local")
logger = logging.getLogger("phase2")


def prewarm(proc: JobProcess):
    # Runs once per worker process, before any job. Loading the ONNX model here
    # means the first user doesn't wait for it.
    proc.userdata["vad"] = silero.VAD.load(
        activation_threshold=0.5,   # probability above this = speech (raise it in noisy rooms)
        min_speech_duration=0.05,   # ignore blips shorter than 50 ms
        min_silence_duration=0.55,  # this much silence ends the speech segment
        prefix_padding_duration=0.5,  # keep 0.5 s before speech start so first word isn't clipped
    )


server = AgentServer(setup_fnc=prewarm)


async def run_vad(track: rtc.Track, vad_model: vad.VAD, who: str):
    # Silero needs 16 kHz mono. The SDK resamples the 48 kHz WebRTC audio for us.
    audio = rtc.AudioStream(track, sample_rate=16000, num_channels=1)
    stream = vad_model.stream()

    async def feed():
        async for event in audio:  # each event.frame = ~10 ms of PCM samples
            stream.push_frame(event.frame)

    feeder = asyncio.create_task(feed())
    ticks = 0
    async for ev in stream:
        if ev.type == vad.VADEventType.START_OF_SPEECH:
            logger.info("🎤 %s START speaking", who)
        elif ev.type == vad.VADEventType.END_OF_SPEECH:
            # ev.frames holds the whole utterance audio: exactly what Phase 3 sends to STT.
            samples = sum(f.samples_per_channel for f in ev.frames)
            logger.info("🔇 %s END speaking: %.2fs speech, %d samples captured", who, ev.speech_duration, samples)
        elif ev.type == vad.VADEventType.INFERENCE_DONE:
            ticks += 1
            if ev.speaking and ticks % 8 == 0:  # live probability meter while talking
                logger.info("   p=%.2f %s", ev.probability, "█" * int(ev.probability * 20))
    feeder.cancel()


@server.rtc_session(agent_name="assistant")
async def entrypoint(ctx: JobContext):
    vad_model = ctx.proc.userdata["vad"]

    @ctx.room.on("track_subscribed")
    def on_track(track: rtc.Track, pub: rtc.RemoteTrackPublication, p: rtc.RemoteParticipant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info("listening to %s", p.identity)
            asyncio.create_task(run_vad(track, vad_model, p.identity))

    await ctx.connect()


if __name__ == "__main__":
    agents.cli.run_app(server)
