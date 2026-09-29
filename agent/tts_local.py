"""Local text-to-speech: Piper wrapped as a LiveKit TTS plugin.

Piper = small VITS voices exported to ONNX, built for Raspberry Pi-class hardware.
Text -> phonemes (bundled espeak-ng) -> audio, fully offline, ~60 MB per voice.
Chosen over Kokoro because this laptop has little free RAM and a 15 W CPU:
Kokoro measured RTF 1.8-4 (slower than real time), Piper low RTF ~0.5.

Download voices once:
  uv run python -m piper.download_voices --download-dir models/piper en_US-lessac-low
Browse voices: https://huggingface.co/rhasspy/piper-voices

Self-check / benchmark: uv run agent/tts_local.py "Hello there" [voice]
"""

import asyncio
import json
from pathlib import Path

import onnxruntime as ort

from livekit.agents import APIConnectOptions, tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from piper import PiperVoice
from piper.config import PiperConfig

MODELS_DIR = Path(__file__).parent.parent / "models" / "piper"


class PiperTTS(tts.TTS):
    def __init__(self, voice: str = "en_US-lessac-low", threads: int = 2):
        # "low" voices = 16 kHz, fastest. "medium" = 22.05 kHz, nicer but ~2x slower here.
        self._voice_name = voice
        model_path = MODELS_DIR / f"{voice}.onnx"
        # Build the ONNX session ourselves to cap its threads: by default it grabs
        # every core and starves Whisper, VAD and WebRTC running in the same process.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads, opts.inter_op_num_threads = threads, 1
        # Idle ORT threads normally busy-spin waiting for work, stealing CPU from Whisper.
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        config = PiperConfig.from_dict(json.loads(Path(f"{model_path}.json").read_text(encoding="utf-8")))
        self._voice = PiperVoice(session, config)
        # streaming=False: Piper synthesizes one piece of text at a time. AgentSession
        # splits the LLM's streamed reply into sentences and calls us per sentence,
        # so the first sentence plays while later ones are still being generated.
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=self._voice.config.sample_rate,
            num_channels=1,
        )

    @property
    def model(self) -> str:
        return self._voice_name

    @property
    def provider(self) -> str:
        return "piper"

    def synthesize(self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> tts.ChunkedStream:
        return _PiperStream(tts=self, input_text=text, conn_options=conn_options)

    def generate(self, text: str) -> bytes:
        """Blocking: text -> 16-bit PCM mono bytes at self.sample_rate."""
        return b"".join(chunk.audio_int16_bytes for chunk in self._voice.synthesize(text))


class _PiperStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._tts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",  # raw 16-bit PCM; the emitter slices it into frames
        )
        # CPU-heavy + blocking -> worker thread, so audio playback isn't blocked.
        output_emitter.push(await asyncio.to_thread(self._tts.generate, self.input_text))


if __name__ == "__main__":
    import sys
    import time
    import wave

    voice = sys.argv[2] if len(sys.argv) > 2 else "en_US-lessac-low"
    text = sys.argv[1] if len(sys.argv) > 1 else "Hello there. The capital of Japan is Tokyo."
    t = time.perf_counter()
    piper_tts = PiperTTS(voice)
    print(f"load {time.perf_counter() - t:.2f}s")
    piper_tts.generate("warm up")
    t = time.perf_counter()
    pcm = piper_tts.generate(text)
    took, secs = time.perf_counter() - t, len(pcm) / 2 / piper_tts.sample_rate
    print(f"{secs:.1f}s audio in {took:.2f}s (RTF {took / secs:.2f})")
    out = MODELS_DIR.parent / "tts_test.wav"
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(piper_tts.sample_rate)
        w.writeframes(pcm)
    print("wrote", out, "- play it to hear the voice")
