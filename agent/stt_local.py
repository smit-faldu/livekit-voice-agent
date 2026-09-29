"""Local speech-to-text: faster-whisper wrapped as a LiveKit STT plugin.

faster-whisper = OpenAI's Whisper model re-implemented on CTranslate2, which is
much faster on CPU (int8 quantization). Runs fully offline after the first
model download into models/whisper.

Self-check / benchmark: uv run agent/stt_local.py path/to/speech.wav
"""

import asyncio
import time
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel
from livekit import rtc
from livekit.agents import APIConnectOptions, stt
from livekit.agents.language import LanguageCode
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, combine_frames

MODELS_DIR = Path(__file__).parent.parent / "models" / "whisper"
WHISPER_RATE = 16000  # Whisper was trained on 16 kHz mono audio


class WhisperSTT(stt.STT):
    def __init__(self, model: str = "base.en", language: str | None = "en", compute_type: str = "int8", cpu_threads: int = 4):
        # streaming=False: Whisper transcribes a finished clip, not a live stream.
        # AgentSession pairs a non-streaming STT with VAD, which cuts the clips (Phase 2).
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self._model_name = model
        self._language = language  # None = auto-detect (slower, less reliable on short clips)
        # cpu_threads caps CPU use (default = all cores) so VAD/TTS/WebRTC keep some headroom.
        self._whisper = WhisperModel(
            model, device="cpu", compute_type=compute_type, cpu_threads=cpu_threads, download_root=str(MODELS_DIR)
        )

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "faster-whisper"

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        audio = _to_whisper_input(buffer)
        lang = language if language is not NOT_GIVEN else self._language
        # Whisper is CPU-heavy and blocking; run it in a thread so the event loop
        # (audio in/out, WebRTC) keeps running while we transcribe.
        text, detected = await asyncio.to_thread(self._transcribe, audio, lang)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language=LanguageCode(detected), text=text)],
        )

    def _transcribe(self, audio: np.ndarray, language: str | None) -> tuple[str, str]:
        segments, info = self._whisper.transcribe(
            audio,
            language=language,
            beam_size=1,                       # greedy decoding: fastest, fine for short utterances
            vad_filter=False,                  # Silero already trimmed the silence
            condition_on_previous_text=False,  # each utterance is independent; avoids repetition loops
        )
        # `segments` is a lazy generator: the real work happens while we iterate it.
        return " ".join(s.text.strip() for s in segments).strip(), info.language


def _to_whisper_input(buffer: AudioBuffer) -> np.ndarray:
    """LiveKit frames (int16 PCM, any rate) -> float32 mono 16 kHz in [-1, 1]."""
    frame = combine_frames(buffer)
    if frame.sample_rate != WHISPER_RATE:
        resampler = rtc.AudioResampler(frame.sample_rate, WHISPER_RATE, num_channels=frame.num_channels)
        frame = combine_frames(resampler.push(frame) + resampler.flush())
    pcm = np.frombuffer(frame.data, dtype=np.int16)
    if frame.num_channels > 1:
        pcm = pcm.reshape(-1, frame.num_channels).mean(axis=1)
    return pcm.astype(np.float32) / 32768.0


if __name__ == "__main__":
    import sys
    import wave

    wav = wave.open(sys.argv[1])
    frame = rtc.AudioFrame(wav.readframes(wav.getnframes()), wav.getframerate(), wav.getnchannels(), wav.getnframes())
    seconds = wav.getnframes() / wav.getframerate()
    for name in sys.argv[2:] or ["base.en"]:
        t = time.perf_counter()
        whisper = WhisperSTT(model=name)
        load = time.perf_counter() - t
        t = time.perf_counter()
        event = asyncio.run(whisper.recognize(frame))
        took = time.perf_counter() - t
        print(f"[{name}] load {load:.1f}s | {seconds:.1f}s audio -> {took:.2f}s (RTF {took / seconds:.2f}) | {event.alternatives[0].text!r}")
