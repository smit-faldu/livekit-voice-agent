"""Local STREAMING speech-to-text: sherpa-onnx streaming Zipformer as a LiveKit STT plugin.

Whisper (stt_local.py) is batch: it waits for the whole utterance, then transcribes
(~1.2 s after you stop). A streaming transducer eats audio in small chunks while you
talk, keeps a running hypothesis, and only has to finish the last few hundred ms when
you stop, so the final transcript arrives almost immediately.

  frames (10 ms) -> buffer 100 ms -> accept_waveform -> decode -> partial text
     partial changed          -> INTERIM_TRANSCRIPT  (live captions)
     endpoint (silence found) -> FINAL_TRANSCRIPT    (the turn's text) -> reset

Models (download once into models/sherpa, see README):
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/<model>.tar.bz2

Self-check / benchmark: uv run agent/stt_streaming.py speech.wav [model_dir_name]
"""

import asyncio
import os
import time
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx
from livekit import rtc
from livekit.agents import APIConnectOptions, stt
from livekit.agents.language import LanguageCode
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, combine_frames

MODELS_DIR = Path(__file__).parent.parent / "models" / "sherpa"
DEFAULT_MODEL = "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
SAMPLE_RATE = 16000
CHUNK = SAMPLE_RATE // 10  # decode every 100 ms of audio (fewer thread hops than per 10 ms frame)
WARMUP = np.zeros(int(2.5 * SAMPLE_RATE), dtype=np.float32)  # see new_stream()
# Debug: set SAVE_UTTERANCES=<dir> to save each utterance as WAV + transcript (to test models on your voice).
SAVE_DIR = os.getenv("SAVE_UTTERANCES")


def _save_utterance(samples: np.ndarray, text: str) -> None:
    out = Path(SAVE_DIR)
    out.mkdir(parents=True, exist_ok=True)
    name = time.strftime("%Y%m%d-%H%M%S")
    with wave.open(str(out / f"{name}.wav"), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    (out / f"{name}.txt").write_text(text, encoding="utf-8")


def _find(model_dir: Path, prefix: str) -> str:
    # Archives ship fp32 and int8 files; prefer int8 (smaller, faster on CPU).
    files = sorted(model_dir.glob(f"{prefix}*.int8.onnx")) or sorted(model_dir.glob(f"{prefix}*.onnx"))
    if not files:
        raise FileNotFoundError(f"no {prefix}*.onnx in {model_dir}: download the model first (README step 5)")
    return str(files[0])


def _pretty(text: str) -> str:
    # The model outputs UPPERCASE without punctuation: "WHAT IS THE TIME"
    text = text.strip().lower()
    return text[:1].upper() + text[1:]


class SherpaStreamingSTT(stt.STT):
    def __init__(self, model: str = DEFAULT_MODEL, threads: int = 2, endpoint_silence: float = 0.3):
        # streaming=True: AgentSession calls stream() and pushes audio continuously,
        # instead of waiting for VAD to cut a finished clip (like with Whisper).
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))
        self._model_name = model
        d = MODELS_DIR / model
        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(d / "tokens.txt"),
            encoder=_find(d, "encoder"),
            decoder=_find(d, "decoder"),
            joiner=_find(d, "joiner"),
            num_threads=threads,  # low-resource rule: cap threads
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            # Endpointing = the model decides "this utterance is over" by itself.
            # AgentSession never flushes the STT, so this is what produces FINAL transcripts.
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,  # silence with no words at all -> endpoint
            rule2_min_trailing_silence=endpoint_silence,  # silence AFTER words -> endpoint (the fast path)
            rule3_min_utterance_length=20.0,  # force an endpoint on very long monologues
        )

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "sherpa-onnx"

    def stream(
        self, *, language: NotGivenOr[str] = NOT_GIVEN, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "SherpaRecognizeStream":
        # sample_rate=16000: the base class resamples whatever arrives (48 kHz WebRTC) for us.
        return SherpaRecognizeStream(stt=self, conn_options=conn_options, sample_rate=SAMPLE_RATE)

    async def _recognize_impl(
        self, buffer: AudioBuffer, *, language: NotGivenOr[str] = NOT_GIVEN, conn_options: APIConnectOptions
    ) -> stt.SpeechEvent:
        # Batch fallback (a whole clip at once); the agent uses stream() instead.
        frame = combine_frames(buffer)
        if frame.sample_rate != SAMPLE_RATE:
            r = rtc.AudioResampler(frame.sample_rate, SAMPLE_RATE)
            frame = combine_frames(r.push(frame) + r.flush())
        text = await asyncio.to_thread(self.transcribe_all, _to_float(frame.data))
        return _event(stt.SpeechEventType.FINAL_TRANSCRIPT, text)

    def new_stream(self):
        """A decoding stream, primed with 2.5 s of silence.

        Measured: a fresh stream outputs nothing for the first ~2 s of audio (the encoder
        needs left context), which cut "What is the capital of Japan" to "Ol of japan".
        Feeding silence first fixes it; the stream then stays warm for the whole call.
        """
        s = self._recognizer.create_stream()
        s.accept_waveform(SAMPLE_RATE, WARMUP)
        self._decode(s)
        return s

    def transcribe_all(self, samples: np.ndarray) -> str:
        s = self.new_stream()
        s.accept_waveform(SAMPLE_RATE, samples)
        s.accept_waveform(SAMPLE_RATE, np.zeros(SAMPLE_RATE // 2, dtype=np.float32))  # tail padding
        s.input_finished()
        self._decode(s)
        return _pretty(self._recognizer.get_result(s))

    def _decode(self, s) -> None:
        while self._recognizer.is_ready(s):
            self._recognizer.decode_stream(s)


class SherpaRecognizeStream(stt.RecognizeStream):
    async def _run(self) -> None:
        stt_: SherpaStreamingSTT = self._stt  # type: ignore[assignment]
        rec = stt_._recognizer
        s = stt_.new_stream()  # one decoding state per conversation (warm, see new_stream)
        pending: list[np.ndarray] = []
        pending_len, last = 0, ""
        utterance: list[np.ndarray] = []  # audio since the last endpoint (only kept if SAVE_DIR)

        def step(samples: np.ndarray) -> tuple[str, bool]:
            # Runs in a worker thread: feed audio, decode what's ready, check for endpoint.
            s.accept_waveform(SAMPLE_RATE, samples)
            stt_._decode(s)
            text = _pretty(rec.get_result(s))
            # Reset only after a real sentence. Resetting during silence (the "2.4 s, no words"
            # rule) throws away warm context, and the first words after it get lost
            # (measured: "And how many people live there" -> "Many people live there").
            ended = bool(text) and rec.is_endpoint(s)
            if ended:
                rec.reset(s)  # start a fresh utterance
            return text, ended

        async for item in self._input_ch:
            if isinstance(item, self._FlushSentinel):
                if last:  # forced end: emit what we have
                    self._event_ch.send_nowait(_event(stt.SpeechEventType.FINAL_TRANSCRIPT, last))
                    last = ""
                continue

            pending.append(_to_float(item.data))
            pending_len += item.samples_per_channel
            if pending_len < CHUNK:
                continue
            samples = np.concatenate(pending)
            pending, pending_len = [], 0
            if SAVE_DIR:
                utterance.append(samples)

            text, ended = await asyncio.to_thread(step, samples)
            if text and text != last and not ended:
                self._event_ch.send_nowait(_event(stt.SpeechEventType.INTERIM_TRANSCRIPT, text))
            if ended and text:
                self._event_ch.send_nowait(_event(stt.SpeechEventType.FINAL_TRANSCRIPT, text))
                if SAVE_DIR:
                    _save_utterance(np.concatenate(utterance), text)
            if ended or not text:
                utterance = [] if ended else utterance[-30:]  # keep ~3 s of lead-in while silent
            last = "" if ended else text


def _to_float(pcm16: bytes | memoryview) -> np.ndarray:
    return np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0


def _event(kind: stt.SpeechEventType, text: str) -> stt.SpeechEvent:
    return stt.SpeechEvent(type=kind, alternatives=[stt.SpeechData(language=LanguageCode("en"), text=text)])


if __name__ == "__main__":
    import sys
    import time
    import wave

    wav = wave.open(sys.argv[1])
    assert wav.getframerate() == SAMPLE_RATE and wav.getnchannels() == 1, "need 16 kHz mono wav"
    audio = _to_float(wav.readframes(wav.getnframes()))
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL

    t = time.perf_counter()
    engine = SherpaStreamingSTT(model)
    print(f"[{model}] load {time.perf_counter() - t:.2f}s")

    # Simulate live streaming: feed 100 ms chunks, then 1 s of silence, and time
    # how long after the speech ends the FINAL transcript appears.
    rec, s = engine._recognizer, engine.new_stream()
    work = 0.0
    for i in range(0, len(audio), CHUNK):
        t = time.perf_counter()
        s.accept_waveform(SAMPLE_RATE, audio[i : i + CHUNK])
        engine._decode(s)
        work += time.perf_counter() - t
    silence = np.zeros(CHUNK, dtype=np.float32)
    for n in range(1, 31):  # up to 3 s
        s.accept_waveform(SAMPLE_RATE, silence)
        engine._decode(s)
        if rec.is_endpoint(s):
            break
    secs = len(audio) / SAMPLE_RATE
    print(f"  {secs:.1f}s audio, CPU while streaming {work:.2f}s (RTF {work / secs:.2f})")
    print(f"  FINAL after {n * 0.1:.1f}s of silence: {_pretty(rec.get_result(s))!r}")
