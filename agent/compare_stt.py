"""Compare STT models on the same recordings: accuracy (side by side) + speed.

Record your own voice first (the agent saves each utterance):
    $env:SAVE_UTTERANCES="debug/utterances"; uv run server.py      # then talk in the browser

Then compare:
    uv run agent/compare_stt.py debug/utterances
Optional: put the true sentence in <name>.ref.txt next to a wav to get a word error rate.
"""

import re
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.stt_local import WhisperSTT
from agent.stt_streaming import MODELS_DIR, SherpaStreamingSTT, _to_float


def wer(ref: str, hyp: str) -> float:
    """Word error rate: word-level edit distance / number of reference words."""
    r, h = (re.sub(r"[^a-z0-9' ]", " ", t.lower()).split() for t in (ref, hyp))
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / max(len(r), 1)


def main(folder: str):
    wavs = sorted(Path(folder).glob("*.wav"))
    clips = [(p, _to_float(wave.open(str(p)).readframes(10**9))) for p in wavs]
    engines = {f"sherpa:{d.name.replace('sherpa-onnx-', '').replace('streaming-', '').replace('fast-conformer-transducer-', '')}": SherpaStreamingSTT(model=d.name) for d in sorted(MODELS_DIR.iterdir()) if d.is_dir()}
    whisper = WhisperSTT(model="base.en")
    engines["whisper:base.en (batch)"] = whisper
    totals = {name: [0.0, 0.0, 0] for name in engines}  # seconds, wer sum, wer count

    for path, audio in clips:
        ref_file = path.with_suffix(".ref.txt")
        ref = ref_file.read_text(encoding="utf-8").strip() if ref_file.exists() else None
        print(f"\n{path.name} ({len(audio) / 16000:.1f}s)" + (f"  REF: {ref!r}" if ref else ""))
        for name, eng in engines.items():
            t = time.perf_counter()
            text = eng._transcribe(audio, "en")[0] if eng is whisper else eng.transcribe_all(audio)
            totals[name][0] += time.perf_counter() - t
            score = ""
            if ref:
                e = wer(ref, text)
                totals[name][1] += e
                totals[name][2] += 1
                score = f" WER {e:.0%}"
            print(f"  {name:48} {text!r}{score}")

    print("\nTOTAL")
    for name, (secs, w, n) in totals.items():
        print(f"  {name:48} time {secs:.2f}s" + (f", avg WER {w / n:.0%}" if n else ""))


if __name__ == "__main__":
    main(sys.argv[1])
