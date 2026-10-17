"""QA the neural narration: transcribe every clip with Whisper, compare with the script, and
re-generate clips that dropped or garbled words (new seed each retry).

    .tts-venv/Scripts/python scripts/voice_check.py [--fix] [--threshold 0.85]

Needs the same env vars as the voice run (CB_EXAGGERATION, CB_CFG, …) when fixing, so the
re-generated clip matches the others. Report: out/voice-check.json.
"""
import argparse
import difflib
import json
import os
import re
import sys

import soundfile as sf

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

def norm(s):
    s = s.lower().replace("-", " ")
    s = re.sub(r"[^a-z0-9' ]", " ", s)
    return [w for w in s.split() if w]


def similarity(ref, hyp):
    """Word-level similarity of the script text vs. what Whisper heard."""
    r, h = norm(ref), norm(hyp)
    return difflib.SequenceMatcher(None, r, h).ratio()


def load_audio(path):
    x, sr = sf.read(path, dtype="float32")
    if sr != 16000:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--retries", type=int, default=4)
    a = ap.parse_args()
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device="cpu")
    man = json.load(open(os.path.join(ROOT, "src", "generated", "voice.json"), encoding="utf-8"))
    vdir = os.path.join(ROOT, "public", "voice")
    report, bad = [], []
    for key, c in man["clips"].items():
        path = os.path.join(vdir, c["file"])
        hyp = asr(load_audio(path))["text"]
        sim = similarity(c["text"], hyp)
        report.append({"key": key, "sim": round(sim, 3), "text": c["text"], "heard": hyp.strip()})
        if sim < a.threshold:
            bad.append((key, c, path))
        print(f"{sim:.2f} {key}", flush=True)
    if a.fix and bad:
        import tts_worker
        say = tts_worker.chatterbox_engine()
        for key, c, path in bad:
            best = None
            for k in range(a.retries):
                tmp = path + f".try{k}.wav"
                say(c["text"], tmp, f"#{k + 1}")
                hyp = asr(load_audio(tmp))["text"]
                sim = similarity(c["text"], hyp)
                print(f"  retry {k + 1} {sim:.2f} {key}", flush=True)
                if best is None or sim > best[0]:
                    best = (sim, tmp, hyp)
                if sim >= a.threshold:
                    break
            os.replace(best[1], path)
            for f in os.listdir(vdir):
                if f.startswith(os.path.basename(path) + ".try"):
                    os.remove(os.path.join(vdir, f))
            for r in report:
                if r["key"] == key:
                    r.update(sim=round(best[0], 3), heard=best[2].strip(), regenerated=True)
    json.dump(report, open(os.path.join(ROOT, "out", "voice-check.json"), "w"), indent=1)
    low = [r for r in report if r["sim"] < a.threshold]
    print(f"checked {len(report)} clips, {len(bad)} flagged, {len(low)} still below {a.threshold}")


if __name__ == "__main__":
    main()
