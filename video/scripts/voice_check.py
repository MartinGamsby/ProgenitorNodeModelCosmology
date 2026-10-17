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

SAME = {"neighbors": "neighbours", "neighbor": "neighbour", "catalog": "catalogue", "disc": "disk",
        "percent": "%", "zeroes": "zeros"}


def norm(s):
    from num2words import num2words
    s = s.lower().replace("-", " ").replace(",", "")
    s = re.sub(r"(\d+(?:\.\d+)?)%", r" percent", s)
    out = []
    for w in re.sub(r"[^a-z0-9.' ]", " ", s).split():
        w = w.strip(".'")
        if re.fullmatch(r"\d+(\.\d+)?", w):
            out += norm(num2words(float(w) if "." in w else int(w)))
        elif w:
            out.append(SAME.get(w, w))
    return out


def check(ref, hyp):
    """(similarity, problem) of script text vs. what Whisper heard. problem flags extra words
    appended at the end (Chatterbox sometimes keeps talking) or a repeated/inserted phrase."""
    r, h = norm(ref), norm(hyp)
    sm = difflib.SequenceMatcher(None, r, h)
    problem = ""
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        extra = j2 - j1 - (i2 - i1)
        if tag in ("insert", "replace") and extra >= 2 and i2 == len(r):
            problem = "extra words at the end: " + " ".join(h[j1:j2])
        elif tag in ("insert", "replace") and extra >= 3:
            problem = "inserted words: " + " ".join(h[j1:j2])
    return sm.ratio(), problem


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
        sim, problem = check(c["text"], hyp)
        report.append({"key": key, "sim": round(sim, 3), "problem": problem, "text": c["text"], "heard": hyp.strip()})
        if sim < a.threshold or problem:
            bad.append((key, c, path))
        print(f"{sim:.2f} {key} {problem}", flush=True)
    if a.fix and bad:
        import tts_worker
        say = tts_worker.chatterbox_engine()
        for key, c, path in bad:
            best = None
            for k in range(a.retries):
                tmp = path + f".try{k}.wav"
                say(c["text"], tmp, f"#{k + 1}")
                hyp = asr(load_audio(tmp))["text"]
                sim, problem = check(c["text"], hyp)
                score = sim - (0.5 if problem else 0)
                print(f"  retry {k + 1} {sim:.2f} {key} {problem}", flush=True)
                if best is None or score > best[0]:
                    best = (score, tmp, hyp, sim, problem)
                if sim >= a.threshold and not problem:
                    break
            os.replace(best[1], path)
            for f in os.listdir(vdir):
                if f.startswith(os.path.basename(path) + ".try"):
                    os.remove(os.path.join(vdir, f))
            for r in report:
                if r["key"] == key:
                    r.update(sim=round(best[3], 3), problem=best[4], heard=best[2].strip(), regenerated=True)
    json.dump(report, open(os.path.join(ROOT, "out", "voice-check.json"), "w"), indent=1)
    low = [r for r in report if r["sim"] < a.threshold or r.get("problem")]
    print(f"checked {len(report)} clips, {len(bad)} flagged, {len(low)} still below {a.threshold}")


if __name__ == "__main__":
    main()
