"""QA the neural narration: transcribe every clip with Whisper, compare with the script, and
re-generate clips that dropped or garbled words, added words, or paused mid-phrase (new seed each retry).

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

import numpy as np
import soundfile as sf

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

SAME = {"neighbors": "neighbours", "neighbor": "neighbour", "catalog": "catalogue", "disc": "disk",
        "percent": "%", "zeroes": "zeros"}


LANG = "en"


def norm(s):
    from num2words import num2words
    s = s.lower().replace("-", " ").replace(",", "")
    if LANG != "en":  # French elisions: "l’univers" / "l'univers" -> "l univers" on both sides
        s = s.replace("’", " ").replace("'", " ")
    s = re.sub(r"(\d+(?:\.\d+)?)%", r"\1 percent", s)
    out = []
    for w in re.sub(r"[^a-z0-9.'àâäçéèêëîïôöùûüÿœæ ]", " ", s).split():
        w = w.strip(".'")
        if re.fullmatch(r"\d+(\.\d+)?", w):
            out += norm(num2words(float(w) if "." in w else int(w), lang=LANG))
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


def silences(x, sr, min_s=0.45):
    """(midpoint s, duration s) of silences inside the clip (energy below 6% of peak)."""
    fr = int(0.02 * sr)
    e = np.array([np.sqrt(np.mean(x[i:i + fr] ** 2)) for i in range(0, len(x) - fr, fr)])
    quiet, out, start = e < 0.06 * e.max(), [], None
    for i, v in enumerate(quiet):
        if v and start is None:
            start = i
        if not v and start is not None:
            if (i - start) * 0.02 >= min_s and start > 5:
                out.append(((start + i) / 2 * 0.02, (i - start) * 0.02))
            start = None
    return out


def unnatural_pauses(ref, chunks, x, sr):
    """Silences that fall after a script word with no punctuation (e.g. 'call ... whatever').
    Whisper's word timings place each silence after a heard word; difflib maps it to the script."""
    words = re.sub(r"\s+([:;!?»])", r"\1", ref).split()  # French "vérité :" -> "vérité:"
    rn = [" ".join(norm(w)) or "_" for w in words]
    hn = [" ".join(norm(c["text"])) or "_" for c in chunks]
    h2r = {}
    for a, b, n in difflib.SequenceMatcher(None, rn, hn).get_matching_blocks():
        for k in range(n):
            h2r[b + k] = a + k
    bad = []
    for t, d in silences(x, sr):
        idx = max([i for i, c in enumerate(chunks) if c["timestamp"][0] <= t] or [0])
        r = h2r.get(idx)
        if r is not None and r < len(words) - 1 and not re.search(r"[,.;:!?…]$", words[r]):
            bad.append(f"{d:.2f}s pause after '{words[r]}'")
    return "unnatural pause: " + "; ".join(bad) if bad else ""


def hanging_ending(ref, x, sr):
    """A statement should end on a falling pitch. Flags lines ending in '.'/'!' whose pitch over
    the final voiced 0.5 s is flat or rising (sounds unfinished, like 'would allow, ...')."""
    if not re.search(r"[.!]$", ref.strip()):
        return ""
    import librosa
    f0, vf, _ = librosa.pyin(x, fmin=60, fmax=300, sr=sr, frame_length=1024, hop_length=256)
    t = np.arange(len(f0)) * 256 / sr
    v = np.where(vf & ~np.isnan(f0))[0]
    if len(v) < 8:
        return ""
    last = v[t[v] >= t[v[-1]] - 0.5]
    if len(last) < 4:
        return ""
    slope = np.polyfit(t[last], 12 * np.log2(f0[last] / np.median(f0[v])), 1)[0]
    return f"hanging ending (final pitch {slope:+.0f} st/s)" if slope > -3 else ""


def listen(asr, path, ref):
    """Transcribe a clip and return (heard text, similarity, problem)."""
    x, sr = sf.read(path, dtype="float32")
    r = asr(to16k(x, sr), return_timestamps="word")
    sim, problem = check(ref, r["text"])
    return r["text"], sim, problem or unnatural_pauses(ref, r["chunks"], x, sr) or hanging_ending(ref, x, sr)


def to16k(x, sr):
    if sr == 16000:
        return x
    import librosa
    return librosa.resample(x, orig_sr=sr, target_sr=16000)


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
    ap.add_argument("--seed-offset", type=int, default=0, help="start re-roll seeds further on (a second pass tries new takes)")
    a = ap.parse_args()
    from transformers import pipeline
    global LANG
    LANG = os.environ.get("LOCALE", "en")
    model = "openai/whisper-base.en" if LANG == "en" else "openai/whisper-small"  # base mishears French (~half of its flags were false)
    kw = {} if LANG == "en" else {"generate_kwargs": {"language": LANG, "task": "transcribe"}}
    _asr = pipeline("automatic-speech-recognition", model=model, device="cpu")
    asr = lambda x, **k: _asr(x, **k, **kw)
    man = json.load(open(os.path.join(ROOT, "src", "generated", f"voice.{LANG}.json"), encoding="utf-8"))
    vdir = os.path.join(ROOT, "public", "voice", LANG)
    report, bad = [], []
    for key, c in man["clips"].items():
        path = os.path.join(vdir, c["file"])
        hyp, sim, problem = listen(asr, path, c["text"])
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
                say(c["text"], tmp, f"#{a.seed_offset + k + 1}", c.get("exag"), c.get("cfg"))
                hyp, sim, problem = listen(asr, tmp, c["text"])
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
    json.dump(report, open(os.path.join(ROOT, "out", f"voice-check.{LANG}.json"), "w"), indent=1)
    low = [r for r in report if r["sim"] < a.threshold or r.get("problem")]
    print(f"checked {len(report)} clips, {len(bad)} flagged, {len(low)} still below {a.threshold}")


if __name__ == "__main__":
    main()
