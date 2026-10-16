"""Neural TTS worker for scripts/voice.mjs (runs inside video/.tts-venv, CPU is fine).

    python scripts/tts_worker.py --engine kokoro|chatterbox --jobs jobs.json

jobs.json = [{"text": ..., "out": "abs/path.wav"}, ...]. Settings come from env vars:
  kokoro:     KOKORO_VOICE (default bm_george), KOKORO_SPEED (1.0)
  chatterbox: CB_EXAGGERATION (0.6), CB_CFG (0.5), CB_PROMPT (optional reference .wav to clone)
Every clip: 16-bit mono WAV, leading/trailing silence trimmed, deterministic seed per text.
"""
import argparse
import json
import os
import sys
import time
import zlib

import numpy as np
import soundfile as sf


def trim(x, sr, thresh=0.01, pad=0.08):
    idx = np.where(np.abs(x) > thresh * max(1e-6, np.abs(x).max()))[0]
    if not len(idx):
        return x
    p = int(pad * sr)
    return x[max(0, idx[0] - p): min(len(x), idx[-1] + p)]


def save(path, x, sr):
    x = trim(np.asarray(x, dtype=np.float32).ravel(), sr)
    x = x / max(1e-6, np.abs(x).max()) * 0.89
    sf.write(path, x, sr, subtype="PCM_16")


def kokoro_engine():
    from kokoro import KPipeline

    voice = os.environ.get("KOKORO_VOICE", "bm_george")
    speed = float(os.environ.get("KOKORO_SPEED", "1.0"))
    pipe = KPipeline(lang_code=voice[0])  # 'a' American, 'b' British

    def say(text, out):
        chunks = [a for _, _, a in pipe(text, voice=voice, speed=speed)]
        save(out, np.concatenate([np.asarray(c) for c in chunks]), 24000)

    return say


def chatterbox_engine():
    import torch
    from chatterbox.tts import ChatterboxTTS

    torch.set_num_threads(os.cpu_count() or 4)
    model = ChatterboxTTS.from_pretrained(device="cpu")
    exag = float(os.environ.get("CB_EXAGGERATION", "0.6"))
    cfg = float(os.environ.get("CB_CFG", "0.5"))
    prompt = os.environ.get("CB_PROMPT") or None

    def say(text, out):
        torch.manual_seed(zlib.crc32(text.encode()))
        wav = model.generate(text, audio_prompt_path=prompt, exaggeration=exag, cfg_weight=cfg)
        save(out, wav.squeeze(0).cpu().numpy(), model.sr)

    return say


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["kokoro", "chatterbox"])
    ap.add_argument("--jobs", required=True)
    a = ap.parse_args()
    jobs = json.load(open(a.jobs, encoding="utf-8"))
    say = {"kokoro": kokoro_engine, "chatterbox": chatterbox_engine}[a.engine]()
    t0 = time.time()
    for k, j in enumerate(jobs):
        tmp = j["out"] + ".part.wav"
        say(j["text"], tmp)
        os.replace(tmp, j["out"])  # only finished clips land, so an interrupted run resumes
        el = time.time() - t0
        print(f"[{k + 1}/{len(jobs)}] {el / 60:.1f} min, ~{el / (k + 1) * (len(jobs) - k - 1) / 60:.0f} min left", flush=True)


if __name__ == "__main__":
    sys.exit(main())
