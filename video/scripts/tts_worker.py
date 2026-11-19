"""Neural TTS worker for scripts/voice.mjs and the Videos repo's reel.py.

    python scripts/tts_worker.py --engine kokoro|chatterbox|qwen --jobs jobs.json

kokoro and chatterbox run inside video/.tts-venv on the CPU; qwen runs inside video/.tts2-venv on the GPU.
jobs.json = [{"text": ..., "out": "abs/path.wav"}, ...]. Settings come from env vars:
  kokoro:     KOKORO_VOICE (default bm_george), KOKORO_SPEED (1.0)
  chatterbox: CB_EXAGGERATION (0.6), CB_CFG (0.5), CB_PROMPT (optional reference .wav to clone)
  qwen:       QWEN_REF (reference .wav to clone, its transcript in the .txt beside it), QWEN_TEMP (0.6),
              QWEN_LANGUAGE (English), QWEN_POST (optional ffmpeg audio filter run on every clip)
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

    def say(text, out, salt="", exag=None, cfg=None):  # kokoro: deterministic, no expressiveness knobs
        chunks = [a for _, _, a in pipe(text, voice=voice, speed=speed)]
        save(out, np.concatenate([np.asarray(c) for c in chunks]), 24000)

    return say


def chatterbox_engine():
    import torch

    torch.set_num_threads(os.cpu_count() or 4)
    lang = os.environ.get("CB_LANGUAGE", "en")
    if lang == "en":
        from chatterbox.tts import ChatterboxTTS
        model = ChatterboxTTS.from_pretrained(device="cpu")
        extra = {}
    else:  # Chatterbox Multilingual (fr, es, de, …)
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")  # a string: from_local only maps CUDA weights to CPU for device in ["cpu", "mps"]
        extra = {"language_id": lang}
    exag = float(os.environ.get("CB_EXAGGERATION", "0.6"))
    cfg = float(os.environ.get("CB_CFG", "0.5"))
    prompt = os.environ.get("CB_PROMPT") or None

    def say(text, out, salt="", exag_=None, cfg_=None):
        torch.manual_seed(zlib.crc32((text + salt).encode()))  # salt = retry with a new seed
        wav = model.generate(text, audio_prompt_path=prompt,
                             exaggeration=exag if exag_ is None else exag_,
                             cfg_weight=cfg if cfg_ is None else cfg_, **extra)
        save(out, wav.squeeze(0).cpu().numpy(), model.sr)

    return say


def wait_for_vram(need_mb, wait_s=600):
    """The local LLM (Strata) holds about 10 GB of the 12 GB card and unloads after 15 s idle."""
    import subprocess
    t0 = time.time()
    while True:
        try:
            free = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                      capture_output=True, text=True, timeout=20).stdout.split()[0])
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            return
        if free >= need_mb:
            return
        if time.time() - t0 > wait_s:
            sys.exit(f"the GPU has only {free} MB free after {wait_s} s, and the voice needs {need_mb}")
        print(f"waiting for the GPU: {free} MB free, need {need_mb}", flush=True)
        time.sleep(10)


def qwen_engine():
    """Qwen3-TTS 1.7B Base voice clone, picked in voice_audition.py (out/voice-audition, voice-audition.md):
    temperature 0.6 held the accent better than the 0.9 default, and still varied the delivery."""
    import subprocess
    import torch
    from qwen_tts import Qwen3TTSModel

    ref = os.environ["QWEN_REF"]
    ref_text = open(os.path.splitext(ref)[0] + ".txt", encoding="utf-8").read().strip()
    temp = float(os.environ.get("QWEN_TEMP", "0.6"))
    lang = os.environ.get("QWEN_LANGUAGE", "English")
    post = os.environ.get("QWEN_POST") or None
    wait_for_vram(7000)
    # A cap makes PyTorch raise out-of-memory instead of letting Windows spill into system RAM, which hung the PC
    # during the audition (2026-11-18). One line at a time peaks around 6 GB.
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(0.95, 9000 * 2**20 / total))
    model = Qwen3TTSModel.from_pretrained(os.environ.get("QWEN_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
                                          device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    prompt = model.create_voice_clone_prompt(ref_audio=ref, ref_text=ref_text)

    def say(text, out, salt="", exag_=None, cfg_=None):
        torch.manual_seed(zlib.crc32((text + salt).encode()))  # salt = retry with a new seed
        # 12 codec frames per second of audio. A runaway take once ran to 147 s for a 7 s line, so stop at
        # 1 s a word (30 s at least): far past any real line, and QA flags what gets cut.
        cap = 12 * max(30, len(text.split()))
        wavs, sr = model.generate_voice_clone(text=[text], language=[lang], voice_clone_prompt=prompt,
                                              temperature=temp, subtalker_temperature=temp, max_new_tokens=cap)
        save(out, wavs[0], sr)
        if post:  # e.g. the even-out filter: Martin's own reference fades at the ends of sentences, and clones copy it
            tmp = out + ".post.wav"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", out, "-af", post, "-c:a", "pcm_s16le", tmp], check=True)
            os.replace(tmp, out)

    return say


ENGINES = {"kokoro": kokoro_engine, "chatterbox": chatterbox_engine, "qwen": qwen_engine}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=list(ENGINES))
    ap.add_argument("--jobs", required=True)
    a = ap.parse_args()
    jobs = json.load(open(a.jobs, encoding="utf-8"))
    say = ENGINES[a.engine]()
    t0 = time.time()
    for k, j in enumerate(jobs):
        tmp = j["out"] + ".part.wav"
        say(j["text"], tmp, "", j.get("exag"), j.get("cfg"))
        os.replace(tmp, j["out"])  # only finished clips land, so an interrupted run resumes
        el = time.time() - t0
        print(f"[{k + 1}/{len(jobs)}] {el / 60:.1f} min, ~{el / (k + 1) * (len(jobs) - k - 1) / 60:.0f} min left", flush=True)


if __name__ == "__main__":
    sys.exit(main())
