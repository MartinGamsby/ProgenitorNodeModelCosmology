"""A/B voice audition: the same three narration lines in several engines/settings.

    .tts-venv/Scripts/python scripts/voice_samples.py [variant ...]

Writes out/voice-samples/<variant>.wav (lines joined with short pauses). Variants are defined
below; pick one, then set the matching env vars and run `npm run voice`.
"""
import os
import subprocess
import sys
import json

import numpy as np
import soundfile as sf

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(ROOT, "out", "voice-samples")
LINES = [
    "And here's the weird part: it's speeding up.",
    "Standard Lambda C D M nails it. And the node model? Right on top of it.",
    "One last, wild what-if. Pure speculation: no simulation behind this one.",
]
VARIANTS = {
    "kokoro-george": ("kokoro", {"KOKORO_VOICE": "bm_george"}),
    "kokoro-michael": ("kokoro", {"KOKORO_VOICE": "am_michael"}),
    "kokoro-heart": ("kokoro", {"KOKORO_VOICE": "af_heart"}),
    "chatterbox-0.5": ("chatterbox", {"CB_EXAGGERATION": "0.5", "CB_CFG": "0.5"}),
    "chatterbox-0.8": ("chatterbox", {"CB_EXAGGERATION": "0.8", "CB_CFG": "0.35"}),
    "chatterbox-calm": ("chatterbox", {"CB_EXAGGERATION": "0.5", "CB_CFG": "0.3"}),
    "martin-0.5": ("chatterbox", {"CB_EXAGGERATION": "0.5", "CB_CFG": "0.5", "CB_PROMPT": os.path.join(ROOT, "voice-ref", "martin.wav")}),
    "martin-0.7": ("chatterbox", {"CB_EXAGGERATION": "0.7", "CB_CFG": "0.3", "CB_PROMPT": os.path.join(ROOT, "voice-ref", "martin.wav")}),
    "kokoro-george-slow": ("kokoro", {"KOKORO_VOICE": "bm_george", "KOKORO_SPEED": "0.9"}),
}


def main(names):
    os.makedirs(OUT, exist_ok=True)
    for name in names or VARIANTS:
        engine, env = VARIANTS[name]
        tmp = os.path.join(OUT, name)
        os.makedirs(tmp, exist_ok=True)
        jobs = [{"text": t, "out": os.path.join(tmp, f"{k}.wav")} for k, t in enumerate(LINES)]
        jf = os.path.join(tmp, "jobs.json")
        json.dump(jobs, open(jf, "w"))
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "tts_worker.py"), "--engine", engine, "--jobs", jf],
                       env={**os.environ, **env}, check=True)
        parts, sr = [], None
        for j in jobs:
            x, sr = sf.read(j["out"])
            parts += [x, np.zeros(int(0.6 * sr))]
        sf.write(os.path.join(OUT, f"{name}.wav"), np.concatenate(parts), sr, subtype="PCM_16")
        print("wrote", name)


if __name__ == "__main__":
    main(sys.argv[1:])
