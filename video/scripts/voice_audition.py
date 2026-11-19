"""A/B audition of voice-cloning engines against the reels' Chatterbox setup, on lines Chatterbox gets wrong.

    .tts-venv/Scripts/python  scripts/voice_audition.py render chatterbox     CPU, the reels' settings
    .tts2-venv/Scripts/python scripts/voice_audition.py render qwen3          GPU, Qwen3-TTS 1.7B Base
    .tts2-ming-venv/Scripts/python scripts/voice_audition.py render ming      GPU, Ming-3.0-tts-0.5B
    .tts2-ming-venv/Scripts/python scripts/voice_audition.py render ming-short  the same from an 8.7 s reference
    .tts-venv/Scripts/python  scripts/voice_audition.py check                 Whisper QA of every take
    python                    scripts/voice_audition.py heard                 large Whisper's transcript of every take
    python                    scripts/voice_audition.py readme                out/voice-audition/README.md + listen.html and listen.html

Takes land in out/voice-audition/<engine>/<line>-<take>.wav (git-ignored: they're Martin's cloned voice).
Every engine clones voice-ref/martin-en-cut.wav. Qwen3 and Ming also need its transcript, which `render`
reads from out/voice-audition/ref.txt (large Whisper's transcript of the reference, so it stays out of git).
A GPU engine waits for free VRAM first: the local LLM (Strata) holds about 10 GB and unloads after 15 s idle.
"""
import json
import os
import subprocess
import sys
import time
import zlib

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(ROOT, "out", "voice-audition")
REF = os.path.join(ROOT, "voice-ref", "martin-en-cut.wav")
VIDEOS = os.path.normpath(os.path.join(ROOT, "..", "..", "Videos"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# (key, text, takes, what to listen for). The first three are voice_samples.py's fixed lines; the rest are
# lines Chatterbox got wrong in every take in recent reels, so they get three takes each.
LINES = [
    ("fixed-1", "And here's the weird part: it's speeding up.", 1, ""),
    ("fixed-2", "Standard Lambda C D M nails it. And the node model? Right on top of it.", 1, ""),
    ("fixed-3", "One last, wild what-if. Pure speculation: no simulation behind this one.", 1, ""),
    ("set-off", "The oldest light we can see set off before the Sun existed.", 3,
     "axis-of-evil: Chatterbox said \"set up\" in 4 takes"),
    ("checks", "Checks so far haven't caught either.", 3, "axis-of-evil: Chatterbox said \"Check so far\" in every take"),
    ("pull", "The Moon is close, so its pull is about seven percent stronger on the near side.", 3,
     "tides-style noun \"pull\": Chatterbox said \"pole\" or \"poll\" in every take"),
]
# The reels' voice (reel.py VOICE_SETTINGS), so the Chatterbox column is what a reel sounds like today.
CHATTERBOX = {"CB_EXAGGERATION": "0.6", "CB_CFG": "0.3", "CB_PROMPT": REF}
ENGINES = {
    "chatterbox": {"name": "Chatterbox (current reels)", "venv": ".tts-venv", "device": "CPU",
                   "licence": "MIT (code and weights, ResembleAI/chatterbox card)", "need_mb": 0},
    "qwen3": {"name": "Qwen3-TTS 12Hz 1.7B Base", "venv": ".tts2-venv", "device": "GPU, bf16, SDPA attention",
              "licence": "Apache 2.0 (Qwen/Qwen3-TTS-12Hz-1.7B-Base card and QwenLM/Qwen3-TTS repo)", "need_mb": 6000},
    "ming": {"name": "Ming-3.0-tts 0.5B", "venv": ".tts2-ming-venv", "device": "GPU",
             "licence": "MIT (inclusionAI/Ming-3.0-tts-0.5B card and inclusionAI/Ming-3.0-tts repo)", "need_mb": 5000},
    # Ming cloned from the full 19 s reference dropped the first words of most lines, so this one clones from its
    # last 8.7 s ("So what happens is ... to fix your hiccup."), cut by ffmpeg into out/voice-audition/ref-short.wav.
    "ming-short": {"name": "Ming-3.0-tts 0.5B, 8.7 s reference", "venv": ".tts2-ming-venv", "device": "GPU",
                   "licence": "MIT (as above)", "need_mb": 5000, "ref": "ref-short"},
}


def takes():
    for key, text, n, _ in LINES:
        for t in range(1, n + 1):
            yield key, text, t, f"{key}-{t}.wav"


def seed(text, take):
    return zlib.crc32(f"{text}#{take}".encode())


def wait_for_room(need_mb, wait_s=600):
    """Block until the GPU has need_mb free and no dashboard job is running in the Videos lab queue."""
    sys.path.insert(0, os.path.join(VIDEOS, ".claude", "skills", "famous-reel-editor", "scripts"))
    from reelkit import free_vram
    t0, said = time.time(), False
    while True:
        jobs = subprocess.run(["python", "-m", "lab", "jobs"], cwd=VIDEOS, capture_output=True, text=True).stdout
        busy = any(l.split()[1:2] == ["running"] for l in jobs.splitlines() if l.strip())
        free = free_vram()
        if not busy and (free is None or free >= need_mb):
            return free
        if time.time() - t0 > wait_s:
            sys.exit(f"gave up after {wait_s} s: {'a lab job is running' if busy else f'only {free} MB free'}")
        if not said:
            print(f"waiting: {'a lab job is running; ' if busy else ''}{free} MB free, need {need_mb}", flush=True)
            said = True
        time.sleep(5)


def engine_chatterbox():
    os.environ.update(CHATTERBOX)
    sys.path.insert(0, os.path.join(VIDEOS, ".claude", "skills", "famous-reel-editor", "scripts"))
    import lean_load, tts_worker
    lean_load.patch()
    say = tts_worker.chatterbox_engine()

    def run(text, take, out):
        say(text, out, "" if take == 1 else f"#{take - 1}")  # take 1 = the seed a reel's first recording uses
    return run


def engine_qwen3(ref_text):
    import torch
    from qwen_tts import Qwen3TTSModel
    from tts_worker import save
    model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0",
                                          dtype=torch.bfloat16, attn_implementation="sdpa")
    prompt = model.create_voice_clone_prompt(ref_audio=REF, ref_text=ref_text, x_vector_only_mode=False)

    def run(text, take, out):
        torch.manual_seed(seed(text, take))
        wavs, sr = model.generate_voice_clone(text=text, language="English", voice_clone_prompt=prompt)
        save(out, wavs[0], sr)
    return run


def engine_ming(ref_text, ref=REF):
    import torch
    repo = os.environ.get("MING_REPO", os.path.join(ROOT, ".tts2-ming-venv", "src", "Ming-3.0-tts"))
    sys.path.insert(0, repo)
    os.chdir(repo)  # its modules read a few files by relative path
    import soundfile as sf
    import torchaudio

    def sf_load(path, *a, **k):  # torchaudio 2.9+ loads through torchcodec, which needs FFmpeg's shared DLLs
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(x.T.copy()), sr
    torchaudio.load = sf_load
    from ming_tts import MingTTS
    from tts_worker import save
    # text_normalization=False: its normaliser needs pynini, which has no reliable Windows wheel, and these
    # lines have no digits or symbols to normalise. raw_text=True sends each line whole, as the published
    # benchmarks did, instead of splitting it into fragments of about 24 tokens.
    model = MingTTS(os.environ.get("MING_MODEL", "inclusionAI/Ming-3.0-tts-0.5B"), text_normalization=False)

    def run(text, take, out):
        torch.manual_seed(seed(text, take))
        wav = model.synthesize(task="TTS", language_id="English", text=text, prompt_wav_path=ref,
                               prompt_text=ref_text, use_spk_emb=True, raw_text=True)
        save(out, wav.float().cpu().numpy(), model.sample_rate)
    return run


def render(engine):
    import soundfile as sf
    spec = ENGINES[engine]
    d = os.path.join(OUT, engine)
    if os.path.exists(os.path.join(d, "stats.json")) or os.path.exists(os.path.join(d, "rendering")):
        print(f"{engine}: already rendered or rendering (delete {d} to redo)")
        return
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "rendering"), "w").close()
    ref_name = spec.get("ref", "ref")  # a cut of the reference lives in OUT as <ref>.wav + <ref>.txt
    ref_text = open(os.path.join(OUT, ref_name + ".txt"), encoding="utf-8").read().strip()
    ref = REF if ref_name == "ref" else os.path.join(OUT, ref_name + ".wav")
    free_before = wait_for_room(spec["need_mb"], wait_s=3600)  # Chatterbox too: a reel job may be using the CPU voice
    t0 = time.time()
    run = {"chatterbox": engine_chatterbox, "qwen3": lambda: engine_qwen3(ref_text),
           "ming": lambda: engine_ming(ref_text), "ming-short": lambda: engine_ming(ref_text, ref)}[engine]()
    load_s = time.time() - t0
    stats = {"engine": engine, "load_s": round(load_s, 1), "free_mb_before": free_before, "takes": {}}
    for key, text, take, fname in takes():
        out = os.path.join(d, fname)
        t1 = time.time()
        run(text, take, out + ".part.wav")
        os.replace(out + ".part.wav", out)
        el = time.time() - t1
        x, sr = sf.read(out)
        stats["takes"][fname] = {"key": key, "text": text, "gen_s": round(el, 1), "audio_s": round(len(x) / sr, 2)}
        print(f"{fname}: {el:.1f} s for {len(x) / sr:.1f} s of audio", flush=True)
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.max_memory_reserved():
            stats["peak_vram_mb"] = round(torch.cuda.max_memory_reserved() / 2**20)
            stats["peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
            stats["torch"] = torch.__version__
    except ImportError:
        pass
    json.dump(stats, open(os.path.join(d, "stats.json"), "w"), indent=1)
    os.remove(os.path.join(d, "rendering"))


def check():
    """voice_check.listen (Whisper base.en, the reels' QA) on every take, with voice_qa.py's spelling list."""
    import voice_check as vc
    from transformers import pipeline
    vc.SAME.update({"it's": "its"})
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device="cpu")
    for engine in ENGINES:
        d = os.path.join(OUT, engine)
        if not os.path.exists(os.path.join(d, "stats.json")):
            continue
        res = {}
        for key, text, take, fname in takes():
            heard, sim, problem = vc.listen(asr, os.path.join(d, fname), text)
            res[fname] = {"heard": heard.strip(), "sim": round(sim, 3), "problem": problem}
            print(f"{engine} {fname} {sim:.2f} {problem or 'ok'} | {heard.strip()}", flush=True)
        json.dump(res, open(os.path.join(d, "check.json"), "w"), indent=1)


def heard():
    """large-v3 transcript of every take (base.en's flags were wrong more often than right on the reels)."""
    from faster_whisper import WhisperModel
    m = WhisperModel("large-v3", device="cpu", compute_type="int8")
    for engine in ENGINES:
        d = os.path.join(OUT, engine)
        if not os.path.exists(os.path.join(d, "stats.json")):
            continue
        res = {}
        for key, text, take, fname in takes():
            segs, _ = m.transcribe(os.path.join(d, fname), language="en", beam_size=5)
            res[fname] = " ".join(s.text.strip() for s in segs)
            print(f"{engine} {fname} | {res[fname]}", flush=True)
        json.dump(res, open(os.path.join(d, "heard-large.json"), "w"), indent=1)


def readme():
    def load(engine, name):
        p = os.path.join(OUT, engine, name)
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}

    L = ["# Voice audition: Chatterbox vs newer cloning models", "",
         "Every engine clones `voice-ref/martin-en-cut.wav`. Written by `scripts/voice_audition.py readme`; "
         "the audio is Martin's cloned voice, so this folder is never committed.", "",
         "**Whisper columns**: `sim` and `flag` come from `voice_check.listen` (Whisper base.en, the reels' QA: "
         "similarity to the script, dropped/added words, mid-phrase pauses, rising endings). The reels re-record "
         "below 0.97. `large-v3 heard` is large Whisper's transcript, which is the better judge of the hard words.", ""]
    notes = os.path.join(OUT, "notes.md")  # the hand-written verdict, kept above the tables
    if os.path.exists(notes):
        L += [open(notes, encoding="utf-8").read().strip(), ""]
    L += ["## Lines", "", "| key | line | takes | what to listen for |", "|---|---|---|---|"]
    L += [f"| {k} | {t} | {n} | {w} |" for k, t, n, w in LINES]
    for engine, spec in ENGINES.items():
        st, ck, hl = load(engine, "stats.json"), load(engine, "check.json"), load(engine, "heard-large.json")
        if not st:
            continue
        g = [v["gen_s"] for v in st["takes"].values()]
        a = [v["audio_s"] for v in st["takes"].values()]
        vram = f"{st['peak_vram_mb']} MB reserved ({st['peak_alloc_mb']} MB allocated), torch {st['torch']}" \
            if "peak_vram_mb" in st else "none (CPU)"
        L += ["", f"## {spec['name']}", "",
              f"- Licence as the card states it: {spec['licence']}",
              f"- Runs on: {spec['device']}, venv `video/{spec['venv']}`",
              f"- Peak VRAM: {vram}",
              f"- Speed: model load {st['load_s']} s; {sum(g) / len(g):.1f} s per line on average, "
              f"{sum(g) / sum(a):.2f} s of compute per second of audio", "",
              "| take | sim | flag | large-v3 heard | gen s | audio s |", "|---|---|---|---|---|---|"]
        for fname, v in st["takes"].items():
            c = ck.get(fname, {})
            L.append(f"| `{engine}/{fname}` | {c.get('sim', '')} | {c.get('problem') or ('ok' if c else '')} "
                     f"| {hl.get(fname, '')} | {v['gen_s']} | {v['audio_s']} |")
    L += ["", "## Files to listen to", "",
          "Easiest: open `listen.html` in this folder (one row per take, one player per engine). Or the files:", ""]
    for key, text, take, fname in takes():
        have = [f"`{e}/{fname}`" for e in ENGINES if os.path.exists(os.path.join(OUT, e, fname))]
        L.append(f"- {key} take {take}: " + ", ".join(have))
    open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", os.path.join(OUT, "README.md"))
    listen_page()


def listen_page():
    """listen.html: one row per take, one player per engine, to compare by ear in a browser."""
    import html
    have = [e for e in ENGINES if os.path.exists(os.path.join(OUT, e, "stats.json"))]
    hl = {}
    for e in have:
        p = os.path.join(OUT, e, "heard-large.json")
        hl[e] = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    rows = []
    for key, text, take, fname in takes():
        cells = "".join(f'<td><audio controls preload="none" src="{e}/{fname}"></audio>'
                        f'<div class="h">{html.escape(hl[e].get(fname, ""))}</div></td>' for e in have)
        rows.append(f'<tr><th>{html.escape(text)}<div class="k">{key} take {take}</div></th>{cells}</tr>')
    head = "".join(f"<th>{html.escape(ENGINES[e]['name'])}</th>" for e in have)
    style = ("body{font:14px system-ui;margin:16px;background:#fff;color:#111}table{border-collapse:collapse}"
             "td,th{border:1px solid #ccc;padding:6px;vertical-align:top;text-align:left}th{max-width:260px}"
             ".h{color:#555;font-size:12px;max-width:300px}.k{color:#888;font-weight:normal;font-size:12px}audio{width:240px}")
    page = (f'<!doctype html><meta charset="utf-8"><title>Voice audition</title><style>{style}</style>'
            "<h1>Voice audition</h1><p>Grey text under each player is what large Whisper heard.</p>"
            f'<table><tr><th>line</th>{head}</tr>{"".join(rows)}</table>')
    open(os.path.join(OUT, "listen.html"), "w", encoding="utf-8").write(page)
    print("wrote", os.path.join(OUT, "listen.html"))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "render" and len(sys.argv) > 2 and sys.argv[2] in ENGINES:
        render(sys.argv[2])
    elif cmd in ("check", "heard", "readme"):
        globals()[cmd]()
    else:
        sys.exit(__doc__)
