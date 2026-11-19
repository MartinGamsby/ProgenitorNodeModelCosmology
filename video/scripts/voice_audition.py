"""A/B audition of voice engines against the reels' Chatterbox setup, on lines Chatterbox gets wrong, in English and French.

    .tts-venv/Scripts/python  scripts/voice_audition.py render chatterbox     CPU, the reels' settings
    .tts2-venv/Scripts/python scripts/voice_audition.py render [variant ...]  GPU, Qwen3-TTS (all its variants if none named)
    .tts-venv/Scripts/python  scripts/voice_audition.py check                 Whisper QA of every take (base.en / small for French)
    python                    scripts/voice_audition.py heard                 large Whisper's transcript of every take
    python                    scripts/voice_audition.py readme                out/voice-audition/README.md + listen.html

Takes land in out/voice-audition/<variant>/<line>-<take>.wav (git-ignored: they're Martin's cloned voice).
A clone variant needs a reference out/voice-audition/<ref>.wav and its transcript <ref>.txt there, also git-ignored
(large Whisper's transcripts of Martin's clips): ref = voice-ref/martin-en-cut.wav, ref-long = the first 47.9 s of
voice-ref/martin-full.wav (ends on "...to fix your hiccup."), ref-fr = voice-ref/martin-fr-cut.wav.
A GPU render waits for free VRAM and an idle lab queue: the local LLM (Strata) holds about 10 GB, unloads after 15 s.
"""
import json
import os
import subprocess
import sys
import time
import zlib

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(ROOT, "out", "voice-audition")
VIDEOS = os.path.normpath(os.path.join(ROOT, "..", "..", "Videos"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# (key, text, takes, what to listen for). The fixed lines are voice_samples.py's (in French, the explainer's own
# fr.ts lines); the rest are lines Chatterbox got wrong in recent reels, so they get three takes each.
LINES = {
    "en": [
        ("fixed-1", "And here's the weird part: it's speeding up.", 1, ""),
        ("fixed-2", "Standard Lambda C D M nails it. And the node model? Right on top of it.", 1, ""),
        ("fixed-3", "One last, wild what-if. Pure speculation: no simulation behind this one.", 1, ""),
        ("set-off", "The oldest light we can see set off before the Sun existed.", 3,
         "axis-of-evil: Chatterbox said \"set up\" in 4 takes"),
        ("checks", "Checks so far haven't caught either.", 3, "axis-of-evil: Chatterbox said \"Check so far\" in every take"),
        ("pull", "The Moon is close, so its pull is about seven percent stronger on the near side.", 3,
         "tides-style noun \"pull\": Chatterbox said \"pole\" or \"poll\" in every take"),
    ],
    "fr": [
        ("fixed-1", "Et voici le plus bizarre. Ça accélère.", 1, "Quebec or France accent?"),
        ("fixed-2", "Lambda cé-dé-emme, le modèle standard, tombe pile dessus.", 1, ""),
        ("fixed-3", "Une dernière hypothèse, complètement folle. Celle-là, c'est de la pure spéculation, "
                    "sans aucune simulation derrière.", 1, ""),
        ("set-off", "La plus vieille lumière qu'on peut voir est partie avant que le Soleil existe.", 3, "the axis-of-evil hook"),
        ("checks", "Jusqu'ici, aucune vérification n'a détecté ni l'un ni l'autre.", 3, ""),
        ("pull", "La Lune est proche, donc son attraction est environ sept pour cent plus forte du côté le plus près.", 3, ""),
    ],
}
# The reels' voice (reel.py VOICE_SETTINGS), so the Chatterbox column is what a reel sounds like today.
CHATTERBOX = {"CB_EXAGGERATION": "0.6", "CB_CFG": "0.3", "CB_PROMPT": os.path.join(ROOT, "voice-ref", "martin-en-cut.wav")}
BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DESIGN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
QWEN_LICENCE = "Apache 2.0 (the model's Hugging Face card, and the QwenLM/Qwen3-TTS repo)"
DESIGN_EN = ("A man in his forties with a Canadian English accent, a warm, clear, slightly low voice, "
             "speaking calmly and with curiosity, like a science documentary narrator.")
DESIGN_FR = ("A man in his forties speaking French with a Quebec (Canadian French) accent, a warm, clear, slightly "
             "low voice, speaking calmly and with curiosity, like a science documentary narrator.")
# temp sets both of Qwen's sampling temperatures (the talker's and the sub-talker's); 0.9 is its default.
# Variants without "batch": False render all their takes in one batched call, as a reel's voice step could.
ENGINES = {
    "chatterbox": {"name": "Chatterbox (current reels)", "lang": "en", "venv": ".tts-venv", "device": "CPU",
                   "licence": "MIT (code and weights, ResembleAI/chatterbox card)", "need_mb": 0},
    "qwen3": {"name": "Qwen3 clone, temp 0.9 (default), one take at a time", "lang": "en", "model": BASE, "ref": "ref",
              "temp": None, "batch": False},
    "qwen3-t07": {"name": "Qwen3 clone, temp 0.7", "lang": "en", "model": BASE, "ref": "ref", "temp": 0.7},
    "qwen3-t05": {"name": "Qwen3 clone, temp 0.5", "lang": "en", "model": BASE, "ref": "ref", "temp": 0.5},
    "qwen3-t03": {"name": "Qwen3 clone, temp 0.3", "lang": "en", "model": BASE, "ref": "ref", "temp": 0.3},
    "qwen3-t05-long": {"name": "Qwen3 clone, temp 0.5, 48 s reference", "lang": "en", "model": BASE, "ref": "ref-long",
                       "temp": 0.5},
    "qwen3-t05-xvec": {"name": "Qwen3 clone, temp 0.5, speaker embedding only", "lang": "en", "model": BASE, "ref": "ref",
                       "temp": 0.5, "xvec": True},
    "design-en": {"name": "Qwen3 VoiceDesign (no cloning), temp 0.5", "lang": "en", "model": DESIGN, "instruct": DESIGN_EN,
                  "temp": 0.5},
    "qwen3-fr": {"name": "Qwen3 clone, French reference, temp 0.9 (default)", "lang": "fr", "model": BASE, "ref": "ref-fr",
                 "temp": None},
    "qwen3-fr-t05": {"name": "Qwen3 clone, French reference, temp 0.5", "lang": "fr", "model": BASE, "ref": "ref-fr",
                     "temp": 0.5},
    "design-fr": {"name": "Qwen3 VoiceDesign (no cloning), Quebec accent asked, temp 0.5", "lang": "fr", "model": DESIGN,
                  "instruct": DESIGN_FR, "temp": 0.5},
}
for _k, _v in ENGINES.items():
    if "model" in _v:
        _v.update(venv=".tts2-venv", device="GPU, bf16, SDPA attention", licence=QWEN_LICENCE, need_mb=7000)
LANGUAGE = {"en": "English", "fr": "French"}
BATCH = 3  # takes per batched call
VRAM_CAP_MB = 9000  # PyTorch's limit on the 12 GB RTX 5070, leaving room for the desktop


def takes(lang="en"):
    for key, text, n, _ in LINES[lang]:
        for t in range(1, n + 1):
            yield key, text, t, f"{key}-{t}.wav"


def seed(text, take):
    return zlib.crc32(f"{text}#{take}".encode())


def wait_for_room(need_mb, wait_s=3600):
    """Block until the GPU has need_mb free and no dashboard job is running in the Videos lab queue."""
    sys.path.insert(0, os.path.join(VIDEOS, ".claude", "skills", "famous-reel-editor", "scripts"))
    from reelkit import free_vram
    t0, said = time.time(), False
    while True:
        jobs = subprocess.run(["python", "-m", "lab", "jobs"], cwd=VIDEOS, capture_output=True, text=True).stdout
        busy = any(l.split()[1:2] == ["running"] for l in jobs.splitlines() if l.strip())
        free = free_vram()
        if not busy and (need_mb == 0 or free is None or free >= need_mb):
            return free
        if time.time() - t0 > wait_s:
            sys.exit(f"gave up after {wait_s} s: {'a lab job is running' if busy else f'only {free} MB free'}")
        if not said:
            print(f"waiting: {'a lab job is running; ' if busy else ''}{free} MB free, need {need_mb}", flush=True)
            said = True
        time.sleep(5)


def ref_files(name):
    return os.path.join(OUT, name + ".wav"), open(os.path.join(OUT, name + ".txt"), encoding="utf-8").read().strip()


def render_chatterbox():
    import soundfile as sf
    d = os.path.join(OUT, "chatterbox")
    if os.path.exists(os.path.join(d, "stats.json")):
        return print("chatterbox: already rendered")
    os.makedirs(d, exist_ok=True)
    free = wait_for_room(0)
    os.environ.update(CHATTERBOX)
    sys.path.insert(0, os.path.join(VIDEOS, ".claude", "skills", "famous-reel-editor", "scripts"))
    t0 = time.time()
    import lean_load, tts_worker
    lean_load.patch()
    say = tts_worker.chatterbox_engine()
    stats = {"engine": "chatterbox", "load_s": round(time.time() - t0, 1), "free_mb_before": free, "takes": {}}
    for key, text, take, fname in takes("en"):
        out, t1 = os.path.join(d, fname), time.time()
        say(text, out + ".part.wav", "" if take == 1 else f"#{take - 1}")  # take 1 = a reel's first-recording seed
        os.replace(out + ".part.wav", out)
        x, sr = sf.read(out)
        stats["takes"][fname] = {"key": key, "text": text, "gen_s": round(time.time() - t1, 1), "audio_s": round(len(x) / sr, 2)}
    json.dump(stats, open(os.path.join(d, "stats.json"), "w"), indent=1)


def render_qwen(names):
    """Each variant's takes in one batched call (or one by one with "batch": False), loading each model once."""
    import torch
    from qwen_tts import Qwen3TTSModel
    from tts_worker import save
    # A batch of all 12 takes peaked at 19.7 GB "reserved" on the 12 GB card: Windows spilled it into system RAM, and
    # the next batch (the 48 s reference) hung the whole PC (2026-11-18, Kernel-Power 41, no bugcheck). So: small
    # batches, and a cap that makes PyTorch raise out-of-memory instead of spilling.
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(0.95, VRAM_CAP_MB * 2**20 / total))
    todo = [n for n in names if not os.path.exists(os.path.join(OUT, n, "stats.json"))]
    for n in set(names) - set(todo):
        print(f"{n}: already rendered (delete its folder to redo)")
    models, loaded = {}, None
    for name in sorted(todo, key=lambda n: ENGINES[n]["model"]):  # one model load per model
        spec = ENGINES[name]
        if spec["model"] != loaded:
            model = prompt = gen = None  # drop every reference to the old model, or empty_cache can't free it
            models.clear()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            free = wait_for_room(spec["need_mb"])
            t0 = time.time()
            models[spec["model"]] = Qwen3TTSModel.from_pretrained(spec["model"], device_map="cuda:0", dtype=torch.bfloat16,
                                                                  attn_implementation="sdpa")
            load_s, loaded = round(time.time() - t0, 1), spec["model"]
        model = models[loaded]
        d = os.path.join(OUT, name)
        os.makedirs(d, exist_ok=True)
        lang = spec["lang"]
        kw = {} if spec["temp"] is None else {"temperature": spec["temp"], "subtalker_temperature": spec["temp"]}
        if "ref" in spec:
            wav, text = ref_files(spec["ref"])
            prompt = model.create_voice_clone_prompt(ref_audio=wav, ref_text=text, x_vector_only_mode=spec.get("xvec", False))
        jobs = list(takes(lang))
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()

        def gen(texts):
            if "ref" in spec:
                return model.generate_voice_clone(text=texts, language=[LANGUAGE[lang]] * len(texts),
                                                  voice_clone_prompt=prompt * len(texts), **kw)
            return model.generate_voice_design(text=texts, language=[LANGUAGE[lang]] * len(texts),
                                               instruct=[spec["instruct"]] * len(texts), **kw)
        stats = {"engine": name, "load_s": load_s, "free_mb_before": free, "batch": spec.get("batch", True), "takes": {}}
        if spec.get("batch", True):
            torch.manual_seed(zlib.crc32(name.encode()))
            wavs, times = [], []
            for i in range(0, len(jobs), BATCH):
                t2 = time.time()
                w, sr = gen([text for _, text, _, _ in jobs[i:i + BATCH]])
                wavs += w
                times += [(time.time() - t2) / len(w)] * len(w)  # a batch has one time: share it out
        else:
            wavs, times = [], []
            for _, text, take, _ in jobs:
                t2 = time.time()
                torch.manual_seed(seed(text, take))
                w, sr = gen([text])
                wavs += w
                times.append(time.time() - t2)
        for (key, text, take, fname), w, el in zip(jobs, wavs, times):
            save(os.path.join(d, fname), w, sr)
            stats["takes"][fname] = {"key": key, "text": text, "gen_s": round(el, 1), "audio_s": round(len(w) / sr, 2)}
        stats.update(total_s=round(time.time() - t1, 1), peak_vram_mb=round(torch.cuda.max_memory_reserved() / 2**20),
                     peak_alloc_mb=round(torch.cuda.max_memory_allocated() / 2**20), torch=torch.__version__)
        json.dump(stats, open(os.path.join(d, "stats.json"), "w"), indent=1)
        print(f"{name}: {len(jobs)} takes in {stats['total_s']} s, peak {stats['peak_vram_mb']} MB", flush=True)


def rendered():
    return [e for e in ENGINES if os.path.exists(os.path.join(OUT, e, "stats.json"))]


def check():
    """voice_check.listen (the reels' QA: Whisper base.en, or small for French as voice_check.py does) on every take."""
    import voice_check as vc
    from transformers import pipeline
    vc.SAME.update({"it's": "its"})
    asrs = {"en": pipeline("automatic-speech-recognition", model="openai/whisper-base.en", device="cpu")}
    small = pipeline("automatic-speech-recognition", model="openai/whisper-small", device="cpu")
    asrs["fr"] = lambda x, **k: small(x, **k, generate_kwargs={"language": "fr", "task": "transcribe"})
    for engine in rendered():
        lang = ENGINES[engine]["lang"]
        vc.LANG = lang
        res = {}
        for key, text, take, fname in takes(lang):
            heard, sim, problem = vc.listen(asrs[lang], os.path.join(OUT, engine, fname), text)
            res[fname] = {"heard": heard.strip(), "sim": round(sim, 3), "problem": problem}
            print(f"{engine} {fname} {sim:.2f} {problem or 'ok'} | {heard.strip()}", flush=True)
        json.dump(res, open(os.path.join(OUT, engine, "check.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def heard():
    """large-v3 transcript of every take (base.en's flags were wrong more often than right on the reels)."""
    from faster_whisper import WhisperModel
    m = WhisperModel("large-v3", device="cpu", compute_type="int8")
    for engine in rendered():
        lang = ENGINES[engine]["lang"]
        res = {}
        for key, text, take, fname in takes(lang):
            segs, _ = m.transcribe(os.path.join(OUT, engine, fname), language=lang, beam_size=5)
            res[fname] = " ".join(s.text.strip() for s in segs)
            print(f"{engine} {fname} | {res[fname]}", flush=True)
        json.dump(res, open(os.path.join(OUT, engine, "heard-large.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def load(engine, name):
    p = os.path.join(OUT, engine, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def readme():
    L = ["# Voice audition: Chatterbox vs Qwen3-TTS", "",
         "Written by `scripts/voice_audition.py readme`. The audio is Martin's cloned voice, so this folder is never "
         "committed. **To listen, open `listen.html`** (one row per take, one player per variant).", "",
         "**Whisper columns**: `sim` and `flag` come from `voice_check.listen` (the reels' QA: base.en, or small for "
         "French; similarity to the script, dropped/added words, mid-phrase pauses, rising endings). The reels re-record "
         "below 0.97. `large-v3 heard` is large Whisper's transcript, the better judge of the hard words. Neither "
         "Whisper notices an accent.", ""]
    notes = os.path.join(OUT, "notes.md")  # the hand-written verdict, kept above the tables
    if os.path.exists(notes):
        L += [open(notes, encoding="utf-8").read().strip(), ""]
    for lang in LINES:
        L += [f"## Lines ({LANGUAGE[lang]})", "", "| key | line | takes | what to listen for |", "|---|---|---|---|"]
        L += [f"| {k} | {t} | {n} | {w} |" for k, t, n, w in LINES[lang]] + [""]
    L += ["## Variants", "", "Qwen3 licence as the cards state it: " + QWEN_LICENCE + ". Chatterbox: MIT.", "",
          "| variant | what | VRAM peak | time per line | compute per s of audio |", "|---|---|---|---|---|"]
    for engine in rendered():
        st, spec = load(engine, "stats.json"), ENGINES[engine]
        g = [v["gen_s"] for v in st["takes"].values()]
        a = [v["audio_s"] for v in st["takes"].values()]
        vram = f"{st['peak_vram_mb']} MB" if "peak_vram_mb" in st else "CPU"
        how = "batched" if st.get("batch") else "one at a time"
        L.append(f"| `{engine}` | {spec['name']} | {vram} | {sum(g) / len(g):.1f} s ({how}) | {sum(g) / sum(a):.2f} s |")
    if "design-en" in rendered() or "design-fr" in rendered():
        L += ["", f"VoiceDesign instructions: English: \"{DESIGN_EN}\" French: \"{DESIGN_FR}\""]
    for engine in rendered():
        st, ck, hl = load(engine, "stats.json"), load(engine, "check.json"), load(engine, "heard-large.json")
        L += ["", f"### {engine}: {ENGINES[engine]['name']}", "",
              "| take | sim | flag | large-v3 heard | audio s |", "|---|---|---|---|---|"]
        for fname, v in st["takes"].items():
            c = ck.get(fname, {})
            L.append(f"| `{engine}/{fname}` | {c.get('sim', '')} | {c.get('problem') or ('ok' if c else '')} "
                     f"| {hl.get(fname, '')} | {v['audio_s']} |")
    open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", os.path.join(OUT, "README.md"))
    listen_page()


def listen_page():
    """listen.html: per language, one row per take and one player per variant, to compare by ear in a browser."""
    import html
    style = ("body{font:14px system-ui;margin:16px;background:#fff;color:#111}table{border-collapse:collapse;margin-bottom:32px}"
             "td,th{border:1px solid #ccc;padding:6px;vertical-align:top;text-align:left}th{max-width:220px}"
             ".h{color:#555;font-size:12px;max-width:230px}.k{color:#888;font-weight:normal;font-size:12px}audio{width:200px}")
    parts = [f'<!doctype html><meta charset="utf-8"><title>Voice audition</title><style>{style}</style>',
             "<h1>Voice audition</h1><p>Grey text under each player is what large Whisper heard. "
             "Whisper doesn't notice accents: that's for your ears.</p>"]
    for lang in LINES:
        have = [e for e in rendered() if ENGINES[e]["lang"] == lang]
        if not have:
            continue
        hl = {e: load(e, "heard-large.json") for e in have}
        head = "".join(f"<th>{html.escape(ENGINES[e]['name'])}<div class=\"k\">{e}</div></th>" for e in have)
        rows = []
        for key, text, take, fname in takes(lang):
            cells = "".join(f'<td><audio controls preload="none" src="{e}/{fname}"></audio>'
                            f'<div class="h">{html.escape(hl[e].get(fname, ""))}</div></td>' for e in have)
            rows.append(f'<tr><th>{html.escape(text)}<div class="k">{key} take {take}</div></th>{cells}</tr>')
        parts.append(f"<h2>{LANGUAGE[lang]}</h2><table><tr><th>line</th>{head}</tr>{''.join(rows)}</table>")
    open(os.path.join(OUT, "listen.html"), "w", encoding="utf-8").write("".join(parts))
    print("wrote", os.path.join(OUT, "listen.html"))


if __name__ == "__main__":
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    if cmd == "render" and args == ["chatterbox"]:
        render_chatterbox()
    elif cmd == "render" and all(a in ENGINES and a != "chatterbox" for a in args):
        render_qwen(args or [e for e in ENGINES if e != "chatterbox"])
    elif cmd in ("check", "heard", "readme"):
        globals()[cmd]()
    else:
        sys.exit(__doc__)
