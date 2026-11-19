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
# VoiceDesign designs a new voice on every call, so its accent drifted between lines even at temp 0.5. Two fixes:
# the structured description its model card uses (every attribute spelled out), and "design once, clone after":
# one designed clip becomes the clone reference (ref-design-<lang>), so every line comes from the same voice.
ATTR = ("gender: Male.\npitch: Mid-low male pitch, steady, with gentle rises on questions.\n"
        "speed: Moderate, unhurried pace with short natural pauses.\n"
        "volume: Steady conversational volume, holding up to the very end of each sentence.\n"
        "age: Middle-aged adult, around forty.\nclarity: Very clear, precise pronunciation.\n"
        "fluency: Fluent, no hesitations.\naccent: {accent}, the same in every sentence.\n"
        "texture: Warm, clear, slightly low.\nemotion: Calm curiosity, quietly fascinated.\n"
        "tone: Friendly and explanatory, like a science documentary narrator.\npersonality: Thoughtful and engaging.")
ATTR_EN = ATTR.format(accent="General North American English (Canadian)")
ATTR_FR = ATTR.format(accent="Québécois French (Canadian French from Quebec)")
DESIGNED_REFS = {  # name: (instruct, language, what the designed voice says, which becomes the clone transcript)
    "ref-design-en": (ATTR_EN, "en", "Every night, the sky shows us light that left its source long before we were "
                      "born. Some of it is older than the Earth itself. So when you look up, you're not just seeing far "
                      "away. You're seeing back in time."),
    "ref-design-fr": (ATTR_FR, "fr", "Chaque nuit, le ciel nous montre de la lumière partie bien avant notre naissance. "
                      "Une partie est plus vieille que la Terre elle-même. Alors quand on lève les yeux, on ne voit pas "
                      "seulement loin. On voit dans le passé."),
}
# Qwen3's clones lose about 5 dB (French 6-7) over a line's last quarter; Martin heard it as the sound going down.
# "-even" refs run ffmpeg's fast leveller over the reference first; "post" variants level the finished takes instead.
EVEN = "dynaudnorm=f=100:g=7:p=0.9:m=10"
ENGINES.update({
    "qwen3-t06": {"name": "Qwen3 clone, temp 0.6", "lang": "en", "model": BASE, "ref": "ref", "temp": 0.6},
    "qwen3-t06-evenref": {"name": "Qwen3 clone, temp 0.6, evened reference", "lang": "en", "model": BASE,
                          "ref": "ref-even", "temp": 0.6},
    "qwen3-t06-evenout": {"name": "Qwen3 clone, temp 0.6, takes evened after", "lang": "en", "from": "qwen3-t06", "post": EVEN},
    "design-en-attr": {"name": "VoiceDesign, structured description, temp 0.5", "lang": "en", "model": DESIGN,
                       "instruct": ATTR_EN, "temp": 0.5},
    "design-en-attr-t03": {"name": "VoiceDesign, structured description, temp 0.3", "lang": "en", "model": DESIGN,
                           "instruct": ATTR_EN, "temp": 0.3},
    "design-clone-en-t06": {"name": "VoiceDesign once, then cloned at temp 0.6", "lang": "en", "model": BASE,
                            "ref": "ref-design-en", "temp": 0.6},
    "qwen3-fr-t06": {"name": "Qwen3 clone, French, temp 0.6", "lang": "fr", "model": BASE, "ref": "ref-fr", "temp": 0.6},
    "qwen3-fr-t07": {"name": "Qwen3 clone, French, temp 0.7", "lang": "fr", "model": BASE, "ref": "ref-fr", "temp": 0.7},
    "qwen3-fr-t08": {"name": "Qwen3 clone, French, temp 0.8", "lang": "fr", "model": BASE, "ref": "ref-fr", "temp": 0.8},
    "qwen3-fr-t06-evenref": {"name": "Qwen3 clone, French, temp 0.6, evened reference", "lang": "fr", "model": BASE,
                             "ref": "ref-fr-even", "temp": 0.6},
    "qwen3-fr-t06-evenout": {"name": "Qwen3 clone, French, temp 0.6, takes evened after", "lang": "fr",
                             "from": "qwen3-fr-t06", "post": EVEN},
    "design-fr-attr": {"name": "VoiceDesign, French, structured description (Quebec), temp 0.5", "lang": "fr",
                       "model": DESIGN, "instruct": ATTR_FR, "temp": 0.5},
    "design-clone-fr-t06": {"name": "VoiceDesign once (Quebec), then cloned at temp 0.6", "lang": "fr", "model": BASE,
                            "ref": "ref-design-fr", "temp": 0.6},
})
# Round 4 (2026-11-19): Martin liked both evened French variants and asked for "half even out ref, half even out
# after". HALF blends each signal 50/50 with its levelled copy, so it applies about half of dynaudnorm's correction.
HALF = f"asplit[a][b];[b]{EVEN}[e];[a][e]amix=inputs=2:normalize=1"
ENGINES.update({
    "qwen3-t06-halfref": {"name": "Qwen3 clone, temp 0.6, half-evened reference", "lang": "en", "model": BASE,
                          "ref": "ref-half", "temp": 0.6},
    "qwen3-t06-halfboth": {"name": "Qwen3 clone, temp 0.6, half-evened reference + takes half-evened after",
                           "lang": "en", "from": "qwen3-t06-halfref", "post": HALF},
    "qwen3-t06-evenboth": {"name": "Qwen3 clone, temp 0.6, evened reference + takes evened after", "lang": "en",
                           "from": "qwen3-t06-evenref", "post": EVEN},
    "design-clone-en-t06-halfout": {"name": "Design-clone, temp 0.6, takes half-evened after", "lang": "en",
                                    "from": "design-clone-en-t06", "post": HALF},
    "design-clone-en-t06-evenout": {"name": "Design-clone, temp 0.6, takes evened after", "lang": "en",
                                    "from": "design-clone-en-t06", "post": EVEN},
    "qwen3-fr-t06-halfref": {"name": "Qwen3 clone, French, temp 0.6, half-evened reference", "lang": "fr",
                             "model": BASE, "ref": "ref-fr-half", "temp": 0.6},
    "qwen3-fr-t06-halfboth": {"name": "Qwen3 clone, French, temp 0.6, half-evened reference + takes half-evened after",
                              "lang": "fr", "from": "qwen3-fr-t06-halfref", "post": HALF},
    "qwen3-fr-t06-evenboth": {"name": "Qwen3 clone, French, temp 0.6, evened reference + takes evened after",
                              "lang": "fr", "from": "qwen3-fr-t06-evenref", "post": EVEN},
})
# What listen.html puts first, side by side: the current picks and this round's new variants.
SHORTLIST = ["design-clone-en-t06", "design-clone-en-t06-halfout", "design-clone-en-t06-evenout",
             "qwen3-t06-evenout", "qwen3-t06-halfboth", "qwen3-t06-evenboth",
             "qwen3-fr-t06-evenref", "qwen3-fr-t06-evenout", "qwen3-fr-t06-halfboth", "qwen3-fr-t06-evenboth"]
for _k, _v in ENGINES.items():
    if "from" in _v:
        _v.update(venv="any (ffmpeg)", device="CPU, ffmpeg", licence=QWEN_LICENCE, need_mb=0)
    elif "model" in _v:
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
    for r in {ENGINES[n].get("ref") for n in todo} - {None}:
        level = {"-even": EVEN, "-half": HALF}.get(r[-5:])
        if level and not os.path.exists(os.path.join(OUT, r + ".wav")):  # an evened copy of a reference
            src = r[:-5]
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", os.path.join(OUT, src + ".wav"), "-af", level,
                            os.path.join(OUT, r + ".wav")], check=True)
            open(os.path.join(OUT, r + ".txt"), "w", encoding="utf-8").write(ref_files(src)[1] + "\n")
    designs = [r for r in DESIGNED_REFS if not os.path.exists(os.path.join(OUT, r + ".wav"))
               and any(ENGINES[n].get("ref") == r for n in todo)]
    models, loaded = {}, None
    order = sorted(todo, key=lambda n: ENGINES[n]["model"] != DESIGN)  # VoiceDesign first: it makes the designed refs
    for name in ([None] if designs and ENGINES[order[0]]["model"] != DESIGN else []) + order:
        mid = DESIGN if name is None else ENGINES[name]["model"]
        if mid != loaded:
            model = prompt = gen = None  # drop every reference to the old model, or empty_cache can't free it
            models.clear()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            free = wait_for_room(ENGINES[order[0]]["need_mb"])
            t0 = time.time()
            models[mid] = Qwen3TTSModel.from_pretrained(mid, device_map="cuda:0", dtype=torch.bfloat16,
                                                        attn_implementation="sdpa")
            load_s, loaded = round(time.time() - t0, 1), mid
            if mid == DESIGN:
                for r in designs:  # design each voice once; its clip and words become a clone reference
                    instruct, lang, text = DESIGNED_REFS[r]
                    torch.manual_seed(zlib.crc32(r.encode()))
                    w, sr = models[mid].generate_voice_design(text=[text], language=[LANGUAGE[lang]], instruct=[instruct],
                                                              temperature=0.5, subtalker_temperature=0.5)
                    save(os.path.join(OUT, r + ".wav"), w[0], sr)
                    open(os.path.join(OUT, r + ".txt"), "w", encoding="utf-8").write(text + "\n")
                    print(f"designed {r}: {len(w[0]) / sr:.1f} s", flush=True)
        if name is None:
            continue
        spec = ENGINES[name]
        model = models[loaded]
        d = os.path.join(OUT, name)
        os.makedirs(d, exist_ok=True)
        lang = spec["lang"]
        kw = {} if spec["temp"] is None else {"temperature": spec["temp"], "subtalker_temperature": spec["temp"]}
        # 12 codec frames per second of audio: stop at 30 s. A VoiceDesign take of a 7 s French line ran on for
        # 147 s, and its whole batch waited for it (design-fr, 2026-11-18).
        kw["max_new_tokens"] = 12 * 30
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


def render_post(names):
    """Variants made from another variant's takes by an ffmpeg filter (no GPU)."""
    for name in names:
        spec, d = ENGINES[name], os.path.join(OUT, name)
        src = load(spec["from"], "stats.json")
        if os.path.exists(os.path.join(d, "stats.json")) or not src:
            print(f"{name}: {'already made' if src else spec['from'] + ' is not rendered yet'}")
            continue
        os.makedirs(d, exist_ok=True)
        for fname in src["takes"]:
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", os.path.join(OUT, spec["from"], fname), "-af", spec["post"],
                            "-c:a", "pcm_s16le", os.path.join(d, fname)], check=True)
        json.dump({**src, "engine": name, "post": spec["post"]}, open(os.path.join(d, "stats.json"), "w"), indent=1)
        print(f"{name}: made from {spec['from']}")


def fade_db(path):
    """How much a take fades: loudness of its last voiced quarter minus the rest, in dB (50 ms frames, voiced = within
    35 dB of the loudest frame). Qwen3's clones measured about -5 dB in English, VoiceDesign about -2."""
    import numpy as np
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32")
    x = x.mean(1) if x.ndim > 1 else x
    fr = int(0.05 * sr)
    db = np.array([10 * np.log10(np.mean(x[i:i + fr] ** 2) + 1e-12) for i in range(0, len(x) - fr, fr)])
    v = db[db > db.max() - 35]
    if len(v) < 8:
        return None
    k = int(len(v) * 0.75)
    return round(float(v[k:].mean() - v[:k].mean()), 1)


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
    for engine in [e for e in rendered() if not load(e, "check.json")]:  # delete check.json to check again
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
    for engine in [e for e in rendered() if not load(e, "heard-large.json")]:
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
         "Whisper notices an accent. `end fade` = loudness of a take's last voiced quarter minus the rest (Martin heard "
         "Qwen's lines go down at the end).", ""]
    notes = os.path.join(OUT, "notes.md")  # the hand-written verdict, kept above the tables
    if os.path.exists(notes):
        L += [open(notes, encoding="utf-8").read().strip(), ""]
    for lang in LINES:
        L += [f"## Lines ({LANGUAGE[lang]})", "", "| key | line | takes | what to listen for |", "|---|---|---|---|"]
        L += [f"| {k} | {t} | {n} | {w} |" for k, t, n, w in LINES[lang]] + [""]
    L += ["## Variants", "", "Qwen3 licence as the cards state it: " + QWEN_LICENCE + ". Chatterbox: MIT.", "",
          "| variant | what | VRAM peak | time per line | compute per s of audio | end fade |", "|---|---|---|---|---|---|"]
    for engine in rendered():
        st, spec = load(engine, "stats.json"), ENGINES[engine]
        g = [v["gen_s"] for v in st["takes"].values()]
        a = [v["audio_s"] for v in st["takes"].values()]
        vram = f"{st['peak_vram_mb']} MB" if "peak_vram_mb" in st else "CPU"
        how = "batched" if st.get("batch") else "one at a time"
        fades = [f for f in (fade_db(os.path.join(OUT, engine, n)) for n in st["takes"]) if f is not None]
        how = "ffmpeg after " + spec["from"] if "from" in spec else how
        L.append(f"| `{engine}` | {spec['name']} | {vram} | {sum(g) / len(g):.1f} s ({how}) | {sum(g) / sum(a):.2f} s "
                 f"| {sum(fades) / len(fades):+.1f} dB |")
    if "design-en" in rendered() or "design-fr" in rendered():
        L += ["", f"VoiceDesign instructions: English: \"{DESIGN_EN}\" French: \"{DESIGN_FR}\""]
    for engine in rendered():
        st, ck, hl = load(engine, "stats.json"), load(engine, "check.json"), load(engine, "heard-large.json")
        L += ["", f"### {engine}: {ENGINES[engine]['name']}", "",
              "| take | sim | flag | large-v3 heard | audio s | end fade dB |", "|---|---|---|---|---|---|"]
        for fname, v in st["takes"].items():
            c = ck.get(fname, {})
            L.append(f"| `{engine}/{fname}` | {c.get('sim', '')} | {c.get('problem') or ('ok' if c else '')} "
                     f"| {hl.get(fname, '')} | {v['audio_s']} | {fade_db(os.path.join(OUT, engine, fname))} |")
    open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", os.path.join(OUT, "README.md"))
    listen_page()


def listen_page():
    """listen.html: the shortlist first, then every variant; per language, one row per take, one player per variant."""
    style = ("body{font:14px system-ui;margin:16px;background:#fff;color:#111}table{border-collapse:collapse;margin-bottom:32px}"
             "td,th{border:1px solid #ccc;padding:6px;vertical-align:top;text-align:left}th{max-width:220px}"
             ".h{color:#555;font-size:12px;max-width:230px}.k{color:#888;font-weight:normal;font-size:12px}audio{width:200px}")
    parts = [f'<!doctype html><meta charset="utf-8"><title>Voice audition</title><style>{style}</style>',
             "<h1>Voice audition</h1><p>Grey text under each player is what large Whisper heard. "
             "Whisper doesn't notice accents: that's for your ears.</p>"]
    for title, pick in [("Shortlist", lambda e: e in SHORTLIST), ("Everything", lambda e: True)]:
        parts.append(f"<h1>{title}</h1>")
        for lang in LINES:
            parts += page_table(lang, [e for e in (SHORTLIST if title == "Shortlist" else rendered())
                                       if e in rendered() and pick(e) and ENGINES[e]["lang"] == lang])
    open(os.path.join(OUT, "listen.html"), "w", encoding="utf-8").write("".join(parts))
    print("wrote", os.path.join(OUT, "listen.html"))


def page_table(lang, have):
    import html
    if not have:
        return []
    hl = {e: load(e, "heard-large.json") for e in have}
    head = "".join(f"<th>{html.escape(ENGINES[e]['name'])}<div class=\"k\">{e}</div></th>" for e in have)
    rows = []
    for key, text, take, fname in takes(lang):
        cells = "".join(f'<td><audio controls preload="none" src="{e}/{fname}"></audio>'
                        f'<div class="h">{html.escape(hl[e].get(fname, ""))}</div></td>' for e in have)
        rows.append(f'<tr><th>{html.escape(text)}<div class="k">{key} take {take}</div></th>{cells}</tr>')
    return [f"<h2>{LANGUAGE[lang]}</h2><table><tr><th>line</th>{head}</tr>{''.join(rows)}</table>"]


if __name__ == "__main__":
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    if cmd == "render" and args == ["chatterbox"]:
        render_chatterbox()
    elif cmd == "render" and all(a in ENGINES and a != "chatterbox" for a in args):
        names = args or [e for e in ENGINES if e != "chatterbox"]
        gpu = [n for n in names if "model" in ENGINES[n]]
        if gpu:
            render_qwen(gpu)
        render_post([n for n in names if "from" in ENGINES[n]])
    elif cmd in ("check", "heard", "readme"):
        globals()[cmd]()
    else:
        sys.exit(__doc__)
