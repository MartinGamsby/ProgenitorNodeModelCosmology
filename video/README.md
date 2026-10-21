# Explainer video — "What If Dark Energy Is Just… the Neighbours?"

A ~11-minute YouTube-style explainer of the Progenitor Node Model, generated entirely from
code with [Remotion](https://www.remotion.dev) (React → MP4). No video editor: change a
line of text or a number, re-render.

Every chart uses **real repo data**: Pantheon+ supernovae, the simulated best-observer
a(t), real particle snapshots, the relaxed virialized medium, and the 72-cell universe-like
verdict.

## Quick start

```bash
npm install          # once
npm run studio       # live preview in the browser, scrub the timeline
npm run render       # out/full-en.mp4 (≈30-45 min; loudness-normalised to -14 LUFS)
node scripts/render.mjs full fr   # any cut x language: out/<cut>-<locale>.mp4
npm run thumbnail    # out/thumbnail.png (1280x720)
```

## How to change things

| I want to…                         | Edit                                     | Then                       |
|-----------------------------------|------------------------------------------|----------------------------|
| reword narration / captions        | `src/i18n/en.ts`, `src/i18n/fr.ts` (`narration`) | `LOCALE=en npm run voice` |
| on-screen labels                   | `src/i18n/<locale>.ts` (`labels`)        | `npm run i18n:check`       |
| scene structure (sfx, lead-in…)    | `src/content/scenes.ts`                  | —                          |
| which scenes a video uses          | `src/cuts.ts` (full / idea / short)      | —                          |
| change a scene's visuals           | `src/scenes/*.tsx` (keys in `scenes/index.ts`) | —                    |
| colours / fonts                    | `src/theme.ts`                           | —                          |
| pacing (gaps, what-if stamp length)| `src/constants.ts` (`T`)                 | —                          |
| update numbers after new sims      | `scripts/export_data.py` (source files)  | `npm run data`             |
| music / sound effects              | `scripts/music.py` (synthesised)         | `npm run music`            |

Caption markup in narration: `*word*` highlights, `{shown|spoken}` separates caption text
from what the voice says, `10^120` renders as a superscript.

Timing is automatic: each narration line is one voice clip, every animation is keyed to a
beat number, so rewording re-times the whole video.

## Languages and cuts

Every word is in `src/i18n/<locale>.ts` (English is the reference; a missing French string falls
back to English and `npm run i18n:check` lists it). Each video ("cut") is a scene list plus beats to
skip, in `src/cuts.ts`; compositions are named `<cut>-<locale>` (`full-en`, `full-fr`, …).

## Voice

The voice sets the pace: each line's clip length is its beat length, so **a new voice means a
new render** (the visuals re-time to stay in sync).

| engine | quality | speed (CPU) | knobs |
|---|---|---|---|
| `sapi` (default) | robotic, offline Windows voice | seconds | `SAPI_VOICE`, `SAPI_RATE` |
| `kokoro` | clean neural narrator (Apache 2.0) | minutes | `KOKORO_VOICE` (bm_george, am_michael, af_heart…), `KOKORO_SPEED` |
| `chatterbox` | expressive neural (MIT), voice cloning | slow, hours | `CB_EXAGGERATION` (0.5 neutral … 1+ dramatic), `CB_CFG`, `CB_PROMPT` (reference .wav to clone) |
| `piper` | decent neural | fast | `PIPER_MODEL` |
| `none` | no audio, timing estimated from word count | — | — |

Neural engines run in an isolated venv (keeps the research Python env untouched):

```bash
py -3.12 -m venv .tts-venv
.tts-venv/Scripts/python -m pip install chatterbox-tts kokoro soundfile "setuptools<81" --extra-index-url https://download.pytorch.org/whl/cpu
.tts-venv/Scripts/python scripts/voice_samples.py        # A/B audition -> out/voice-samples/*.wav
export VOICE_ENGINE=chatterbox CB_EXAGGERATION=0.6 CB_CFG=0.3 CB_PROMPT=voice-ref/martin-en-cut.wav
LOCALE=en npm run voice                                          # all clips (hours on CPU), resumable
LOCALE=en .tts-venv/Scripts/python scripts/voice_check.py --fix  # Whisper QA, re-rolls bad clips
LOCALE=en npm run voice                                          # refresh timing after a fix
# French: LOCALE=fr CB_LANGUAGE=fr CB_PROMPT=voice-ref/martin-fr-cut.wav (Chatterbox Multilingual)
```

The narration clones Martin's voice from a short reference recording (`voice-ref/`, never
committed), exaggeration 0.6, CFG 0.3.

Clip files carry a hash of engine + settings, so only missing or changed lines are generated.

## Review helpers

```bash
npm run frames                         # every scene: start frame, length, beat frames (CUT=, LOCALE= env)
node scripts/stills.mjs data:6:30 outro:3   # stills at scene:beat:offset -> out/stills + out/sheet-*.jpg
```

## Layout

```
scripts/   export_data.py  voice.mjs  voice_check.py  music.py  render.mjs  stills.mjs  frames.mjs
src/       i18n/ (ALL THE TEXT: en.ts, fr.ts)  content/scenes.ts  cuts.ts  timing.ts  Video.tsx  Root.tsx
           components/  (Space3D canvas renderer, captions, charts, KaTeX)
           scenes/      (one file per group of scenes)
public/    data/*.json (exported)  voice/<locale>/*.wav (generated)  audio/*.wav (synthesised)
```

Remotion is free for individuals and small teams; check its licence for company use.
