# Explainer video — "What If Dark Energy Is Just… the Neighbours?"

A ~10-minute YouTube-style explainer of the Progenitor Node Model, generated entirely from
code with [Remotion](https://www.remotion.dev) (React → MP4). No video editor: change a
line of text or a number, re-render.

Every chart uses **real repo data**: Pantheon+ supernovae, the simulated best-observer
a(t), real particle snapshots, the relaxed virialized medium, and the 72-cell universe-like
verdict.

## Quick start

```bash
npm install          # once
npm run studio       # live preview in the browser, scrub the timeline
npm run render       # out/explainer.mp4 (≈30 min; loudness-normalised to -14 LUFS)
npm run thumbnail    # out/thumbnail.png (1280x720)
```

## How to change things

| I want to…                         | Edit                                     | Then                       |
|-----------------------------------|------------------------------------------|----------------------------|
| reword narration / captions        | `src/script.ts` (`beats`)                | `npm run voice`            |
| reorder, drop or add a scene       | `src/script.ts` (`SCENES` array)         | —                          |
| change a scene's visuals           | `src/scenes/*.tsx` (keys in `scenes/index.ts`) | —                    |
| colours / fonts                    | `src/theme.ts`                           | —                          |
| pacing (gaps, what-if stamp length)| `src/constants.ts` (`T`)                 | —                          |
| update numbers after new sims      | `scripts/export_data.py` (source files)  | `npm run data`             |
| music / sound effects              | `scripts/music.py` (synthesised)         | `npm run music`            |

Caption markup in `script.ts`: `*word*` highlights, `{shown|spoken}` separates caption text
from what the voice says, `10^120` renders as a superscript.

Timing is automatic: each narration line is one voice clip, every animation is keyed to a
beat number, so rewording re-times the whole video.

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
VOICE_ENGINE=chatterbox CB_EXAGGERATION=0.8 CB_CFG=0.35 npm run voice   # all clips (~1 h on CPU), resumable
CB_EXAGGERATION=0.8 CB_CFG=0.35 .tts-venv/Scripts/python scripts/voice_check.py --fix  # Whisper QA, re-rolls bad clips
VOICE_ENGINE=chatterbox CB_EXAGGERATION=0.8 CB_CFG=0.35 npm run voice   # refresh timing after a fix
```

The current narration is Chatterbox's default voice at exaggeration 0.8, CFG 0.35.

Clip files carry a hash of engine + settings, so only missing or changed lines are generated.

## Review helpers

```bash
npm run frames                         # every scene: start frame, length, beat frames
node scripts/stills.mjs data:6:30 outro:3   # stills at scene:beat:offset -> out/stills + out/sheet-*.jpg
```

## Layout

```
scripts/   export_data.py  voice.mjs  music.py  stills.mjs  frames.mjs  sheet.py
src/       script.ts (THE SCRIPT)  timing.ts  beats.tsx  Video.tsx  Root.tsx
           components/  (Space3D canvas renderer, captions, charts, KaTeX)
           scenes/      (one file per group of scenes)
public/    data/*.json (exported)  voice/*.wav (generated)  audio/*.wav (synthesised)
```

Remotion is free for individuals and small teams; check its licence for company use.
