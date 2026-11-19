# Voice audition: what was tried and what was picked (2026-11-18/19)

Chatterbox has been cloning Martin's voice for the reels. Qwen3-TTS (Apache 2.0) was auditioned against it with
`scripts/voice_audition.py`. Its takes, Whisper results and `listen.html` are in `out/voice-audition/`, which is
git-ignored because most of it is Martin's cloned voice. This file keeps what was learned, so nothing has to be
redone. Martin judged by ear; Whisper and level measurements backed it up. Claude can't hear the takes.

## Decisions

| | What | Settings | Status |
|---|---|---|---|
| English reels | **narrator**: VoiceDesign once, then cloned | Qwen3 Base clone of `voice-design/narrator-en.wav` at temperature 0.6 | **In use from 2026-11-19** (`SPEAKERS` in the Videos repo's `reel.py`) |
| Martin's own voice | Qwen3 clone, clips evened after | clone of `voice-ref/martin-en-cut.wav` at 0.6, then ffmpeg `EVEN` on every clip | On hold: the `martin` speaker in `reel.py` |
| French | Qwen3 clone of Martin's French | clone of `voice-ref/martin-fr-cut.wav` at 0.6, `language="French"`, with the reference evened before cloning or the clips evened after (Martin liked both) | Later: reel qa and captions are English-only |
| Chatterbox | Martin's clone on the CPU | exaggeration 0.6, cfg 0.3 | Kept for the reels recorded with it |

`EVEN = dynaudnorm=f=100:g=7:p=0.9:m=10`. `HALF` blends a signal 50/50 with its evened copy:
`asplit[a][b];[b]EVEN[e];[a][e]amix=inputs=2:normalize=1`.

The narrator's description, which made `narrator-en.wav` (VoiceDesign 1.7B, temperature 0.5, seed crc32("ref-design-en")):
the `ATTR_EN` text in `scripts/voice_audition.py`. That is a structured description (male, mid-low pitch, moderate pace,
volume holding to the end of each sentence, around forty, "General North American English (Canadian), the same in
every sentence", warm, calm curiosity, documentary narrator). The clip is committed, because a new VoiceDesign run
would make a different person.

## What was learned

- **Temperature.** At Qwen's default of 0.9, Martin heard the accent wander between lines (American, British, maybe
  Spanish, French-sounding). He ranked 0.7 above 0.5 and 0.9 in English, then picked 0.6 as the best for everything.
  In French, 0.5 beat 0.9; 0.6, 0.7 and 0.8 scored the same in Whisper, with 0.8 fading most.
- **Hard words.** Large Whisper heard Qwen3 say "set off", "Checks so far" and the noun "pull" right in every take.
  Chatterbox's own QA (Whisper base.en) heard "set up", "Check so far" and "pole". Large Whisper heard 8 of 9
  of Chatterbox's right, so base.en may have been mishearing the cloned accent rather than the voice getting it wrong.
- **Accent can't be set when cloning.** Qwen3's language is just a name ("English", "French"). Only VoiceDesign and
  CustomVoice take instructions.
- **VoiceDesign drifts.** It designs a new voice on every call, so each line sounds like a slightly different person,
  even at temperature 0.3 to 0.5 and even when the description says "the same in every sentence". Designing once
  (one 15 s paragraph) and cloning every line from that clip fixes it.
- **Québécois.** Neither designed voice sounded Québécois to Martin, whatever the description said. A Quebec
  accent has to come from cloning a Quebec speaker.
- **Ending fade.** Clones of Martin got about 5 dB quieter over a line's last quarter (French 6 to 7 dB), because his
  reference does. Measured by `fade_db`. The narrator fades about 2 dB. The fixes, measured on the same lines:
  - Evening the reference before cloning (`-evenref`): halves the fade. In English, 9 of 12 takes then ended on a
    rising pitch, against 3. French didn't show that.
  - Evening the clips after (`-evenout`): removes the fade without changing the voice. Martin preferred it for his
    English voice. The level swings a little line to line.
  - Half of each (`-halfboth`), at Martin's suggestion: see round 4 below.
- **Reference length.** There's no official limit; the model card says cloning works from 3 s. A 48 s reference was
  no better in Whisper than the 19 s one, and used more memory (8.9 against 7.6 GB in a batch of 3). 15 to 20 s of
  clean, evenly read speech is enough.
- **Speed.** About 15 s a line one at a time on the RTX 5070 (4 s of compute per second of audio), or about 5 s a
  take in batches of 3. Chatterbox took about 32 s a line on the CPU.

## Machine safety (RTX 5070, 12 GB)

- The PC hard-reset twice during the first GPU renders (Kernel-Power 41, no bugcheck). One was a 12-take batch peaking
  at 19.7 GB, which Windows spilled into system RAM. The other ran with large Whisper loading the CPU at the same
  time. Since then, PyTorch is capped at 9 GB (`set_per_process_memory_fraction`), batches are 3 takes at most, and
  no CPU Whisper runs beside a GPU render. Nothing has reset since, and the GPU log showed at most 181 W of 250 and 62 °C.
- One VoiceDesign take ran on to 147 s of audio for a 7 s line. Every generation is now capped by `max_new_tokens`
  (12 codec frames a second).
- The local LLM (Strata) holds about 10 GB and unloads after 15 s idle, so the voice waits for 7 GB of free VRAM.

## Round 4 (2026-11-19): half and half

Martin asked whether half of each fix would beat either one alone. Results on the same 12 takes (end fade: 0 = no
fade; rising endings out of 12, by voice_check):

| variant | end fade | rising endings | notes |
|---|---|---|---|
| English: plain 0.6 / evenout | -4.7 / +0.0 dB | 3 / 3 | evenout: Martin's pick for his voice |
| English: evenref / evenboth | -2.1 / -0.4 dB | 9 / 9 | "Jacks so far" once |
| English: halfref / **halfboth** | -2.8 / **-1.2 dB** | 6 / 6 | between the two, as expected |
| narrator / halfout / evenout | -2.1 / -1.5 / -1.3 dB | 4 / 4 / 4 | the narrator barely fades, so neither changes much |
| French: plain 0.6 / evenout | -6.7 / -1.7 dB | 8 / 8 | |
| French: evenref / evenboth | -2.6 / -1.1 dB | 6 / 6 | evenboth's Whisper similarity dips (0.90) |
| French: halfref / **halfboth** | -6.0 / **-3.7 dB** | 5 / 5 | one halfref take added "Par en bas" at the start (a seed, not the filter) |

Halving the reference levelling halves the rising-ending problem in English (6 instead of 9) but leaves more fade
than evening the clips after. In French, half-evening the reference barely helped (-6.0 dB). On numbers alone,
evening after (`evenout`) still looks best for Martin's English, and the narrator doesn't need either. Martin judges
by ear: `out/voice-audition/listen.html` puts these side by side under "Shortlist".
