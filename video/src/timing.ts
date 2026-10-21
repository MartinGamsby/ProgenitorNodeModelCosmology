import {SCENE_SPECS} from './content/scenes';
import {CUTS} from './cuts';
import {LOCALES, narrationFor} from './i18n';
import {en} from './i18n/en';
import {clipKey, spokenText} from './markup';
import type {CutId, LocaleId, SceneDef, SceneTiming, VoiceManifest} from './types';
import {FPS, T} from './constants';
import voiceEn from './generated/voice.en.json';
import voiceFr from './generated/voice.fr.json';

const MANIFESTS: Record<LocaleId, VoiceManifest> = {en: voiceEn as VoiceManifest, fr: voiceFr as VoiceManifest};
const WORDS_PER_SECOND = 2.6; // fallback when no voice clip exists

const sec = (s: number) => Math.round(s * FPS);

/** The scenes of a cut, with their text in one language. */
export const resolveScenes = (cut: CutId, locale: LocaleId): SceneDef[] =>
  CUTS[cut].scenes.map((id) => {
    const s = SCENE_SPECS.find((x) => x.id === id);
    if (!s) throw new Error(`cut ${cut}: unknown scene "${id}"`);
    const L = LOCALES[locale];
    const beats = narrationFor(cut, locale, id);
    if (beats.length !== s.beats) {
      throw new Error(`${locale}/${cut}/${id}: ${beats.length} narration lines, but the visuals expect ${s.beats}`);
    }
    return {
      id,
      scene: s.scene,
      chapter: L.chapters[id] ?? en.chapters[id] ?? '',
      whatIf: s.whatIf ? (L.whatIf[id] ?? en.whatIf[id]) : undefined,
      beats,
      voice: s.voice,
      leadSeconds: s.leadSeconds,
      stars: s.stars,
      minSeconds: s.minSeconds,
      sfx: s.sfx,
    };
  });

export type Timeline = {scenes: SceneTiming[]; defs: SceneDef[]; total: number};

/** Frame timing for a cut in one language: beats last as long as their voice clip (+ gap);
 *  beats the cut skips take zero time, so everything keyed to them collapses. */
export const buildTimeline = (cut: CutId, locale: LocaleId): Timeline => {
  const manifest = MANIFESTS[locale];
  const defs = resolveScenes(cut, locale);
  const skips = CUTS[cut].skip ?? {};
  let cursor = 0;
  const scenes = defs.map((s) => {
    const skip = new Set(skips[s.id] ?? []);
    const intro = (s.whatIf ? sec(T.whatIfIntro) : 0) + sec(s.leadSeconds ?? 0);
    let t = intro;
    const beats = s.beats.map((text, i) => {
      if (skip.has(i)) return {from: t, dur: 0, text: ''};
      const spoken = spokenText(text);
      const clip = spoken ? manifest.clips[clipKey(s.id, i, text)] : undefined;
      const seconds = !spoken
        ? T.silentBeat
        : (clip?.seconds ?? spoken.split(/\s+/).length / WORDS_PER_SECOND) + T.beatGap;
      const b = {from: t, dur: sec(seconds), text, audio: clip?.file};
      t += b.dur;
      return b;
    });
    const dur = Math.max(t + sec(T.tail), sec(s.minSeconds ?? 0));
    const timing: SceneTiming = {id: s.id, from: cursor, dur, intro, beats};
    cursor += dur;
    return timing;
  });
  return {scenes, defs, total: cursor};
};
