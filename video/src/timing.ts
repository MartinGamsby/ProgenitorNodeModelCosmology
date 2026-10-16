import {SCENES} from './script';
import {clipKey, spokenText} from './markup';
import type {SceneTiming, VoiceManifest} from './types';
import {FPS, T} from './constants';
import voice from './generated/voice.json';

const manifest = voice as VoiceManifest;
const WORDS_PER_SECOND = 2.6; // fallback when no voice clip exists

const sec = (s: number) => Math.round(s * FPS);

export const buildTimeline = (): {scenes: SceneTiming[]; total: number} => {
  let cursor = 0;
  const scenes = SCENES.map((s) => {
    const intro = s.whatIf ? sec(T.whatIfIntro) : 0;
    let t = intro;
    const beats = s.beats.map((text, i) => {
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
  return {scenes, total: cursor};
};

export const TIMELINE = buildTimeline();
