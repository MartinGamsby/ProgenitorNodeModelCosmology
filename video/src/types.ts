export type Sound = 'boom' | 'ding' | 'whoosh';

export type SceneDef = {
  /** Stable id: names the voice clips and the timing entry. */
  id: string;
  /** Visual component key, see src/scenes/index.ts. */
  scene: string;
  /** Small label shown top-left ('' = none). */
  chapter: string;
  /** For "What if…" scenes: the question on the stamp card (shown before beat 0). */
  whatIf?: string;
  /** Narration, one caption/voice clip per entry. '' = silent beat. */
  beats: string[];
  /** Twinkling stars behind this scene (default true). Off whenever we look at the universe
   *  or the attractor swarm from outside: there are no stars out there. */
  stars?: boolean;
  /** Pad the scene to at least this long. */
  minSeconds?: number;
  sfx?: {beat: number; sound: Sound}[];
};

/** Per-beat timing in frames, relative to the scene start. */
export type Beat = {from: number; dur: number; text: string; audio?: string};

export type SceneTiming = {
  id: string;
  from: number; // absolute frame
  dur: number;
  intro: number; // frames of what-if stamp before beat 0
  beats: Beat[];
};

/** Written by scripts/voice.mjs */
export type VoiceManifest = {
  engine: string;
  clips: Record<string, {file: string; seconds: number; text: string}>;
};
