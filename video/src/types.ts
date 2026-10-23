export type Sound = 'boom' | 'ding' | 'whoosh';

export type LocaleId = 'en' | 'fr';
export type CutId = 'full' | 'idea' | 'short';

/** Language-independent scene structure (src/content/scenes.ts). */
export type SceneSpec = {
  id: string;
  scene: string;
  whatIf?: boolean;
  voice?: Record<number, {exaggeration?: number; cfg?: number}>;
  leadSeconds?: number;
  stars?: boolean;
  minSeconds?: number;
  sfx?: {beat: number; sound: Sound; offset?: number; volume?: number}[];
  /** number of narration beats the visuals are keyed to */
  beats: number;
};

/** Everything a language says (src/i18n/<locale>.ts). */
export type Locale = {
  title: string;
  subtitle: string;
  credits: Record<string, string>;
  chapters: Record<string, string>;
  whatIf: Record<string, string>;
  /** narration[cut][sceneId] = one string per beat; cuts other than 'full' override only some scenes */
  narration: Partial<Record<CutId, Record<string, string[]>>>;
  /** on-screen labels, keyed 'scene.name' */
  labels: Record<string, string>;
};

/** A video: which scenes, in what order, which beats to skip, which frame format. */
export type CutDef = {
  id: CutId;
  format: 'landscape' | 'vertical';
  scenes: string[];
  /** beats that take zero time in this cut, e.g. the ones that talk about numbers */
  skip?: Record<string, number[]>;
  /** timing knobs overridden for this cut (src/constants.ts T), e.g. a tighter short */
  pace?: Partial<{beatGap: number; silentBeat: number; tail: number; whatIfIntro: number}>;
  /** per-scene minimum length for this cut (replaces SceneSpec.minSeconds), e.g. a shorter end card */
  minSeconds?: Record<string, number>;
};

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
  /** Per-line voice overrides for neural engines, keyed by beat index (e.g. a calmer line). */
  voice?: Record<number, {exaggeration?: number; cfg?: number}>;
  /** Silent seconds before beat 0 (an animation-only lead-in). */
  leadSeconds?: number;
  /** Twinkling stars behind this scene (default true). Off whenever we look at the universe
   *  or the attractor swarm from outside: there are no stars out there. */
  stars?: boolean;
  /** Pad the scene to at least this long. */
  minSeconds?: number;
  /** offset: seconds after the beat starts; volume: 0..1 (default 0.4). */
  sfx?: {beat: number; sound: Sound; offset?: number; volume?: number}[];
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
