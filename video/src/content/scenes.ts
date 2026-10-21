/**
 * Scene structure, language-independent: which visual, stars, lead-in, voice overrides, sfx.
 * All text (narration, chapter, what-if question) lives in src/i18n/<locale>.ts, keyed by id.
 * Which scenes a video uses, and which beats it skips: src/cuts.ts.
 */
import type {SceneSpec} from '../types';

export const SCENE_SPECS: SceneSpec[] = [
  {
    id: "hook",
    scene: "Hook",
    stars: false,
    leadSeconds: 8.5,
    voice: {"1": {exaggeration: 0.55}},
    beats: 7,
  },
  {
    id: "title",
    scene: "Title",
    stars: false,
    minSeconds: 4.5,
    sfx: [{beat: 0, sound: "boom"}],
    beats: 1,
  },
  {
    id: "problem",
    scene: "Problem",
    beats: 8,
  },
  {
    id: "beyond",
    scene: "Beyond",
    stars: false,
    beats: 9,
  },
  {
    id: "tides",
    scene: "Tides",
    stars: false,
    beats: 6,
  },
  {
    id: "whatif-center",
    scene: "WhatIfCenter",
    stars: false,
    whatIf: true,
    beats: 4,
  },
  {
    id: "progenitor",
    scene: "Progenitor",
    stars: false,
    sfx: [{beat: 3, sound: "boom"}],
    beats: 9,
  },
  {
    id: "simulation",
    scene: "Simulation",
    stars: false,
    beats: 8,
  },
  {
    id: "data",
    scene: "Data",
    beats: 13,
  },
  {
    id: "whatif-observer",
    scene: "WhatIfObserver",
    stars: false,
    whatIf: true,
    beats: 3,
  },
  {
    id: "whatif-close",
    scene: "WhatIfClose",
    stars: false,
    whatIf: true,
    beats: 8,
  },
  {
    id: "whatif-mass",
    scene: "WhatIfMass",
    whatIf: true,
    beats: 4,
  },
  {
    id: "catch",
    scene: "Catch",
    stars: false,
    beats: 4,
  },
  {
    id: "predictions",
    scene: "Predictions",
    beats: 11,
  },
  {
    id: "whatif-invisible",
    scene: "WhatIfInvisible",
    stars: false,
    whatIf: true,
    beats: 5,
  },
  {
    id: "whatif-cycle",
    scene: "WhatIfCycle",
    stars: false,
    whatIf: true,
    beats: 5,
  },
  {
    id: "limits",
    scene: "Limits",
    beats: 3,
  },
  {
    id: "outro",
    scene: "Outro",
    stars: false,
    minSeconds: 20,
    beats: 4,
  },
];
