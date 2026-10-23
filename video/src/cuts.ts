/**
 * THE VIDEOS ("cuts"). Each cut picks scenes from src/content/scenes.ts, in order, and may skip
 * beats (zero time: their visuals never show). Text per cut: src/i18n/<locale>.ts narration[cut].
 */
import type {CutDef, CutId} from './types';
import {SCENE_SPECS} from './content/scenes';

export const CUTS: Record<CutId, CutDef> = {
  full: {id: 'full', format: 'landscape', scenes: SCENE_SPECS.map((s) => s.id)},
  // the idea, no simulation details; beats that talk about numbers are skipped
  idea: {
    id: 'idea',
    format: 'landscape',
    scenes: ['hook', 'title', 'problem', 'beyond', 'tides', 'whatif-center', 'progenitor', 'predictions', 'whatif-invisible', 'whatif-cycle', 'outro'],
    skip: {
      hook: [3], // 70 % pie
      problem: [4, 5, 6], // two problems: 10^120, coincidence chart
      beyond: [1], // 46 billion light-years
      predictions: [2, 3, 4, 9], // 67 vs 73, w chart
      'whatif-invisible': [3], // 10^-33 kelvin
    },
  },
  // ~1 min vertical (Shorts / TikTok): one idea per line, own narration
  short: {
    id: 'short',
    format: 'vertical',
    scenes: ['hook', 'beyond', 'tides', 'progenitor', 'predictions', 'outro'],
    skip: {
      hook: [3, 4],
      beyond: [0, 1, 2, 3, 5, 7],
      tides: [0, 1, 3, 4],
      progenitor: [0, 2, 3, 5, 6, 7, 8],
      predictions: [0, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      outro: [0],
    },
    pace: {beatGap: 0.3, tail: 0.5},
    minSeconds: {outro: 10}, // the paper card stays up ~3 s
  },
};
