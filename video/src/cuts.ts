/**
 * THE VIDEOS ("cuts"). Each cut picks scenes from src/content/scenes.ts, in order, and may skip
 * beats (zero time: their visuals never show). Text per cut: src/i18n/<locale>.ts narration[cut].
 */
import type {CutDef, CutId} from './types';
import {SCENE_SPECS} from './content/scenes';

export const CUTS: Record<CutId, CutDef> = {
  full: {id: 'full', format: 'landscape', scenes: SCENE_SPECS.filter((s) => !s.variant).map((s) => s.id)},
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
  // "The hollow planet" (vertical, English only for now): no cold open or title, large graphics
  hollow: {
    id: 'hollow',
    format: 'vertical',
    locales: ['en'],
    verticalTitle: false,
    scenes: ['tides', 'beyond', 'hook-expansion', 'tides-universe', 'beyond-swarm', 'tides-stretch', 'hook-racing', 'simulation', 'data', 'limits', 'predictions', 'outro'],
    skip: {
      tides: [4, 5],
      beyond: [1, 2, 3, 4, 5, 6, 7, 8],
      'hook-expansion': [0, 3, 4, 5, 6],
      'tides-universe': [0, 1, 3, 4, 5],
      'beyond-swarm': [0, 1, 2, 3, 5, 6, 7],
      'tides-stretch': [0, 1, 2, 5],
      'hook-racing': [0, 2, 3, 4, 5, 6],
      simulation: [0, 1, 2, 3, 4, 5, 7],
      data: [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12],
      limits: [1, 2],
      predictions: [0, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      outro: [0, 1, 3],
    },
    pace: {beatGap: 0.35, tail: 0.3},
    hold: {
      'beyond-swarm': {8: 1.2}, // "you are here" appears partway through a short line
      data: {8: 1.2}, // the fit-score bars are the proof: time to read them
      outro: {2: 1.6}, // the last line, then the end fade over the swarm
    },
    minSeconds: {outro: 3},
    stage: {
      '*': {scale: 1},
      tides: {scale: 0.92},
      beyond: {scale: 0.8},
      'tides-universe': {scale: 0.92},
      'tides-stretch': {scale: 0.92},
      'hook-expansion': {scale: 0.8},
      'hook-racing': {scale: 0.8},
      simulation: {scale: 1, focusX: 490},
      data: {scale: 0.75},
      limits: {scale: 0.85, focusY: 300},
      predictions: {scale: 0.82, focusY: 500},
      outro: {scale: 0.8},
    },
  },
};
