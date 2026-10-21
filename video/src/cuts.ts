/**
 * THE VIDEOS ("cuts"). Each cut picks scenes from src/content/scenes.ts, in order, and may skip
 * beats (zero time: their visuals never show). Text per cut: src/i18n/<locale>.ts narration[cut].
 */
import type {CutDef, CutId} from './types';
import {SCENE_SPECS} from './content/scenes';

export const CUTS: Record<CutId, CutDef> = {
  full: {id: 'full', format: 'landscape', scenes: SCENE_SPECS.map((s) => s.id)},
  // filled in phase 4 / 5
  idea: {id: 'idea', format: 'landscape', scenes: SCENE_SPECS.map((s) => s.id)},
  short: {id: 'short', format: 'vertical', scenes: ['hook']},
};
