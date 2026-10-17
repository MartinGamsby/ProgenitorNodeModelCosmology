/**
 * CUTTING ROOM — scenes removed from the explainer, kept for future videos.
 *
 * Not rendered. To bring one back, move its entry into SCENES in script.ts (the components
 * are still registered in src/scenes/index.ts, under src/scenes/cut/).
 *
 * - tides-stretch: the original tides scene (tidal stretch of the bubble, 2GMR/S³ vs H0²ΩΛR).
 *   Cut 2026-10-16: too long and too technical for a first video.
 * - numbers: "how big, how far" (500 M_obs, S ≈ 39 Gpc back-of-envelope).
 * - teaser: 13 s standalone cold open with a sky-map montage and "That was the whole story"
 *   text. Replaced 2026-10-17 by the hook's silent lead-in (galaxies, no text, no bangs).
 * - center-shear: traceless tides / shear matrix and the 1/(S−R)² pull-growth plot.
 */
import type {SceneDef} from './types';

export const CUT_SCENES: SceneDef[] = [
  {
    id: 'teaser',
    stars: false,
    scene: 'Teaser',
    chapter: '',
    beats: [''],
    minSeconds: 13,
  },
  {
    id: 'tides-stretch',
    scene: 'TidesOriginal',
    chapter: 'Tides',
    beats: [
      'Quick gravity fact: a perfectly smooth shell of mass around you pulls on you… *not at all*.',
      "Every direction cancels. That's Newton's shell theorem.",
      'But *lumps* don’t cancel.',
      'The side of our bubble closer to a node gets pulled harder than the middle, and the middle harder than the far side.',
      "So the bubble gets *stretched*. That's a tide, the same thing the Moon does to our oceans.",
      'Do the math, and the tidal pull grows in proportion to distance: {2GMR/S³|two G M R over S cubed}.',
      "And dark energy's push? Also proportional to distance.",
      'Same shape. To gravity, a *pull from outside* can look exactly like a *push from inside*.',
    ],
  },
  {
    id: 'numbers',
    scene: 'Numbers',
    chapter: 'How big? How far?',
    beats: [
      'How heavy, and how far? Plug in the numbers.',
      "Nodes about five hundred times our observable universe's mass…",
      '…spaced about 39 {gigaparsecs|giga-parsecs} apart, would do the job.',
      'No exotic vacuum energy. Just ordinary gravity, from very big things, very far away.',
    ],
  },
  {
    id: 'center-shear',
    scene: 'WhatIfCenterOriginal',
    chapter: 'What if…',
    whatIf: 'we sat dead centre?',
    beats: [
      'What if we sat exactly in the middle, with nodes evenly all around us?',
      'Then every pull cancels. No acceleration at all.',
      'And a subtle twist: in empty space, tides can’t simply inflate a volume.',
      'They stretch one way and squeeze the others. Pure *shear*.',
      'The overall speed-up comes from the bubble creeping closer to the nodes, where the pull grows faster and faster.',
      'And that leftover stretch? It’s a *fingerprint*. Hold that thought.',
    ],
  },
];
