// Dependency-free constants (also read by scripts/frames.mjs).
export const FPS = 30;
export const WIDTH = 1920;
export const HEIGHT = 1080;

/** Vertical cuts (1080×1920): each 1920×1080 scene is scaled ×0.68 into a stage across the middle
 *  (its outer ~9 % per side is cropped), the title sits above it, big captions below (kept above
 *  the platform UI at the bottom). */
export const VERTICAL = {width: 1080, height: 1920, scale: 0.68, stageY: 470, captionY: 1270};

/** Timing knobs: seconds, except `fade` (frames). */
export const T = {
  fade: 12,
  whatIfIntro: 2.6, // the "WHAT IF…" stamp before beat 0
  beatGap: 0.7, // silence after each spoken line
  silentBeat: 1.4, // '' beats
  tail: 1.5, // hold after the last beat, so the viewer can digest
};
