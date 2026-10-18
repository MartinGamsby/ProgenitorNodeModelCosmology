// Dependency-free constants (also read by scripts/frames.mjs).
export const FPS = 30;
export const WIDTH = 1920;
export const HEIGHT = 1080;

/** Timing knobs: seconds, except `fade` (frames). */
export const T = {
  fade: 12,
  whatIfIntro: 2.6, // the "WHAT IF…" stamp before beat 0
  beatGap: 0.7, // silence after each spoken line
  silentBeat: 1.4, // '' beats
  tail: 1.5, // hold after the last beat, so the viewer can digest
};
