import React, {createContext, useContext} from 'react';
import {Easing, interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import type {SceneDef, SceneTiming} from './types';

type Ctx = {timing: SceneTiming; def: SceneDef};
const BeatCtx = createContext<Ctx | null>(null);

export const BeatProvider: React.FC<Ctx & {children: React.ReactNode}> = ({children, ...v}) => (
  <BeatCtx.Provider value={v}>{children}</BeatCtx.Provider>
);

const ease = Easing.bezier(0.33, 0, 0.2, 1);

/**
 * Animation helpers keyed to narration beats. Inside a scene:
 *   const b = useBeats();
 *   b.at(2)              frame where beat 2 starts (scene-relative)
 *   b.p(2, 20)           0→1 eased over 20 frames from beat 2 start (+ optional offset)
 *   b.s(2)               spring 0→1 from beat 2
 *   b.on(2)              true once beat 2 has started
 *   b.cur                index of the beat currently playing
 */
export const useBeats = () => {
  const ctx = useContext(BeatCtx);
  if (!ctx) throw new Error('useBeats outside a scene');
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const {beats, dur} = ctx.timing;
  const at = (i: number, offset = 0) =>
    (i >= beats.length ? dur : i < 0 ? 0 : beats[i].from) + offset;
  const p = (i: number, len = 20, offset = 0) =>
    interpolate(frame, [at(i, offset), at(i, offset) + len], [0, 1], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
      easing: ease,
    });
  const s = (i: number, offset = 0, damping = 14) =>
    spring({frame: frame - at(i, offset), fps, config: {damping, mass: 0.8}});
  const on = (i: number, offset = 0) => frame >= at(i, offset);
  let cur = -1;
  beats.forEach((b, i) => {
    if (frame >= b.from) cur = i;
  });
  /** progress through the whole beat i (0→1 linear) */
  const through = (i: number) =>
    interpolate(frame, [at(i), at(i + 1)], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  return {frame, fps, at, p, s, on, cur, through, dur, timing: ctx.timing, def: ctx.def};
};

/** Generic eased range helper. */
export const range = (frame: number, a: number, b: number, from = 0, to = 1) =>
  interpolate(frame, [a, b], [from, to], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: ease});
