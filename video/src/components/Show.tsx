import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {useBeats} from '../beats';

/**
 * Visible from beat `from` until beat `until` (exclusive; omit = scene end), fading in/out.
 * `offset` shifts the start in frames. Children may be a function of the fade-in progress.
 */
export const Show: React.FC<{
  from: number;
  until?: number;
  offset?: number;
  fade?: number;
  rise?: number;
  children: React.ReactNode | ((p: number) => React.ReactNode);
  style?: React.CSSProperties;
}> = ({from, until, offset = 0, fade = 12, rise = 24, children, style}) => {
  const frame = useCurrentFrame();
  const b = useBeats();
  const a = b.at(from, offset);
  const z = until === undefined ? b.dur + 100 : b.at(until);
  if (frame < a || frame > z + fade) return null;
  const pin = interpolate(frame, [a, a + fade], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const pout = until === undefined ? 1 : interpolate(frame, [z, z + fade], [1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const o = Math.min(pin, pout);
  return (
    <div style={{position: 'absolute', inset: 0, opacity: o, transform: `translateY(${(1 - pin) * rise}px)`, ...style}}>
      {typeof children === 'function' ? children(pin) : children}
    </div>
  );
};
