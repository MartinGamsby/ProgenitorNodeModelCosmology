import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {captionTokens} from '../markup';
import {C, FONT} from '../theme';
import type {SceneTiming} from '../types';

/** "10^120" -> 10<sup>120</sup> (caption-only math). */
const withSup = (w: string) => {
  const m = w.match(/^(.*?)\^(-?[\d.]+)(.*)$/);
  if (!m) return w;
  return (
    <>
      {m[1]}
      <sup style={{fontSize: '0.6em'}}>{m[2].replace('-', '−')}</sup>
      {m[3]}
    </>
  );
};

/** YouTube-style subtitles: the current line, words lighting up as they are spoken. */
export const Captions: React.FC<{timing: SceneTiming}> = ({timing}) => {
  const frame = useCurrentFrame();
  const i = timing.beats.findIndex((b) => frame >= b.from && frame < b.from + b.dur);
  if (i < 0) return null;
  const b = timing.beats[i];
  const tokens = captionTokens(b.text);
  if (!tokens.length) return null;
  const local = frame - b.from;
  const speak = Math.max(1, b.dur - 14); // words are lit across the spoken part
  const lit = interpolate(local, [0, speak], [0, tokens.length], {extrapolateRight: 'clamp'});
  const appear = interpolate(local, [0, 6], [0, 1], {extrapolateRight: 'clamp'});
  return (
    <div
      style={{
        position: 'absolute',
        left: 0,
        right: 0,
        bottom: 54,
        display: 'flex',
        justifyContent: 'center',
        opacity: appear,
        transform: `translateY(${(1 - appear) * 10}px)`,
      }}
    >
      <div
        style={{
          maxWidth: 1500,
          padding: '14px 28px',
          borderRadius: 14,
          background: 'rgba(4,6,13,0.72)',
          fontFamily: FONT.body,
          fontWeight: 600,
          fontSize: 40,
          lineHeight: 1.3,
          textAlign: 'center',
          color: C.ink,
        }}
      >
        {tokens.map((t, k) => {
          const on = Math.min(1, Math.max(0, lit - k));
          return (
            <span
              key={k}
              style={{
                color: t.em ? C.gold : C.ink,
                opacity: 0.38 + 0.62 * on,
                fontWeight: t.em ? 800 : 600,
              }}
            >
              {withSup(t.text)}
              {k < tokens.length - 1 ? ' ' : ''}
            </span>
          );
        })}
      </div>
    </div>
  );
};
