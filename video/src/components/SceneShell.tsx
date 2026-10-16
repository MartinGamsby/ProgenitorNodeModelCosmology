import React from 'react';
import {AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT, T} from '../theme';
import type {SceneDef, SceneTiming} from '../types';
import {Captions} from './Captions';

/** Big tilted "WHAT IF…?" stamp shown during the scene intro, then docked as a badge. */
const WhatIf: React.FC<{q: string; intro: number}> = ({q, intro}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const pop = spring({frame, fps, config: {damping: 9, mass: 0.7}});
  const out = interpolate(frame, [intro - 10, intro], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const qIn = spring({frame: frame - 10, fps, config: {damping: 14}});
  const dock = interpolate(frame, [intro - 4, intro + 14], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  return (
    <>
      {out < 1 && (
        <AbsoluteFill
          style={{
            justifyContent: 'center',
            alignItems: 'center',
            flexDirection: 'column',
            background: `rgba(4,6,13,${0.85 * (1 - out)})`,
            opacity: 1 - out,
            transform: `scale(${1 + out * 0.15})`,
          }}
        >
          <div
            style={{
              fontFamily: FONT.display,
              fontWeight: 900,
              fontSize: 210,
              color: C.bg,
              background: C.gold,
              padding: '0 50px',
              transform: `rotate(-4deg) scale(${pop})`,
              letterSpacing: -4,
              boxShadow: `0 0 80px ${C.gold}66`,
            }}
          >
            WHAT IF…
          </div>
          <div
            style={{
              marginTop: 50,
              fontFamily: FONT.display,
              fontWeight: 800,
              fontSize: 76,
              color: C.ink,
              opacity: qIn,
              transform: `translateY(${(1 - qIn) * 40}px)`,
              textAlign: 'center',
              maxWidth: 1500,
            }}
          >
            …{q}
          </div>
        </AbsoluteFill>
      )}
      {dock > 0 && (
        <div
          style={{
            position: 'absolute',
            top: 44,
            left: 56,
            display: 'flex',
            alignItems: 'center',
            gap: 16,
            opacity: dock,
            fontFamily: FONT.display,
          }}
        >
          <span style={{background: C.gold, color: C.bg, fontWeight: 900, fontSize: 30, padding: '4px 14px', transform: 'rotate(-3deg)'}}>
            WHAT IF…
          </span>
          <span style={{color: C.ink, fontWeight: 700, fontSize: 30}}>{q}</span>
        </div>
      )}
    </>
  );
};

const Chapter: React.FC<{label: string}> = ({label}) => {
  const frame = useCurrentFrame();
  const o = interpolate(frame, [8, 24], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  return (
    <div style={{position: 'absolute', top: 44, left: 56, display: 'flex', alignItems: 'center', gap: 14, opacity: o}}>
      <div style={{width: 8, height: 34, background: C.gold, borderRadius: 2}} />
      <div style={{fontFamily: FONT.display, fontWeight: 800, fontSize: 30, color: C.ink, letterSpacing: 1, textTransform: 'uppercase'}}>
        {label}
      </div>
    </div>
  );
};

export const SceneShell: React.FC<{def: SceneDef; timing: SceneTiming; children: React.ReactNode}> = ({def, timing, children}) => {
  const frame = useCurrentFrame();
  const fade = Math.min(
    interpolate(frame, [0, T.fade], [0, 1], {extrapolateRight: 'clamp'}),
    interpolate(frame, [timing.dur - T.fade, timing.dur], [1, 0], {extrapolateLeft: 'clamp'}),
  );
  return (
    <AbsoluteFill style={{opacity: fade}}>
      {children}
      {def.chapter && !def.whatIf && <Chapter label={def.chapter} />}
      <Captions timing={timing} />
      {def.whatIf && <WhatIf q={def.whatIf} intro={timing.intro} />}
    </AbsoluteFill>
  );
};
