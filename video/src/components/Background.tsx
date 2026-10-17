import React, {useMemo} from 'react';
import {AbsoluteFill, random, useCurrentFrame} from 'remotion';
import {C, HEIGHT, WIDTH} from '../theme';

/** Deterministic twinkling starfield + soft nebula glow. `drift` pans the stars slowly. */
export const Starfield: React.FC<{seed?: string; count?: number; drift?: number; opacity?: number; starsOnly?: boolean}> = ({
  seed = 'stars',
  count = 260,
  drift = 0.15,
  opacity = 1,
  starsOnly = false,
}) => {
  const frame = useCurrentFrame();
  const stars = useMemo(
    () =>
      new Array(count).fill(0).map((_, i) => ({
        x: random(`${seed}x${i}`) * WIDTH,
        y: random(`${seed}y${i}`) * HEIGHT,
        r: 0.4 + random(`${seed}r${i}`) ** 3 * 2.2,
        z: 0.3 + random(`${seed}z${i}`) * 0.7,
        ph: random(`${seed}p${i}`) * Math.PI * 2,
      })),
    [seed, count],
  );
  return (
    <AbsoluteFill style={{background: starsOnly ? undefined : `radial-gradient(ellipse at 30% 20%, ${C.bg2} 0%, ${C.bg} 65%)`, opacity}}>
      {!starsOnly && (
        <AbsoluteFill
          style={{
            background:
              'radial-gradient(circle at 78% 70%, rgba(125,77,255,0.10) 0%, transparent 40%), radial-gradient(circle at 15% 85%, rgba(78,168,255,0.08) 0%, transparent 35%)',
          }}
        />
      )}
      <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute'}}>
        {stars.map((s, i) => {
          const x = (s.x - frame * drift * s.z + WIDTH * 10) % WIDTH;
          const tw = 0.55 + 0.45 * Math.sin(frame * 0.05 * s.z + s.ph);
          return <circle key={i} cx={x} cy={s.y} r={s.r} fill="#fff" opacity={tw * s.z} />;
        })}
      </svg>
    </AbsoluteFill>
  );
};
