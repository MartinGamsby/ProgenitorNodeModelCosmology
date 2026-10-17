import React, {useMemo} from 'react';
import {AbsoluteFill, interpolate, random, useCurrentFrame} from 'remotion';
import {Space3D, V3} from '../../components/Space3D';
import {Arrow, NodeDot, SvgDefs, Title} from '../../components/ui';
import medium from '../../../public/data/medium.json';
import {C, FONT, HEIGHT, WIDTH} from '../../theme';

const CX = WIDTH / 2;
const CY = HEIGHT / 2;
const sec = (s: number) => Math.round(s * 30);
const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;
const win = (f: number, a: number, b: number, fade = 10) =>
  Math.min(interpolate(f, [sec(a), sec(a) + fade], [0, 1], clamp), interpolate(f, [sec(b) - fade, sec(b)], [1, 0], clamp));

/**
 * Cold-open teaser (no narration): a fast glimpse of the whole story — the attractor swarm, a
 * bang, the metabolism cycle, the lopsided sky — then "That was the whole story, in ten seconds. Now let's take it step by step." Timing is in seconds below; the scene's minSeconds in script.ts must cover it.
 */
export const Teaser: React.FC = () => {
  const f = useCurrentFrame();
  const nodes = medium.pos as V3[];
  const mass = medium.mass as number[];
  const parts = useMemo(
    () => new Array(260).fill(0).map((_, i) => ({a: random(`ta${i}`) * Math.PI * 2, r: Math.sqrt(random(`tr${i}`)), toB: i % 3 !== 0})),
    [],
  );

  // 0 – 3.2 s: swarm, camera diving in
  const swarmO = win(f, 0, 3.4, 12);
  const scale = interpolate(f, [0, sec(3.2)], [1.0, 4.5], clamp);
  // 2.8 – 5.4 s: a bang
  const bangO = win(f, 2.8, 5.6, 8);
  const flash = interpolate(f, [sec(3.0), sec(3.1), sec(3.7)], [0, 1, 0], clamp);
  const burst = interpolate(f, [sec(3.0), sec(5.2)], [0, 1], {...clamp, easing: (t) => 1 - (1 - t) ** 3});
  // 5.2 – 8.2 s: neighbours feed, one swells and bangs
  const cycO = win(f, 5.2, 8.4, 8);
  const feed = interpolate(f, [sec(5.3), sec(6.8)], [0, 1], clamp);
  const swell = interpolate(f, [sec(6.6), sec(7.4)], [0, 1], clamp);
  const flash2 = interpolate(f, [sec(7.4), sec(7.5), sec(8.0)], [0, 1, 0], clamp);
  const burst2 = interpolate(f, [sec(7.4), sec(8.4)], [0, 1], {...clamp, easing: (t) => 1 - (1 - t) ** 3});
  // 8.0 – 10.4 s: the lopsided sky, everything pointing one way
  const skyO = win(f, 8.0, 10.6, 8);
  const tilt = interpolate(f, [sec(8.0), sec(9.0)], [0, 1], clamp);
  // 10.2 s →: the punchline
  const t1 = win(f, 10.2, 14, 8);
  const t2 = interpolate(f, [sec(11.0), sec(11.4)], [0, 1], clamp);

  const B = {x: CX + 260, y: CY - 90};
  const Cn = {x: CX - 230, y: CY + 140};
  const AX = -0.45;
  const MARK = {x: CX + Math.cos(AX) * 640, y: CY + Math.sin(AX) * 640};
  return (
    <AbsoluteFill style={{background: '#000'}}>
      {swarmO > 0 && (
        <Space3D
          width={WIDTH}
          height={HEIGHT}
          camera={{yaw: f * 0.012, pitch: 0.3, distance: (3 * 540) / scale, scale}}
          layers={[
            {kind: 'glows', pts: nodes, radius: (i) => (5 + 6 * Math.cbrt(mass[i])) * Math.sqrt(scale), color: C.nodeGlow, core: C.node, alpha: swarmO},
            {kind: 'sphere', center: [0, 0, 0], radius: 14, color: C.gold, alpha: swarmO},
          ]}
          style={{position: 'absolute'}}
        />
      )}
      <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute'}}>
        <SvgDefs />
        {bangO > 0 && (
          <g opacity={bangO}>
            {parts.map((p, i) => (
              <circle key={i} cx={CX + Math.cos(p.a) * p.r * 420 * burst} cy={CY + Math.sin(p.a) * p.r * 360 * burst} r={2.8} fill={i % 2 ? '#cfe4ff' : '#ffd9a8'} />
            ))}
          </g>
        )}
        {cycO > 0 && (
          <g opacity={cycO}>
            {parts.map((p, i) => {
              const ox = CX + Math.cos(p.a) * p.r * 420;
              const oy = CY + Math.sin(p.a) * p.r * 360;
              const T = p.toB ? B : Cn;
              const u = Math.min(1, feed * 1.3 - (i % 10) * 0.03);
              if (u >= 0.97) return null;
              return <circle key={i} cx={ox + (T.x - ox) * Math.max(0, u)} cy={oy + (T.y - oy) * Math.max(0, u)} r={2.8} fill={i % 2 ? '#cfe4ff' : '#ffd9a8'} />;
            })}
            <NodeDot x={Cn.x} y={Cn.y} r={28 + 14 * feed} />
            {burst2 === 0 && <NodeDot x={B.x + (swell > 0 ? Math.sin(f * 1.1) * 5 * swell : 0)} y={B.y} r={30 + 18 * feed + 22 * swell} color={swell > 0 ? C.hot : C.node} />}
            {burst2 > 0 &&
              parts.map((p, i) => (
                <circle key={`b${i}`} cx={B.x + Math.cos(p.a + 0.3) * p.r * 380 * burst2} cy={B.y + Math.sin(p.a + 0.3) * p.r * 320 * burst2} r={2.8} fill={i % 2 ? '#cfe4ff' : '#ffd9a8'} />
              ))}
          </g>
        )}
        {skyO > 0 && (
          <g opacity={skyO}>
            <defs>
              <clipPath id="tsky">
                <ellipse cx={CX} cy={CY} rx={560} ry={280} />
              </clipPath>
              <linearGradient id="tdip" x1={0.05} y1={0.75} x2={0.95} y2={0.25}>
                <stop offset="0%" stopColor={C.lcdm} stopOpacity={0.55 * tilt} />
                <stop offset="50%" stopColor="#1a2140" stopOpacity={0.2} />
                <stop offset="100%" stopColor={C.hot} stopOpacity={0.6 * tilt} />
              </linearGradient>
            </defs>
            <ellipse cx={CX} cy={CY} rx={560} ry={280} fill="#0b1226" stroke={C.dim} strokeWidth={3} />
            <g clipPath="url(#tsky)">
              <rect x={CX - 560} y={CY - 280} width={1120} height={560} fill="url(#tdip)" />
              <g transform={`rotate(${(AX * 180) / Math.PI} ${CX} ${CY})`} opacity={tilt}>
                {[-1, 1].map((k) => (
                  <ellipse key={k} cx={CX + k * 230} cy={CY} rx={150} ry={70} fill="none" stroke={C.gold} strokeWidth={5} />
                ))}
              </g>
              {[-2, -1, 0, 1].flatMap((i) =>
                [-1, 0, 1].map((j) => {
                  const x = CX - 300 + i * 130;
                  const y = CY + j * 110;
                  const dl = Math.hypot(MARK.x - x, MARK.y - y);
                  const tt = ((f * 3 + i * 17 + j * 29) % 45) / 45;
                  return (
                    <Arrow key={`${i}${j}`} x1={x + ((MARK.x - x) / dl) * 60 * tt} y1={y + ((MARK.y - y) / dl) * 60 * tt} x2={x + ((MARK.x - x) / dl) * 60 * (tt + 0.8)} y2={y + ((MARK.y - y) / dl) * 60 * (tt + 0.8)} color={C.eds} width={4} head={13} opacity={tilt} />
                  );
                }),
              )}
            </g>
            <line x1={CX} y1={CY} x2={MARK.x} y2={MARK.y} stroke={C.gold} strokeWidth={4} strokeDasharray="14 10" opacity={tilt} />
            <NodeDot x={MARK.x} y={MARK.y} r={26} opacity={tilt} />
          </g>
        )}
      </svg>
      <AbsoluteFill style={{background: '#fff', opacity: Math.max(flash, flash2) * 0.9}} />
      {t1 > 0 && (
        <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', flexDirection: 'column', gap: 24, opacity: t1, background: 'rgba(0,0,0,0.75)'}}>
          <Title size={70}>That was the whole story, in ten seconds.</Title>
          <Title size={56} color={C.gold} style={{opacity: t2, fontFamily: FONT.display}}>
            Now let’s take it step by step.
          </Title>
        </AbsoluteFill>
      )}
    </AbsoluteFill>
  );
};
