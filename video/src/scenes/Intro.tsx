import React, {useMemo} from 'react';
import {AbsoluteFill, interpolate, random, useCurrentFrame} from 'remotion';
import {useBeats} from '../beats';
import {Show} from '../components/Show';
import {Space3D} from '../components/Space3D';
import {Arrow, At, Label, NodeDot, SvgDefs, Title} from '../components/ui';
import medium from '../../public/data/medium.json';
import {VIDEO_SUBTITLE, VIDEO_TITLE} from '../script';
import {C, FONT, HEIGHT, WIDTH} from '../theme';

const CX = WIDTH / 2;
const CY = HEIGHT / 2 - 40;

/** Little spiral-ish galaxy glyph. */
const Galaxy: React.FC<{x: number; y: number; s: number; rot: number; hue: string; o: number}> = ({x, y, s, rot, hue, o}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={o}>
    <ellipse rx={14} ry={5} fill={hue} opacity={0.35} />
    <ellipse rx={7} ry={2.6} fill={hue} opacity={0.7} />
    <circle r={2.2} fill="#fff" />
  </g>
);

export const Hook: React.FC = () => {
  const b = useBeats();
  const frame = b.frame;
  const gal = useMemo(
    () =>
      new Array(140).fill(0).map((_, i) => {
        const ang = random(`ga${i}`) * Math.PI * 2;
        const r = 30 + Math.sqrt(random(`gr${i}`)) * 520;
        return {
          ang,
          r,
          s: 0.6 + random(`gs${i}`) * 1.4,
          rot: random(`gq${i}`) * 180,
          hue: ['#9cc9ff', '#ffd9a8', '#e0c4ff', '#ffffff'][Math.floor(random(`gh${i}`) * 4)],
        };
      }),
    [],
  );
  // Expansion factor: slow at first, then accelerating (that's the point).
  const t = frame / b.fps;
  const a = 0.45 + 0.05 * t + 0.0042 * t * t * t * 0.4;
  const pushP = b.p(5, 18);
  const pullP = b.p(6, 18);
  const pieP = b.p(3, 40);
  const galO = interpolate(frame, [b.at(2), b.at(2) + 15, b.at(5), b.at(5) + 15], [1, 0.25, 0.25, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const nodes = [0, 1, 2, 3, 4, 5].map((k) => {
    const ang = (k / 6) * Math.PI * 2 + 0.4;
    return {x: CX + Math.cos(ang) * 760, y: CY + Math.sin(ang) * 420, ang};
  });
  return (
    <AbsoluteFill>
      <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute'}}>
        <SvgDefs />
        {gal.map((g, i) => {
          const r = g.r * a;
          const x = CX + Math.cos(g.ang) * r * 1.5;
          const y = CY + Math.sin(g.ang) * r * 0.85;
          return <Galaxy key={i} x={x} y={y} s={g.s} rot={g.rot} hue={g.hue} o={galO} />;
        })}
        {/* speed arrows on a few galaxies during beat 1 */}
        {b.on(1) &&
          !b.on(2) &&
          gal.slice(0, 18).map((g, i) => {
            const r = g.r * a;
            const x = CX + Math.cos(g.ang) * r * 1.5;
            const y = CY + Math.sin(g.ang) * r * 0.85;
            const L = 20 + r * 0.18 * (1 + b.through(1));
            return (
              <Arrow key={i} x1={x} y1={y} x2={x + Math.cos(g.ang) * L} y2={y + Math.sin(g.ang) * L * 0.6} color={C.gold} width={3} head={10} p={b.p(1, 12)} />
            );
          })}
        {/* PUSH: arrows from the centre outward */}
        {b.on(5) &&
          [0, 1, 2, 3, 4, 5, 6, 7].map((k) => {
            const ang = (k / 8) * Math.PI * 2;
            return (
              <Arrow
                key={k}
                x1={CX + Math.cos(ang) * 60}
                y1={CY + Math.sin(ang) * 40}
                x2={CX + Math.cos(ang) * 300}
                y2={CY + Math.sin(ang) * 190}
                color={C.hot}
                width={9}
                head={28}
                p={pushP}
                opacity={1 - pullP}
              />
            );
          })}
        {/* PULL: dark nodes outside, arrows towards them */}
        {b.on(6) &&
          nodes.map((n, k) => (
            <g key={k}>
              <NodeDot x={n.x} y={n.y} r={34} opacity={pullP} />
              <Arrow
                x1={CX + Math.cos(n.ang) * 330}
                y1={CY + Math.sin(n.ang) * 190}
                x2={n.x - Math.cos(n.ang) * 120}
                y2={n.y - Math.sin(n.ang) * 80}
                color={C.node}
                width={9}
                head={28}
                p={b.p(6, 20, 8)}
              />
            </g>
          ))}
      </svg>

      <Show from={2} until={4}>
        <At x={CX} y={interpolate(frame, [b.at(3), b.at(3) + 20], [CY - 40, 200], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}>
          <Title size={interpolate(frame, [b.at(3), b.at(3) + 20], [150, 100], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})} style={{textShadow: `0 0 60px ${C.nodeGlow}`, letterSpacing: 6, whiteSpace: 'nowrap'}}>
            DARK ENERGY
          </Title>
        </At>
        <At x={CX} y={CY + 90}>
          <Title size={120} color={C.gold} style={{opacity: interpolate(frame, [b.at(3) - 20, b.at(3)], [1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}}>
            ???
          </Title>
        </At>
      </Show>

      <Show from={3} until={4} offset={6}>
        <Pie p={pieP} />
      </Show>

      <Show from={5} until={6}>
        <At x={CX} y={CY}>
          <Title size={110} color={C.hot} style={{textShadow: `0 0 30px ${C.bg}, 0 0 60px ${C.bg}`}}>
            PUSHING?
          </Title>
        </At>
      </Show>
      <Show from={6}>
        <At x={CX} y={CY}>
          <Title size={110} color={C.node} style={{textShadow: `0 0 40px ${C.nodeGlow}, 0 0 30px ${C.bg}`, whiteSpace: 'nowrap'}}>
            …OR PULLING?
          </Title>
        </At>
      </Show>
    </AbsoluteFill>
  );
};

const Pie: React.FC<{p: number}> = ({p}) => {
  const parts = [
    {v: 0.68, c: C.node, l: 'Dark energy 68%'},
    {v: 0.27, c: '#5a6ea8', l: 'Dark matter 27%'},
    {v: 0.05, c: C.gold, l: 'Everything we see 5%'},
  ];
  const R = 150;
  const x0 = WIDTH / 2;
  const y0 = HEIGHT / 2 + 150;
  let acc = 0;
  return (
    <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute'}}>
      {parts.map((s, i) => {
        const a0 = acc * Math.PI * 2 * p - Math.PI / 2;
        acc += s.v;
        const a1 = acc * Math.PI * 2 * p - Math.PI / 2;
        const large = a1 - a0 > Math.PI ? 1 : 0;
        const d = `M${x0},${y0} L${x0 + R * Math.cos(a0)},${y0 + R * Math.sin(a0)} A${R},${R} 0 ${large} 1 ${x0 + R * Math.cos(a1)},${y0 + R * Math.sin(a1)} Z`;
        return <path key={i} d={d} fill={s.c} opacity={0.9} />;
      })}
      <circle cx={x0} cy={y0} r={R * 0.55} fill={C.bg} />
      {parts.map((s, i) => (
        <g key={`l${i}`} opacity={p}>
          <rect x={x0 + 230} y={y0 - 70 + i * 52} width={26} height={26} rx={5} fill={s.c} />
          <text x={x0 + 270} y={y0 - 48 + i * 52} fill={C.ink} fontFamily={FONT.body} fontWeight={700} fontSize={30}>
            {s.l}
          </text>
        </g>
      ))}
    </svg>
  );
};

export const TitleCard: React.FC = () => {
  const b = useBeats();
  const frame = b.frame;
  const nodes = medium.pos as [number, number, number][];
  const mass = medium.mass as number[];
  const pop = b.s(0, 0, 11);
  const yaw = frame * 0.004;
  return (
    <AbsoluteFill>
      <Space3D
        width={WIDTH}
        height={HEIGHT}
        camera={{yaw, pitch: 0.35, distance: 520, scale: 1.25}}
        layers={[
          {kind: 'glows', pts: nodes, radius: (i) => 5 + 4 * Math.cbrt(mass[i]), color: C.nodeGlow, core: C.node, alpha: 0.5 * interpolate(frame, [0, 40], [0, 1], {extrapolateRight: 'clamp'})},
        ]}
        style={{position: 'absolute'}}
      />
      <AbsoluteFill style={{background: 'radial-gradient(circle at 50% 50%, rgba(4,6,13,0.85) 0%, rgba(4,6,13,0.2) 70%)'}} />
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', flexDirection: 'column', gap: 34}}>
        <Title
          size={104}
          style={{
            textAlign: 'center',
            maxWidth: 1600,
            transform: `scale(${0.8 + 0.2 * pop})`,
            opacity: pop,
            textShadow: `0 0 50px ${C.nodeGlow}aa`,
          }}
        >
          {VIDEO_TITLE}
        </Title>
        <Label size={42} color={C.gold} style={{opacity: b.p(0, 20, 20), letterSpacing: 4, textTransform: 'uppercase', fontFamily: FONT.display, fontWeight: 800}}>
          {VIDEO_SUBTITLE}
        </Label>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
