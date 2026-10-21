import React, {useMemo} from 'react';
import {AbsoluteFill, interpolate, random} from 'remotion';
import {useBeats} from '../beats';
import {Show} from '../components/Show';
import {At, Card, Label, NodeDot, SvgDefs, Title} from '../components/ui';
import medium from '../../public/data/medium.json';
import {C, FONT, HEIGHT, WIDTH} from '../theme';
import {useT} from '../i18n';

const CX = WIDTH / 2;
const CY = HEIGHT / 2 - 20;

/** A 2D slice of the real relaxed medium around "us" (node 0 at the origin). */
const useSlice = (scale: number) =>
  useMemo(() => {
    const pos = medium.pos as number[][];
    const mass = medium.mass as number[];
    return pos
      .map((p, i) => ({x: CX + p[0] * scale, y: CY + p[1] * scale, z: p[2], m: mass[i]}))
      .filter((n) => Math.abs(n.z) < 70 && n.x > -60 && n.x < WIDTH + 60 && n.y > -60 && n.y < HEIGHT + 60);
  }, [scale]);

export const Progenitor: React.FC = () => {
  const tr = useT();
  const b = useBeats();
  const f = b.frame;
  const nodes = useSlice(3.4);
  const parts = useMemo(
    () =>
      new Array(420).fill(0).map((_, i) => ({
        a: random(`pa${i}`) * Math.PI * 2,
        r: Math.sqrt(random(`pr${i}`)),
        c: ['#cfe4ff', '#ffd9a8', '#ffffff', C.bubble][Math.floor(random(`pc${i}`) * 4)],
      })),
    [],
  );
  // timeline: bubble (b0) -> rewinds into a node (b1) -> wobble (b2) -> boom (b3) -> expansion (b4+)
  const rewind = b.p(1, 40);
  const wobble = b.on(2) && !b.on(3) ? Math.sin((f - b.at(2)) * 0.5) * 1.5 * b.p(2, 30) : 0;
  const boom = b.p(3, 6);
  const flash = interpolate(f, [b.at(3), b.at(3, 4), b.at(3, 30)], [0, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const expand = interpolate(f, [b.at(3), b.at(5)], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: (t) => 1 - (1 - t) ** 3});
  const split = b.p(6, 25);
  const mainO = 1 - split;
  const bubbleR = 120 * (1 - rewind);
  return (
    <AbsoluteFill>
      <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute', opacity: mainO}}>
        <SvgDefs />
        {nodes.map((n, i) => (
          <NodeDot key={i} x={n.x} y={n.y} r={14 + 10 * Math.cbrt(n.m)} opacity={0.9} />
        ))}
        {/* equal pulls on the progenitor (b2) */}
        {b.on(2) && !b.on(3) &&
          nodes.slice(0, 8).map((n, i) => (
            <line key={i} x1={CX} y1={CY} x2={CX + (n.x - CX) * 0.3} y2={CY + (n.y - CY) * 0.3} stroke={C.gold} strokeWidth={3} opacity={0.6 * b.p(2, 20)} />
          ))}
        {!b.on(3) && bubbleR > 2 && <circle cx={CX} cy={CY} r={bubbleR} fill="url(#bubble)" stroke={C.bubble} strokeWidth={3} />}
        {b.on(1, 20) && !b.on(3) && (
          <g transform={`translate(${wobble},${0})`}>
            {/* an ordinary member of the swarm: drawn like all the others */}
            <NodeDot x={CX} y={CY} r={14 + 10 * Math.cbrt(1)} opacity={rewind} />
          </g>
        )}
        {b.on(3) &&
          parts.map((p, i) => {
            const r = p.r * 300 * expand + 10 * boom;
            return <circle key={i} cx={CX + Math.cos(p.a) * r} cy={CY + Math.sin(p.a) * r} r={2.6} fill={p.c} opacity={0.9} />;
          })}
        {b.on(5) && <circle cx={CX} cy={CY} r={310} fill="none" stroke={C.bubble} strokeWidth={3} strokeDasharray="10 10" opacity={b.p(5, 20)} />}
      </svg>
      <AbsoluteFill style={{background: '#fff', opacity: flash}} />

      <Show from={0} until={1}>
        <At x={CX} y={CY - 200}>
          <Title size={120} color={C.gold}>
            {tr('progenitor.x')}</Title>
        </At>
      </Show>
      <Show from={1} until={3} offset={30}>
        <At x={CX} y={150}>
          <Card accent={C.gold}>
            <Title size={50}>
              {tr('progenitor.what_if_the_big')}<span style={{color: C.gold}}>{tr('progenitor.was_an_attractor')}</span>
            </Title>
          </Card>
        </At>
      </Show>
      <Show from={2} until={3} offset={20}>
        <At x={CX} y={CY + 120}>
          <Label size={32} color={C.gold} style={{fontFamily: FONT.display, fontWeight: 800}}>
            {tr('progenitor.balanced_for_now')}</Label>
        </At>
      </Show>
      <Show from={5} until={6}>
        <At x={CX} y={130}>
          <Card accent={C.bubble}>
            <Title size={48}>
              {tr('progenitor.starts_in_the_middle')}<span style={{color: C.bubble}}>{tr('progenitor.same_in_every_direction')}</span>
            </Title>
          </Card>
        </At>
      </Show>

      {/* beats 6-8: one mass, two epochs */}
      {split > 0 && (
        <AbsoluteFill style={{opacity: split}}>
          <At x={CX} y={140}>
            <Title size={76}>
              {tr('progenitor.one_mass')}<span style={{color: C.gold}}>{tr('progenitor.two_epochs')}</span>
            </Title>
          </At>
          <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute'}}>
            <SvgDefs />
            {/* BEFORE */}
            {[[-1, -1], [1, -1], [-1, 1], [1, 1], [0, -1.4], [0, 1.4], [-1.5, 0], [1.5, 0]].map(([dx, dy], i) => (
              <NodeDot key={i} x={520 + dx * 150} y={560 + dy * 120} r={22} />
            ))}
            <NodeDot x={520} y={560} r={22} />
            {/* arrow */}
            <text x={CX} y={580} fill={C.ink} fontSize={90} fontFamily={FONT.display} fontWeight={900} textAnchor="middle">
              →
            </text>
            {/* AFTER */}
            <circle cx={1400} cy={540} r={160} fill="url(#bubble)" stroke={C.bubble} strokeWidth={4} />
            <circle cx={1400} cy={540} r={235} fill="none" stroke={C.gold} strokeWidth={4} strokeDasharray="6 12" />
            {parts.slice(0, 220).map((p, i) => (
              <circle key={i} cx={1400 + Math.cos(p.a) * p.r * 225} cy={540 + Math.sin(p.a) * p.r * 225} r={2.4} fill={p.r < 0.7 ? p.c : C.gold} opacity={0.8} />
            ))}
          </svg>
          <At x={520} y={730}>
            <Label size={34} color={C.ink} style={{textAlign: 'center', fontWeight: 700}}>
              {tr('progenitor.before_one_member_of')}</Label>
          </At>
          <At x={1400} y={800}>
            <Label size={34} color={C.ink} style={{textAlign: 'center', fontWeight: 700, opacity: b.p(7, 20)}}>
              {tr('progenitor.after_our_universe')}<span style={{color: C.gold}}>{tr('progenitor.matter_beyond_the_horizon')}</span>
            </Label>
          </At>
        </AbsoluteFill>
      )}
      <Show from={8}>
        <At x={CX} y={300}>
          <Card accent={C.gold} style={{padding: '14px 30px'}}>
            <Title size={44} color={C.gold}>
              {tr('progenitor.m')}<sub style={{fontSize: 24}}>{tr('progenitor.progenitor')}</sub> {tr('progenitor.a_typical_member')}</Title>
          </Card>
        </At>
      </Show>
    </AbsoluteFill>
  );
};
