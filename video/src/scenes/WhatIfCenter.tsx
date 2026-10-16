import React from 'react';
import {AbsoluteFill} from 'remotion';
import {useBeats} from '../beats';
import {Axes, LegendItem, scales} from '../components/Chart';
import {Show} from '../components/Show';
import {Arrow, At, Card, Label, NodeDot, pathOf, partial, SvgDefs, Tex, Title} from '../components/ui';
import {C, FONT, HEIGHT, WIDTH} from '../theme';

const CX = WIDTH / 2;
const CY = HEIGHT / 2 - 10;

export const WhatIfCenter: React.FC = () => {
  const b = useBeats();
  const sp = 300;
  const lattice: [number, number][] = [];
  for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) if (i || j) lattice.push([CX + i * sp * 1.25, CY + j * sp * 0.9]);
  const latO = 1 - b.p(2, 15);
  const shear = b.p(3, 40);
  const R = 150;

  // beat 4: exact vs linear tidal pull vs R/S
  const box = {x: 520, y: 210, w: 900, h: 520};
  const {X, Y} = scales(box, [0, 0.7], [0, 6]);
  const xs = new Array(80).fill(0).map((_, i) => (i / 79) * 0.7);
  const exact = xs.filter((u) => 1 / (1 - u) ** 2 - 1 <= 6).map((u) => [X(u), Y(1 / (1 - u) ** 2 - 1)] as [number, number]);
  const linear = xs.map((u) => [X(u), Y(2 * u)] as [number, number]);
  return (
    <AbsoluteFill>
      <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute'}}>
        <SvgDefs />
        {!b.on(2, 15) && (
          <g opacity={latO}>
            {lattice.map(([x, y], i) => (
              <g key={i}>
                <NodeDot x={x} y={y} r={30} opacity={b.p(0, 20, i * 3)} />
                <Arrow x1={CX} y1={CY} x2={CX + (x - CX) * 0.45} y2={CY + (y - CY) * 0.45} color={C.gold} width={6} head={20} p={b.p(0, 25, 30)} opacity={1 - b.p(1, 30, 20)} />
              </g>
            ))}
            <circle cx={CX} cy={CY} r={70} fill="url(#bubble)" stroke={C.bubble} strokeWidth={3} />
          </g>
        )}
        {b.on(2) && !b.on(4) && (
          <g opacity={Math.min(b.p(2, 15), 1 - b.p(4, 12))}>
            <ellipse cx={CX} cy={CY} rx={R} ry={R} fill="none" stroke={C.faint} strokeWidth={3} strokeDasharray="10 8" />
            <ellipse cx={CX} cy={CY} rx={R * (1 + 0.45 * shear)} ry={R / (1 + 0.45 * shear)} fill="url(#bubble)" stroke={C.bubble} strokeWidth={4} />
            {[-1, 1].map((s) => (
              <g key={s}>
                <Arrow x1={CX + s * (R + 40)} y1={CY} x2={CX + s * (R + 160)} y2={CY} color={C.hot} width={8} head={24} p={shear} />
                <Arrow x1={CX} y1={CY + s * (R + 150)} x2={CX} y2={CY + s * (R + 40)} color={C.lcdm} width={8} head={24} p={shear} />
              </g>
            ))}
            <text x={CX + R + 190} y={CY + 12} fill={C.hot} fontFamily={FONT.display} fontWeight={800} fontSize={40}>stretch</text>
            <text x={CX + 30} y={CY - R - 120} fill={C.lcdm} fontFamily={FONT.display} fontWeight={800} fontSize={40}>squeeze</text>
          </g>
        )}
        {b.on(4) && !b.on(5) && (
          <g opacity={Math.min(b.p(4, 15), 1 - b.p(5, 12))}>
            <Axes box={box} xd={[0, 0.7]} yd={[0, 6]} xticks={[0, 0.2, 0.4, 0.6]} yticks={[0, 2, 4, 6]} xlabel="how close we are to a node (R / S)" ylabel="tidal pull" />
            <path d={pathOf(partial(linear, b.p(4, 40, 10)))} stroke={C.dim} strokeWidth={5} strokeDasharray="12 10" fill="none" />
            <path d={pathOf(partial(exact, b.p(4, 50, 25)))} stroke={C.node} strokeWidth={7} fill="none" />
            <LegendItem x={box.x + 40} y={box.y + 30} color={C.dim} dashed text="linear (Λ-like)" />
            <LegendItem x={box.x + 40} y={box.y + 80} color={C.node} text="real pull: 1/(S−R)²" />
          </g>
        )}
      </svg>
      <Show from={1} until={2} offset={20}>
        <At x={CX} y={170}>
          <Title size={64}>
            net pull = <span style={{color: C.gold}}>0</span>
          </Title>
        </At>
      </Show>
      <Show from={3} until={4}>
        <At x={360} y={CY}>
          <Card style={{display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10}}>
            <Label size={34} color={C.ink} style={{fontFamily: FONT.display, fontWeight: 800}}>
              SHEAR
            </Label>
            <Tex tex={String.raw`T\sim\frac{GM}{S^3}\begin{pmatrix}+4&&\\&-2&\\&&-2\end{pmatrix}`} size={44} />
            <Label size={30}>trace = 0 → no net inflation</Label>
          </Card>
        </At>
      </Show>
      <Show from={5}>
        <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', flexDirection: 'column', gap: 30}}>
          <svg width={420} height={300}>
            {[0, 1, 2, 3, 4, 5, 6].map((i) => (
              <ellipse key={i} cx={210} cy={150} rx={40 + i * 26} ry={(40 + i * 26) * 0.62} fill="none" stroke={C.gold} strokeWidth={5} opacity={0.9 - i * 0.1} strokeDasharray={i % 2 ? '30 14' : undefined} />
            ))}
          </svg>
          <Title size={80} color={C.gold}>
            a fingerprint
          </Title>
          <Label size={36}>lopsided expansion · hold that thought</Label>
        </AbsoluteFill>
      </Show>
    </AbsoluteFill>
  );
};
