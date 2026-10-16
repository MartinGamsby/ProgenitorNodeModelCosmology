import React from 'react';
import {AbsoluteFill} from 'remotion';
import {useBeats} from '../beats';
import {Show} from '../components/Show';
import {Arrow, At, Label, NodeDot, SvgDefs, Title} from '../components/ui';
import {C, HEIGHT, WIDTH} from '../theme';

const CX = WIDTH / 2;
const CY = HEIGHT / 2 - 10;
// the "real swarm" version of each lattice site: offset (px) and mass factor
const JITTER = [
  [40, -30, 1.6], [-20, 50, 0.7], [-60, 20, 1.1], [30, 40, 0.6],
  [-30, -40, 1.3], [50, 10, 0.8], [-10, -60, 1.0], [20, 30, 1.9],
];

export const WhatIfCenter: React.FC = () => {
  const b = useBeats();
  const sp = 300;
  const real = b.p(2, 40);
  const nodes: {x: number; y: number; m: number}[] = [];
  let k = 0;
  for (let i = -1; i <= 1; i++)
    for (let j = -1; j <= 1; j++) {
      if (!i && !j) continue;
      const [dx, dy, m] = JITTER[k++];
      nodes.push({x: CX + i * sp * 1.25 + dx * real, y: CY + j * sp * 0.9 + dy * real, m: 1 + (m - 1) * real});
    }
  // net pull on the centre: Σ m r̂ / d²
  let nx = 0;
  let ny = 0;
  nodes.forEach((n) => {
    const dx = n.x - CX;
    const dy = n.y - CY;
    const d = Math.hypot(dx, dy) / sp;
    nx += (n.m * dx) / (d ** 3 * sp);
    ny += (n.m * dy) / (d ** 3 * sp);
  });
  const nL = Math.hypot(nx, ny) || 1;
  const netLen = Math.min(220, 160 * nL) * real;
  const lop = b.p(3, 40);
  const ax = nx / nL;
  const ay = ny / nL;
  return (
    <AbsoluteFill>
      <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute'}}>
        <SvgDefs />
        {nodes.map((n, i) => (
          <g key={i}>
            <NodeDot x={n.x} y={n.y} r={30 * Math.sqrt(n.m)} opacity={b.p(0, 20, i * 3)} />
            <Arrow
              x1={CX}
              y1={CY}
              x2={CX + (n.x - CX) * 0.42}
              y2={CY + (n.y - CY) * 0.42}
              color={C.gold}
              width={6}
              head={20}
              p={b.p(0, 25, 30)}
              opacity={1 - b.p(1, 30, 20)}
            />
          </g>
        ))}
        {/* our universe: round, then expanding a bit faster toward the net pull */}
        <ellipse
          cx={CX + ax * 18 * lop}
          cy={CY + ay * 18 * lop}
          rx={70 * (1 + 0.28 * lop * Math.abs(ax)) + 6 * lop}
          ry={70 * (1 + 0.28 * lop * Math.abs(ay)) + 6 * lop}
          fill="url(#bubble)"
          stroke={C.bubble}
          strokeWidth={3}
        />
        {real > 0 && <Arrow x1={CX} y1={CY} x2={CX + ax * netLen} y2={CY + ay * netLen} color={C.hot} width={9} head={26} />}
      </svg>
      <Show from={1} until={2} offset={20}>
        <At x={CX} y={110}>
          <Title size={64}>
            net pull = <span style={{color: C.gold}}>0</span>
          </Title>
        </At>
      </Show>
      <Show from={2} until={3}>
        <At x={CX} y={110}>
          <Title size={58}>
            real swarm: <span style={{color: C.hot}}>one side pulls harder</span>
          </Title>
        </At>
      </Show>
      <Show from={3}>
        <At x={CX} y={100}>
          <Title size={64} color={C.gold}>
            a fingerprint
          </Title>
        </At>
        <At x={CX} y={170}>
          <Label size={32} color={C.ink} style={{fontWeight: 700}}>
            expansion slightly faster in one direction
          </Label>
        </At>
      </Show>
    </AbsoluteFill>
  );
};
