import React from 'react';
import {AbsoluteFill, interpolate} from 'remotion';
import {useBeats} from '../beats';
import {Show} from '../components/Show';
import {Arrow, At, Label, NodeDot, SvgDefs, Title} from '../components/ui';
import {C, FONT, HEIGHT, WIDTH} from '../theme';

const CX = WIDTH / 2;
const CY = HEIGHT / 2 - 20;
const RING = 330;
const N = 50;
// five clumps: angle, then the "real swarm" version (distance factor, size factor)
const LUMPS = [
  {a: 0.25, d: 0.78, s: 1.7},
  {a: 1.45, d: 1.18, s: 0.7},
  {a: 2.6, d: 1.0, s: 1.0},
  {a: 3.75, d: 1.25, s: 0.55},
  {a: 5.05, d: 0.95, s: 1.25},
];

export const Tides: React.FC = () => {
  const b = useBeats();
  const f = b.frame;
  const clump = b.p(2, 35);
  const uneven = b.p(3, 45);
  const sceneA = 1 - b.p(4, 15); // shell/clumps part fades out at beat 4

  const lumpPos = LUMPS.map((L) => {
    const d = RING * (1 + (L.d - 1) * uneven);
    const s = 1 + (L.s - 1) * uneven;
    return {x: CX + Math.cos(L.a) * d, y: CY + Math.sin(L.a) * d, s, d, a: L.a};
  });
  // dots of the shell slide into the clumps
  const dots = new Array(N).fill(0).map((_, i) => {
    const a0 = (i / N) * Math.PI * 2;
    const k = Math.floor((i / N) * LUMPS.length);
    const t = lumpPos[k];
    const x0 = CX + Math.cos(a0) * RING;
    const y0 = CY + Math.sin(a0) * RING;
    return {x: x0 + (t.x - x0) * clump, y: y0 + (t.y - y0) * clump};
  });
  const shellPull = 140 * b.p(0, 25, 15) * (1 - b.p(1, 25, 25));
  // net pull of the uneven swarm on the centre: sum of m / d²
  let nx = 0;
  let ny = 0;
  lumpPos.forEach((L) => {
    const g = L.s / (L.d / RING) ** 2;
    nx += Math.cos(L.a) * g;
    ny += Math.sin(L.a) * g;
  });
  const nL = Math.hypot(nx, ny) || 1;
  const net = 60 * Math.min(4, nL) * uneven;

  // beat 4: outward pull grows with distance
  const field: {x: number; y: number; r: number; a: number}[] = [];
  for (let r = 50; r <= 290; r += 60) for (let k = 0; k < Math.round(r / 22); k++) {
    const a = (k / Math.round(r / 22)) * Math.PI * 2 + r * 0.01;
    field.push({x: CX + Math.cos(a) * r, y: CY + Math.sin(a) * r, r, a});
  }
  const fieldP = b.p(4, 40);
  const fieldO = 1 - b.p(5, 12);
  const finalP = b.p(5, 20);
  return (
    <AbsoluteFill>
      <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute'}}>
        <SvgDefs />
        {sceneA > 0 && (
          <g opacity={sceneA}>
            {clump < 0.98 &&
              dots.map((p, i) => <circle key={i} cx={p.x} cy={p.y} r={10} fill={C.node} opacity={0.85 * (1 - clump * 0.9)} />)}
            {clump > 0.02 && lumpPos.map((L, i) => <NodeDot key={i} x={L.x} y={L.y} r={26 * L.s} opacity={clump} />)}
            {!b.on(2) ? (
              <>
                <circle cx={CX} cy={CY} r={16} fill={C.bubble} />
                <text x={CX} y={CY + 52} fill={C.bubble} fontFamily={FONT.body} fontWeight={700} fontSize={30} textAnchor="middle">
                  you
                </text>
                {new Array(12).fill(0).map((_, i) => {
                  const a = (i / 12) * Math.PI * 2;
                  return <Arrow key={i} x1={CX + Math.cos(a) * 28} y1={CY + Math.sin(a) * 28} x2={CX + Math.cos(a) * (28 + shellPull)} y2={CY + Math.sin(a) * (28 + shellPull)} color={C.gold} width={5} head={16} />;
                })}
              </>
            ) : (
              <>
                <circle cx={CX} cy={CY} r={95} fill="url(#bubble)" stroke={C.bubble} strokeWidth={3} opacity={b.p(2, 15)} />
                {/* each clump tugs on the side of our universe that faces it */}
                {lumpPos.map((L, i) => {
                  const len = (34 + 30 * L.s) / (L.d / RING) ** 1.5;
                  return (
                    <Arrow
                      key={i}
                      x1={CX + Math.cos(L.a) * 100}
                      y1={CY + Math.sin(L.a) * 100}
                      x2={CX + Math.cos(L.a) * (100 + len)}
                      y2={CY + Math.sin(L.a) * (100 + len)}
                      color={C.node}
                      width={6}
                      head={20}
                      p={b.p(2, 20, 25)}
                    />
                  );
                })}
                {uneven > 0 && <Arrow x1={CX} y1={CY} x2={CX + (nx / nL) * net} y2={CY + (ny / nL) * net} color={C.hot} width={9} head={26} />}
              </>
            )}
          </g>
        )}
        {b.on(4) && fieldO > 0 && (
          <g opacity={Math.min(b.p(4, 15), fieldO)}>
            <circle cx={CX} cy={CY} r={320} fill="url(#bubble)" stroke={C.bubble} strokeWidth={3} />
            {field.map((p, i) => {
              const L = (p.r / 290) * 70 * fieldP;
              return <Arrow key={i} x1={p.x} y1={p.y} x2={p.x + Math.cos(p.a) * L} y2={p.y + Math.sin(p.a) * L} color={C.gold} width={4} head={12} />;
            })}
          </g>
        )}
        {finalP > 0 && (
          <g opacity={finalP}>
            {/* pull from outside */}
            <circle cx={520} cy={CY} r={130} fill="url(#bubble)" stroke={C.bubble} strokeWidth={3} />
            {[0, 1, 2, 3, 4, 5].map((k) => {
              const a = (k / 6) * Math.PI * 2 + 0.3;
              return (
                <g key={k}>
                  <NodeDot x={520 + Math.cos(a) * 300} y={CY + Math.sin(a) * 260} r={20} />
                  <Arrow x1={520 + Math.cos(a) * 140} y1={CY + Math.sin(a) * 140} x2={520 + Math.cos(a) * 225} y2={CY + Math.sin(a) * 200} color={C.node} width={6} head={18} />
                </g>
              );
            })}
            {/* push from inside */}
            <circle cx={1400} cy={CY} r={130} fill="url(#bubble)" stroke={C.bubble} strokeWidth={3} />
            {[0, 1, 2, 3, 4, 5, 6, 7].map((k) => {
              const a = (k / 8) * Math.PI * 2;
              return <Arrow key={k} x1={1400 + Math.cos(a) * 30} y1={CY + Math.sin(a) * 30} x2={1400 + Math.cos(a) * 210} y2={CY + Math.sin(a) * 210} color={C.lcdm} width={6} head={18} />;
            })}
            <text x={CX - 20} y={CY + 30} fill={C.gold} fontFamily={FONT.display} fontWeight={900} fontSize={100} textAnchor="middle">
              =
            </text>
          </g>
        )}
      </svg>

      <Show from={0} until={2} offset={20}>
        <At x={CX} y={110}>
          <Title size={54}>
            Smooth shell → net pull = <span style={{color: C.gold}}>{interpolate(shellPull, [0, 140], [0, 1]) < 0.05 ? '0' : '…'}</span>
          </Title>
        </At>
      </Show>
      <Show from={2} until={3}>
        <At x={CX} y={110}>
          <Title size={58} color={C.node}>
            Clumps don’t cancel: pulled apart
          </Title>
        </At>
      </Show>
      <Show from={3} until={4}>
        <At x={CX} y={110}>
          <Title size={54}>
            different sizes, different distances → <span style={{color: C.hot}}>lopsided</span>
          </Title>
        </At>
      </Show>
      <Show from={4} until={5}>
        <At x={CX} y={110}>
          <Title size={54}>
            outward pull, <span style={{color: C.gold}}>stronger farther out</span>
          </Title>
        </At>
        <At x={CX} y={890}>
          <Label size={32} color={C.ink} style={{fontWeight: 700}}>
            …just like dark energy
          </Label>
        </At>
      </Show>
      <Show from={5}>
        <At x={520} y={120}>
          <Title size={50} color={C.node}>
            pull from outside
          </Title>
        </At>
        <At x={1400} y={120}>
          <Title size={50} color={C.lcdm}>
            push from inside
          </Title>
        </At>
        <At x={CX} y={880}>
          <Label size={34} color={C.eds} style={{fontWeight: 800}}>
            no exotic energy required
          </Label>
        </At>
      </Show>
    </AbsoluteFill>
  );
};
