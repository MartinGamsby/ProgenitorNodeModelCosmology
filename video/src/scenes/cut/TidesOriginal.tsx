import React from 'react';
import {AbsoluteFill} from 'remotion';
import {interpolate} from '../../interp';
import {range, useBeats} from '../../beats';
import {Show} from '../../components/Show';
import {Arrow, At, Card, Label, NodeDot, SvgDefs, Tex, Title} from '../../components/ui';
import {C, FONT, HEIGHT, WIDTH} from '../../theme';

const CX = WIDTH / 2;
const CY = HEIGHT / 2 - 30;

export const TidesOriginal: React.FC = () => {
  const b = useBeats();
  const f = b.frame;
  // ---- beats 0-2: smooth shell -> lumps
  const N = 48;
  const ringR = 330;
  const clump = b.p(2, 40);
  const lumps = [0.3, 1.5, 2.4, 3.9, 5.1];
  const ringO = 1 - b.p(3, 15);
  const shell = new Array(N).fill(0).map((_, i) => {
    const a0 = (i / N) * Math.PI * 2;
    const target = lumps[Math.floor((i / N) * lumps.length)] + (((i % (N / lumps.length)) - 4.5) / 40);
    const a = a0 + (target - a0) * clump;
    return {x: CX + Math.cos(a) * ringR, y: CY + Math.sin(a) * ringR};
  });
  const pullLen = 150 * b.p(0, 25, 10) * (1 - b.p(1, 25, 25));

  // ---- beats 3-4: one node, three test points, stretch
  const nodeX = CX + 560;
  const R = 150;
  const stretch = b.p(4, 40);
  const pts = [
    {x: CX + R, label: 'near', g: 1 / (nodeX - CX - R) ** 2},
    {x: CX, label: 'middle', g: 1 / (nodeX - CX) ** 2},
    {x: CX - R, label: 'far', g: 1 / (nodeX - CX + R) ** 2},
  ];
  const gMid = pts[1].g;
  const k = 230 / pts[0].g;
  const rel = b.p(4, 25); // subtract the middle's pull -> tidal arrows
  return (
    <AbsoluteFill>
      <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute'}}>
        <SvgDefs />
        {!b.on(3, 15) && (
          <g opacity={ringO}>
            {shell.map((p, i) => (
              <circle key={i} cx={p.x} cy={p.y} r={10 + 6 * clump} fill={C.node} opacity={0.85} />
            ))}
            <circle cx={CX} cy={CY} r={16} fill={C.bubble} />
            <text x={CX} y={CY + 50} fill={C.bubble} fontFamily={FONT.body} fontWeight={700} fontSize={30} textAnchor="middle">
              you
            </text>
            {!b.on(2) &&
              new Array(12).fill(0).map((_, i) => {
                const a = (i / 12) * Math.PI * 2;
                return (
                  <Arrow key={i} x1={CX + Math.cos(a) * 30} y1={CY + Math.sin(a) * 30} x2={CX + Math.cos(a) * (30 + pullLen)} y2={CY + Math.sin(a) * (30 + pullLen)} color={C.gold} width={5} head={16} />
                );
              })}
            {b.on(2, 30) && (
              <Arrow x1={CX + 30} y1={CY - 10} x2={CX + 30 + 170} y2={CY - 70} color={C.hot} width={8} head={24} p={b.p(2, 20, 30)} />
            )}
          </g>
        )}
        {b.on(3) && (
          <g opacity={b.p(3, 15)}>
            <NodeDot x={nodeX} y={CY} r={46} />
            <text x={nodeX} y={CY + 110} fill={C.node} fontFamily={FONT.body} fontWeight={700} fontSize={30} textAnchor="middle">
              node
            </text>
            <ellipse
              cx={CX}
              cy={CY}
              rx={R * (1 + 0.35 * stretch)}
              ry={R * (1 - 0.18 * stretch)}
              fill="url(#bubble)"
              stroke={C.bubble}
              strokeWidth={4}
            />
            {pts.map((p, i) => {
              const L = k * (p.g - rel * gMid);
              const px = p.x + (i === 0 ? 1 : i === 2 ? -1 : 0) * R * 0.35 * stretch;
              const y = CY - 60 * (1 - rel);
              return (
                <g key={i}>
                  <circle cx={p.x + (i === 0 ? 1 : i === 2 ? -1 : 0) * R * 0.35 * stretch} cy={CY} r={9} fill="#fff" />
                  <Arrow
                    x1={px}
                    y1={y}
                    x2={px + L}
                    y2={y}
                    color={i === 0 ? C.hot : i === 1 ? C.gold : C.lcdm}
                    width={7}
                    head={22}
                    p={b.p(3, 20, 15 + i * 8)}
                  />
                  <text x={p.x} y={CY - 90} fill={C.dim} fontFamily={FONT.body} fontWeight={700} fontSize={26} textAnchor="middle" opacity={1 - rel}>
                    {p.label}
                  </text>
                </g>
              );
            })}
            <text x={CX} y={CY + R + 70} fill={C.ink} fontFamily={FONT.display} fontWeight={800} fontSize={38} textAnchor="middle" opacity={rel}>
              stretched: a tide
            </text>
          </g>
        )}
      </svg>

      <Show from={0} until={2} offset={20}>
        <At x={CX} y={130}>
          <Title size={54}>
            Smooth shell → net pull = <span style={{color: C.gold}}>{interpolate(pullLen, [0, 150], [0, 1]) < 0.05 ? '0' : '…'}</span>
          </Title>
        </At>
      </Show>
      <Show from={2} until={3}>
        <At x={CX} y={130}>
          <Title size={64} color={C.hot}>
            Lumps don’t cancel
          </Title>
        </At>
      </Show>

      {/* beats 5-7: equations */}
      <Show from={5} until={8}>
        <AbsoluteFill style={{background: `rgba(4,6,13,${0.82 * b.p(5, 15)})`}} />
      </Show>
      <Show from={5}>
        <At x={CX} y={300}>
          <Card accent={C.node} style={{display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6}}>
            <Label size={30} color={C.node} style={{fontFamily: FONT.display, fontWeight: 800}}>
              TIDAL PULL FROM A NODE
            </Label>
            <Tex tex={String.raw`a_{\rm tide}\approx\frac{GM}{(S-R)^2}-\frac{GM}{S^2}\approx\frac{2GM}{S^3}\,\color{#ffd84a}{R}`} size={60} />
          </Card>
        </At>
      </Show>
      <Show from={6}>
        <At x={CX} y={560}>
          <Card accent={C.lcdm} style={{display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6}}>
            <Label size={30} color={C.lcdm} style={{fontFamily: FONT.display, fontWeight: 800}}>
              DARK ENERGY’S PUSH
            </Label>
            <Tex tex={String.raw`a_{\Lambda}=H_0^2\,\Omega_\Lambda\,\color{#ffd84a}{R}`} size={60} />
          </Card>
        </At>
      </Show>
      <Show from={7}>
        <At x={CX} y={770}>
          <Title size={50} style={{textAlign: 'center'}}>
            both <span style={{color: C.gold}}>∝ R</span> · same shape
          </Title>
        </At>
        <At x={CX} y={830}>
          <Label size={30} style={{opacity: range(f, b.at(7, 30), b.at(7, 50))}}>
            (along the axis toward a node)
          </Label>
        </At>
      </Show>
    </AbsoluteFill>
  );
};
