import React from 'react';
import {AbsoluteFill, interpolate} from 'remotion';
import {useBeats} from '../beats';
import {Axes, LegendItem, scales} from '../components/Chart';
import {Show} from '../components/Show';
import {Space3D} from '../components/Space3D';
import {Arrow, At, Card, Label, NodeDot, pathOf, partial, SvgDefs, Tex, Title} from '../components/ui';
import medium from '../../public/data/medium.json';
import {CREDITS} from '../script';
import {C, FONT, HEIGHT, WIDTH} from '../theme';

const CX = WIDTH / 2;

// ---------------------------------------------------------------------------------------------
export const Catch: React.FC = () => {
  const b = useBeats();
  const y = 470;
  return (
    <AbsoluteFill>
      <Show from={0} until={1}>
        <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
          <Title size={130} color={C.hot}>
            The catch.
          </Title>
        </AbsoluteFill>
      </Show>
      <Show from={1}>
        <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute'}}>
          <SvgDefs />
          {/* heavy & far */}
          <circle cx={300} cy={y} r={70} fill="url(#bubble)" stroke={C.bubble} strokeWidth={3} />
          <NodeDot x={820} y={y} r={56} />
          <Arrow x1={390} y1={y} x2={730} y2={y} color={C.node} width={5} p={b.p(1, 20)} />
          <text x={560} y={y + 140} fill={C.ink} fontFamily={FONT.display} fontWeight={800} fontSize={38} textAnchor="middle">heavy, far</text>
          {/* light & near */}
          <circle cx={1150} cy={y} r={70} fill="url(#bubble)" stroke={C.bubble} strokeWidth={3} />
          <NodeDot x={1480} y={y} r={26} opacity={b.p(1, 20, 20)} />
          <Arrow x1={1240} y1={y} x2={1440} y2={y} color={C.node} width={5} p={b.p(1, 20, 20)} />
          <text x={1330} y={y + 140} fill={C.ink} fontFamily={FONT.display} fontWeight={800} fontSize={38} textAnchor="middle" opacity={b.p(1, 20, 20)}>light, near</text>
          <text x={CX - 25} y={y + 30} fill={C.gold} fontFamily={FONT.display} fontWeight={900} fontSize={110} textAnchor="middle" opacity={b.p(1, 20, 45)}>=</text>
        </svg>
      </Show>
      <Show from={2}>
        <At x={CX} y={200}>
          <Card accent={C.gold} style={{display: 'flex', alignItems: 'center', gap: 30}}>
            <Label size={34} color={C.ink} style={{fontWeight: 700}}>
              supernovae only see
            </Label>
            <Tex tex={String.raw`\frac{M}{S^3}`} size={80} color={C.gold} />
          </Card>
        </At>
      </Show>
      <Show from={3}>
        <At x={CX} y={800}>
          <Title size={52}>
            → we need a <span style={{color: C.gold}}>different test</span>
          </Title>
        </At>
      </Show>
    </AbsoluteFill>
  );
};

// ---------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------
export const WhatIfInvisible: React.FC = () => {
  const b = useBeats();
  const f = b.frame;
  const cy = 440;
  return (
    <AbsoluteFill>
      <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute'}}>
        <defs>
          <radialGradient id="lens">
            <stop offset="55%" stopColor="#000" stopOpacity={1} />
            <stop offset="62%" stopColor={C.node} stopOpacity={0.5} />
            <stop offset="70%" stopColor={C.nodeGlow} stopOpacity={0.12} />
            <stop offset="100%" stopColor={C.nodeGlow} stopOpacity={0} />
          </radialGradient>
        </defs>
        <circle cx={CX} cy={cy} r={260 + 6 * Math.sin(f * 0.05)} fill="url(#lens)" opacity={b.p(0, 30)} />
        {/* the disk it no longer has */}
        <g opacity={interpolate(f, [b.at(1), b.at(1, 20), b.at(2)], [0, 0.8, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})}>
          <ellipse cx={CX} cy={cy} rx={330} ry={60} fill="none" stroke={C.gold} strokeWidth={14} opacity={0.6} />
        </g>
      </svg>
      <Show from={1} until={2} offset={30}>
        <At x={CX} y={cy}>
          <Title size={44} color={C.dim} style={{textDecoration: 'line-through'}}>
            glowing disk
          </Title>
        </At>
      </Show>
      <Show from={2} until={3}>
        <At x={CX} y={790}>
          <Card accent={C.lcdm} style={{display: 'flex', alignItems: 'center', gap: 40}}>
            <Label size={30} color={C.ink} style={{fontWeight: 700}}>
              Hawking temperature
            </Label>
            <Tex tex={String.raw`T_H=\frac{\hbar c^3}{8\pi G M k_B}\approx 2.5\times10^{-33}\,\mathrm{K}`} size={46} />
            <Label size={26}>(for M ≈ 5×10⁵⁵ kg)</Label>
          </Card>
        </At>
      </Show>
      <Show from={3}>
        <At x={CX} y={cy}>
          <Title size={120} style={{textShadow: `0 0 40px ${C.nodeGlow}`}}>
            DARK GIANTS
          </Title>
        </At>
        <At x={CX} y={830}>
          <Label size={36} color={C.node} style={{fontWeight: 800}}>
            seen only by their pull
          </Label>
        </At>
      </Show>
    </AbsoluteFill>
  );
};

// ---------------------------------------------------------------------------------------------
export const WhatIfCycle: React.FC = () => {
  const b = useBeats();
  const f = b.frame;
  const R = 330;
  const cy = 500;
  const steps = [
    {t: 'a node bangs', c: C.gold, beat: 1},
    {t: 'matter fills the gap', c: C.bubble, beat: 1},
    {t: 'neighbours feed', c: C.node, beat: 2},
    {t: 'one grows too big…', c: C.hot, beat: 3},
  ];
  const spin = interpolate(f, [b.at(1), b.dur], [0, Math.PI * 2], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  return (
    <AbsoluteFill>
      <At x={1640} y={150} style={{transform: 'translate(-50%,-50%) rotate(6deg)'}}>
        <div style={{border: `5px solid ${C.hot}`, color: C.hot, fontFamily: FONT.display, fontWeight: 900, fontSize: 44, padding: '6px 22px', borderRadius: 10, opacity: b.p(0, 15, 10)}}>
          SPECULATIVE
        </div>
      </At>
      <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute'}}>
        <circle cx={CX} cy={cy} r={R} fill="none" stroke={C.faint} strokeWidth={4} strokeDasharray="4 14" />
        <g transform={`rotate(${(spin * 180) / Math.PI} ${CX} ${cy})`} opacity={b.p(1, 20)}>
          <circle cx={CX + R} cy={cy} r={12} fill={C.gold} />
        </g>
        {steps.map((s, i) => {
          const a = -Math.PI / 2 + (i / steps.length) * Math.PI * 2;
          const x = CX + Math.cos(a) * R;
          const y = cy + Math.sin(a) * R;
          const p = b.p(s.beat, 20, i === 1 ? 40 : 0);
          return (
            <g key={i} opacity={0.15 + 0.85 * p}>
              <circle cx={x} cy={y} r={46} fill={C.bg2} stroke={s.c} strokeWidth={5} />
              <text x={x} y={y + 14} fill={s.c} fontFamily={FONT.display} fontWeight={900} fontSize={40} textAnchor="middle">
                {i + 1}
              </text>
              <text x={x + (Math.cos(a) < -0.3 ? -70 : 70)} y={y + 12} fill={C.ink} fontFamily={FONT.body} fontWeight={700} fontSize={34} textAnchor={Math.cos(a) < -0.3 ? 'end' : 'start'}>
                {s.t}
              </text>
            </g>
          );
        })}
      </svg>
      <Show from={4}>
        <At x={CX} y={cy}>
          <div style={{textAlign: 'center'}}>
            <Title size={40} color={C.gold}>
              the great
            </Title>
            <Title size={46} color={C.gold}>
              cosmic
            </Title>
            <Title size={46} color={C.gold}>
              metabolism
            </Title>
          </div>
        </At>
      </Show>
    </AbsoluteFill>
  );
};

// ---------------------------------------------------------------------------------------------
export const Limits: React.FC = () => {
  const b = useBeats();
  const ins = ['Newtonian gravity, flat space', 'late universe: 2.9 → 13.8 Gyr', 'real supernova fit'];
  const outs = ['general relativity', 'the early universe', 'CMB spectrum', 'baryon acoustic oscillations'];
  return (
    <AbsoluteFill style={{alignItems: 'center', paddingTop: 130}}>
      <div style={{transform: `scale(${0.8 + 0.2 * b.s(0)})`, opacity: b.p(0, 12)}}>
        <Title size={90}>
          a <span style={{color: C.gold}}>toy model</span>
        </Title>
      </div>
      <div style={{display: 'flex', gap: 60, marginTop: 50}}>
        <Card accent={C.eds} style={{width: 640}}>
          <Title size={40} color={C.eds} style={{marginBottom: 16}}>
            IN
          </Title>
          {ins.map((t, i) => (
            <Label key={i} size={34} color={C.ink} style={{margin: '10px 0', opacity: b.p(1, 15, i * 8)}}>
              ✓ {t}
            </Label>
          ))}
        </Card>
        <Card accent={C.hot} style={{width: 640}}>
          <Title size={40} color={C.hot} style={{marginBottom: 16}}>
            NOT (YET)
          </Title>
          {outs.map((t, i) => (
            <Label key={i} size={34} color={C.ink} style={{margin: '10px 0', opacity: b.p(1, 15, 30 + i * 18)}}>
              ✗ {t}
            </Label>
          ))}
        </Card>
      </div>
      <Show from={2}>
        <At x={CX} y={830}>
          <Title size={44}>
            claim: <span style={{color: C.gold}}>ordinary gravity can fake dark energy</span>
          </Title>
        </At>
      </Show>
    </AbsoluteFill>
  );
};

// ---------------------------------------------------------------------------------------------
export const Outro: React.FC = () => {
  const b = useBeats();
  const f = b.frame;
  const nodes = medium.pos as [number, number, number][];
  const mass = medium.mass as number[];
  const credits = b.p(3, 25);

  return (
    <AbsoluteFill>
      <Space3D
        width={WIDTH}
        height={HEIGHT}
        camera={{yaw: f * 0.003 + 1, pitch: 0.25, distance: (3 * 540) / 2.4, scale: 2.4}}
        layers={[
          {kind: 'glows', pts: nodes, radius: (i) => 6 + 6 * Math.cbrt(mass[i]), color: C.nodeGlow, core: C.node, alpha: 0.35 + 0.4 * b.p(2, 30)},
          {kind: 'sphere', center: [0, 0, 0], radius: 14, color: C.bubble, alpha: 1},
        ]}
        style={{position: 'absolute'}}
      />
      <AbsoluteFill style={{opacity: 1 - credits}}>
        <Show from={0} until={1}>
          <At x={CX} y={160}>
            <Title size={48}>dipole · axis · flow → aligned?</Title>
          </At>
        </Show>
        <Show from={1} until={2}>
          <At x={CX} y={160}>
            <Title size={52}>dark energy ≠ a property of space?</Title>
          </At>
        </Show>
        <Show from={2}>
          <At x={CX} y={190}>
            <Title size={96} style={{textShadow: `0 0 50px ${C.nodeGlow}`}}>
              just <span style={{color: C.node}}>the neighbours</span>.
            </Title>
          </At>
        </Show>
      </AbsoluteFill>
      {credits > 0 && (
        <AbsoluteFill style={{opacity: credits, background: 'rgba(4,6,13,0.78)', justifyContent: 'center', alignItems: 'center'}}>
          <Card accent={C.gold} style={{maxWidth: 1400, textAlign: 'center', padding: '50px 70px'}}>
            <Label size={26} color={C.gold} style={{fontFamily: FONT.display, fontWeight: 800, letterSpacing: 3}}>
              BASED ON THE PAPER
            </Label>
            <Title size={44} style={{margin: '18px 0 28px'}}>
              {CREDITS.paper}
            </Title>
            <Label size={30} color={C.ink} style={{fontFamily: FONT.mono}}>
              {CREDITS.code}
            </Label>
            <Label size={30} color={C.ink} style={{fontFamily: FONT.mono}}>
              {CREDITS.doi}
            </Label>
            <div style={{height: 22}} />
            <Label size={24}>Real data: Pantheon+SH0ES (Brout et al. 2022) · simulations from the repository above</Label>
          </Card>
        </AbsoluteFill>
      )}
    </AbsoluteFill>
  );
};
