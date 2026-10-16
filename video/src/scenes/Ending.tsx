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
/** Mollweide-ish sky ellipse with optional dipole tint, multipole lobes and flow arrows. */
const SKY = {cx: CX, cy: 470, rx: 560, ry: 280};
const AXIS = -0.45; // radians: the shared preferred direction on the map

export const Predictions: React.FC = () => {
  const b = useBeats();
  const f = b.frame;
  const dip = b.p(1, 30);
  const multi = b.p(3, 30);
  const flow = b.p(5, 30);
  const phantom = b.on(6) && !b.on(7);
  const final = b.p(7, 30);
  const ax = Math.cos(AXIS);
  const ay = Math.sin(AXIS);
  // phantom chart (beat 6), qualitative
  const box = {x: 520, y: 200, w: 880, h: 480};
  const sc = scales(box, [0, 1], [-1.35, -0.85]);
  const xs = new Array(60).fill(0).map((_, i) => i / 59);
  const wModel = xs.map((u) => [sc.X(u), sc.Y(-1 - 0.25 * u ** 2.2)] as [number, number]);
  const skyO = 1 - (phantom ? b.p(6, 12) : 0) + (b.on(7) ? b.p(7, 12) : 0);
  return (
    <AbsoluteFill>
      <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute', opacity: Math.min(1, skyO)}}>
        <defs>
          <clipPath id="skyclip">
            <ellipse cx={SKY.cx} cy={SKY.cy} rx={SKY.rx} ry={SKY.ry} />
          </clipPath>
          <linearGradient id="dipole" x1={0.5 - ax / 2} y1={0.5 - ay / 2} x2={0.5 + ax / 2} y2={0.5 + ay / 2}>
            <stop offset="0%" stopColor={C.lcdm} stopOpacity={0.55} />
            <stop offset="50%" stopColor="#1a2140" stopOpacity={0.2} />
            <stop offset="100%" stopColor={C.hot} stopOpacity={0.6} />
          </linearGradient>
        </defs>
        <ellipse cx={SKY.cx} cy={SKY.cy} rx={SKY.rx} ry={SKY.ry} fill="#0b1226" stroke={C.dim} strokeWidth={3} />
        <g clipPath="url(#skyclip)">
          {/* faint graticule */}
          {[-2, -1, 0, 1, 2].map((k) => (
            <ellipse key={k} cx={SKY.cx} cy={SKY.cy} rx={Math.abs(k) * SKY.rx * 0.33 || 1} ry={SKY.ry} fill="none" stroke={C.faint} strokeWidth={1.5} />
          ))}
          {[-0.66, -0.33, 0.33, 0.66].map((k) => (
            <line key={k} x1={SKY.cx - SKY.rx} x2={SKY.cx + SKY.rx} y1={SKY.cy + k * SKY.ry} y2={SKY.cy + k * SKY.ry} stroke={C.faint} strokeWidth={1.5} />
          ))}
          <rect x={SKY.cx - SKY.rx} y={SKY.cy - SKY.ry} width={SKY.rx * 2} height={SKY.ry * 2} fill="url(#dipole)" opacity={dip} />
          {/* quadrupole + octopole lobes aligned on the axis */}
          <g opacity={multi} transform={`rotate(${(AXIS * 180) / Math.PI} ${SKY.cx} ${SKY.cy})`}>
            {[0, 1].map((k) => (
              <ellipse key={`q${k}`} cx={SKY.cx + (k ? 1 : -1) * 230} cy={SKY.cy} rx={150} ry={70} fill="none" stroke={C.gold} strokeWidth={5} />
            ))}
            {[-1, 0, 1].map((k) => (
              <ellipse key={`o${k}`} cx={SKY.cx + k * 150} cy={SKY.cy} rx={50} ry={120} fill="none" stroke={C.node} strokeWidth={4} strokeDasharray="10 8" />
            ))}
            <line x1={SKY.cx - 600} x2={SKY.cx + 600} y1={SKY.cy} y2={SKY.cy} stroke={C.gold} strokeWidth={3} strokeDasharray="14 10" />
          </g>
          {/* dark flow arrows */}
          {flow > 0 &&
            [-2, -1, 0, 1, 2].flatMap((i) =>
              [-1, 0, 1].map((j) => {
                const x = SKY.cx - 350 + i * 120;
                const y = SKY.cy + j * 120 + i * 10;
                const L = 70 * flow;
                const tt = ((f * 2 + i * 20 + j * 37) % 60) / 60;
                return <Arrow key={`${i}${j}`} x1={x + ax * L * tt} y1={y + ay * L * tt} x2={x + ax * L * (tt + 0.8)} y2={y + ay * L * (tt + 0.8)} color={C.eds} width={4} head={14} opacity={flow} />;
              }),
            )}
        </g>
        {/* the shared direction */}
        {final > 0 && (
          <g opacity={final}>
            <circle cx={SKY.cx + ax * 470} cy={SKY.cy + ay * 235} r={30 + 8 * Math.sin(f * 0.2)} fill="none" stroke={C.gold} strokeWidth={6} />
            <NodeDot x={SKY.cx + ax * 470} y={SKY.cy + ay * 235} r={16} />
          </g>
        )}
      </svg>

      <Show from={0} until={1}>
        <At x={CX} y={830}>
          <Title size={54}>
            the fingerprint = <span style={{color: C.gold}}>anisotropy</span>
          </Title>
        </At>
      </Show>
      <Show from={1} until={3}>
        <At x={CX} y={830}>
          <Title size={46}>
            <span style={{color: C.gold}}>#1</span> Hubble dipole: <span style={{color: C.hot}}>faster</span> one way, <span style={{color: C.lcdm}}>slower</span> the other
          </Title>
        </At>
      </Show>
      <Show from={2} until={3}>
        <At x={CX} y={110}>
          <Card style={{display: 'flex', gap: 60, alignItems: 'center', padding: '16px 40px'}}>
            <div style={{textAlign: 'center'}}>
              <Title size={56} color={C.lcdm}>67</Title>
              <Label size={22}>whole-sky (CMB)</Label>
            </div>
            <Title size={44} color={C.dim}>vs</Title>
            <div style={{textAlign: 'center'}}>
              <Title size={56} color={C.hot}>73</Title>
              <Label size={22}>distance ladder</Label>
            </div>
            <Label size={26} color={C.ink}>km/s/Mpc · “Hubble tension”</Label>
          </Card>
        </At>
      </Show>
      <Show from={3} until={5}>
        <At x={CX} y={830}>
          <Title size={46}>
            <span style={{color: C.gold}}>#2</span> CMB “Axis of Evil”: biggest patterns share one axis
          </Title>
        </At>
      </Show>
      <Show from={5} until={6}>
        <At x={CX} y={830}>
          <Title size={46}>
            <span style={{color: C.gold}}>#3</span> Dark flow: galaxies drifting in bulk
          </Title>
        </At>
      </Show>
      {phantom && (
        <AbsoluteFill style={{opacity: b.p(6, 15)}}>
          <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute'}}>
            <Axes box={box} xd={[0, 1]} yd={[-1.35, -0.85]} xticks={[0, 1]} fmtX={(v) => (v ? 'later' : 'earlier')} yticks={[-1.3, -1.2, -1.1, -1, -0.9]} fmtY={(v) => v.toFixed(1)} xlabel="time →" ylabel="w (dark-energy “type”)" />
            <line x1={box.x} x2={box.x + box.w} y1={sc.Y(-1)} y2={sc.Y(-1)} stroke={C.lcdm} strokeWidth={6} />
            <path d={pathOf(partial(wModel, b.p(6, 50, 10)))} stroke={C.model} strokeWidth={7} fill="none" />
            <LegendItem x={box.x + 30} y={box.y + 30} color={C.lcdm} text="ΛCDM: w = −1 forever" />
            <LegendItem x={box.x + 30} y={box.y + 80} color={C.model} text="nodes: w drifts below −1 (shape illustrative)" />
          </svg>
          <At x={CX} y={830}>
            <Title size={46}>
              <span style={{color: C.gold}}>#4</span> Phantom acceleration: it keeps getting stronger
            </Title>
          </At>
        </AbsoluteFill>
      )}
      <Show from={7}>
        <At x={CX} y={830}>
          <Title size={50}>
            all pointing <span style={{color: C.gold}}>the same way</span>
          </Title>
        </At>
      </Show>
    </AbsoluteFill>
  );
};

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
            <Title size={40} color={C.ink}>
              {CREDITS.author}
            </Title>
            <Label size={28}>{CREDITS.orcid}</Label>
            <div style={{height: 26}} />
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
