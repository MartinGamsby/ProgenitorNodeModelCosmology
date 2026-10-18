import React from 'react';
import {AbsoluteFill, interpolate} from 'remotion';
import {useBeats} from '../beats';
import {Show} from '../components/Show';
import {Space3D} from '../components/Space3D';
import {Arrow, At, Card, Label, NodeDot, SvgDefs, Tex, Title} from '../components/ui';
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
  // the disk flashes in to show what's missing, then fades away
  const diskO = interpolate(f, [b.at(2), b.at(2, 20), b.at(2, 70), b.at(2, 100)], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
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
        {/* the disk it no longer has: far half behind the black hole, near half in front */}
        <path d={`M${CX - 330},${cy} A330,60 0 0 1 ${CX + 330},${cy}`} fill="none" stroke={C.gold} strokeWidth={14} opacity={0.6 * diskO} />
        <circle cx={CX} cy={cy} r={260 + 6 * Math.sin(f * 0.05)} fill="url(#lens)" opacity={b.p(0, 30)} />
        <path d={`M${CX - 330},${cy} A330,60 0 0 0 ${CX + 330},${cy}`} fill="none" stroke={C.gold} strokeWidth={14} opacity={0.6 * diskO} />
      </svg>
      {/* beat 1: distance isn't the reason; their light would have had time to reach us */}
      <Show from={1} until={2}>
        <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute'}}>
          <SvgDefs />
          <circle cx={1700} cy={cy} r={70} fill="url(#bubble)" stroke={C.bubble} strokeWidth={3} />
          {[-1, 0, 1].map((k) => {
            const t = ((f - b.at(1)) * 0.012 + k * 0.33) % 1;
            const x = CX + 280 + t * (1700 - 90 - CX - 280);
            return <Arrow key={k} x1={x - 60} y1={cy + k * 40} x2={x} y2={cy + k * 40} color={C.gold} width={5} head={16} />;
          })}
        </svg>
        <At x={1700} y={cy + 110}>
          <Label size={28} color={C.bubble} style={{fontWeight: 700}}>
            us
          </Label>
        </At>
        <At x={CX} y={150}>
          <Title size={46}>
            older than the Big Bang → <span style={{color: C.gold}}>their light had time to reach us</span>
          </Title>
        </At>
      </Show>
      <Show from={2} until={3} offset={30}>
        <At x={CX} y={cy + 330}>
          <Title size={40} color={C.dim}>
            ✗ no glowing disk
          </Title>
        </At>
        <At x={CX} y={150}>
          <Title size={50}>
            nothing left to eat → <span style={{color: C.node}}>no light left to give</span>
          </Title>
        </At>
      </Show>
      <Show from={3} until={4}>
        <At x={CX} y={790}>
          <Card accent={C.lcdm} style={{display: 'flex', alignItems: 'center', gap: 40}}>
            <Label size={30} color={C.ink} style={{fontWeight: 700}}>
              Hawking temperature
            </Label>
            <Tex tex={String.raw`T_H=\frac{\hbar c^3}{8\pi G M k_B}\approx 2.5\times10^{-33}\,\mathrm{K}`} size={46} />
            <Label size={26}>(for M ≈ 5×10⁵⁵ kg) · ≈ absolute zero</Label>
          </Card>
        </At>
      </Show>
      <Show from={4}>
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
  const cy = 530;
  const steps = [
    {t: 'an attractor bangs', c: C.gold, beat: 1},
    {t: 'matter fills the gap', c: C.bubble, beat: 1},
    {t: 'neighbours feed', c: C.node, beat: 2},
    {t: 'one grows too big…', c: C.hot, beat: 3},
  ];
  // ---- the mini universe in the middle: A bangs, B and C feed on it, B grows and bangs ----
  const A = {x: CX - 120, y: cy - 10};
  const B = {x: CX + 135, y: cy - 75};
  const Cn = {x: CX + 95, y: cy + 120};
  const bang1 = interpolate(f, [b.at(1), b.at(1, 50)], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: (t) => 1 - (1 - t) ** 3});
  const feed = interpolate(f, [b.at(2), b.at(3)], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const swell = interpolate(f, [b.at(3), b.at(3, 45)], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const bang2T = b.at(3, 60);
  const dotA = (interpolate(f, [b.at(1), b.at(2), b.at(3), bang2T, bang2T + 20], [-90, 0, 90, 180, 270], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}) * Math.PI) / 180;
  const bang2 = interpolate(f, [bang2T, bang2T + 50], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: (t) => 1 - (1 - t) ** 3});
  const flash1 = interpolate(f, [b.at(1), b.at(1, 4), b.at(1, 24)], [0, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const flash2 = interpolate(f, [bang2T, bang2T + 4, bang2T + 24], [0, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const wobble = swell > 0 && bang2 === 0 ? Math.sin(f * 0.9) * 4 * swell : 0;
  const parts = new Array(90).fill(0).map((_, i) => ({a: (i / 90) * Math.PI * 2 + (i % 7) * 0.13, r: 0.45 + ((i * 37) % 55) / 100, toB: i % 3 !== 0}));
  const rB = 22 * (1 + 0.6 * feed * 0.67 + 0.35 * swell);
  const rC = 22 * (1 + 0.6 * feed * 0.33);
  return (
    <AbsoluteFill>
      <At x={1640} y={150} style={{transform: 'translate(-50%,-50%) rotate(6deg)'}}>
        <div style={{border: `5px solid ${C.hot}`, color: C.hot, fontFamily: FONT.display, fontWeight: 900, fontSize: 44, padding: '6px 22px', borderRadius: 10, opacity: b.p(0, 15, 10)}}>
          SPECULATIVE
        </div>
      </At>
      <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute'}}>
        <SvgDefs />
        <circle cx={CX} cy={cy} r={R} fill="none" stroke={C.faint} strokeWidth={4} strokeDasharray="4 14" />
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
        {/* the marker follows the animation: 1→2 bang & fill, 2→3 feeding, 3→4 swelling,
            back to 1 on the second bang, then it rests */}
        <circle cx={CX + Math.cos(dotA) * R} cy={cy + Math.sin(dotA) * R} r={12} fill={C.gold} opacity={b.p(1, 20)} />
        {/* attractor A: bangs at beat 1 */}
        {bang1 < 0.05 && <NodeDot x={A.x} y={A.y} r={28} />}
        {bang1 < 0.05 && <circle cx={A.x} cy={A.y} r={30} fill="none" stroke={C.gold} strokeWidth={4} opacity={b.p(0, 20, 20)} />}
        <circle cx={A.x} cy={A.y} r={60 * flash1 + 10} fill="#fff" opacity={flash1} filter="url(#soft)" />
        {/* its matter: bursts out, then drifts into B or C */}
        {bang1 > 0 &&
          parts.map((p, i) => {
            const ox = A.x + Math.cos(p.a) * p.r * 170 * bang1;
            const oy = A.y + Math.sin(p.a) * p.r * 150 * bang1;
            const T = p.toB ? B : Cn;
            const u = Math.max(0, Math.min(1, feed * 1.4 - (i % 10) * 0.04));
            const x = ox + (T.x - ox) * u;
            const y = oy + (T.y - oy) * u;
            return <circle key={i} cx={x} cy={y} r={3} fill={i % 2 ? '#cfe4ff' : '#ffd9a8'} opacity={u > 0.97 ? 0 : 0.9} />;
          })}
        {/* the neighbours */}
        <NodeDot x={Cn.x} y={Cn.y} r={rC} />
        {bang2 < 0.05 && <NodeDot x={B.x + wobble} y={B.y} r={rB} color={swell > 0 ? C.hot : C.node} />}
        <circle cx={B.x} cy={B.y} r={70 * flash2 + 10} fill="#fff" opacity={flash2} filter="url(#soft)" />
        {bang2 > 0 &&
          parts.map((p, i) => (
            <circle key={`b${i}`} cx={B.x + Math.cos(p.a + 0.2) * p.r * 160 * bang2} cy={B.y + Math.sin(p.a + 0.2) * p.r * 140 * bang2} r={3} fill={i % 2 ? '#cfe4ff' : '#ffd9a8'} opacity={0.9} />
          ))}
      </svg>
      <Show from={4}>
        <At x={CX} y={110}>
          <Title size={52} color={C.gold}>
            the great cosmic metabolism
          </Title>
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
