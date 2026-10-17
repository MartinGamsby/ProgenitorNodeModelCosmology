import React from 'react';
import {AbsoluteFill, interpolate} from 'remotion';
import {useBeats} from '../beats';
import {Axes, LegendItem, scales} from '../components/Chart';
import {Show} from '../components/Show';
import {At, Label, partial, pathOf, Tex, Title} from '../components/ui';
import hubble from '../../public/data/hubble.json';
import {C, FONT, HEIGHT, WIDTH} from '../theme';

const BOX = {x: 300, y: 170, w: 1180, h: 600};
const XD: [number, number] = [0.01, 2.3];
const YD: [number, number] = [-0.3, 0.85];

export const Problem: React.FC = () => {
  const b = useBeats();
  const f = b.frame;
  // Classic 1998-style plot: brightness relative to a slowing (no-dark-energy) universe,
  // normalised to agree at the lowest redshift.
  const {X, Y} = scales(BOX, XD, YD, true);
  const showChart = b.on(1) && !b.on(3);
  const chartO = Math.min(b.p(1, 15), 1 - b.p(3, 12));
  const dotsP = b.p(1, 45);
  const edsP = b.p(1, 40, 25);
  const lcdmP = b.p(2, 40, 15);
  const g = hubble.grid;
  const r = hubble.resid;
  const d0 = -r.eds[0]; // LCDM - EdS at z = 0.01
  const lcdmMinusEds = (z: number) => {
    const k = g.z.findIndex((v) => v >= z);
    const j = Math.max(1, k < 0 ? g.z.length - 1 : k);
    const u = (z - g.z[j - 1]) / (g.z[j] - g.z[j - 1]);
    return -(r.eds[j - 1] + (r.eds[j] - r.eds[j - 1]) * u) - d0;
  };
  const edsPts = g.z.map((z) => [X(z), Y(0)] as [number, number]);
  const lcdmPts = g.z.map((z) => [X(z), Y(lcdmMinusEds(z))] as [number, number]);
  const bins = r.bins_z.map((z, i) => ({x: X(z), y: Y(r.bins_d[i] + lcdmMinusEds(z)), e: (r.bins_err[i] / (YD[1] - YD[0])) * BOX.h}));

  // supernova flash in beat 0
  const flash = interpolate(f, [b.at(0, 20), b.at(0, 28), b.at(0, 70)], [0, 1, 0.25], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

  // coincidence chart (beat 5): densities vs time
  const cBox = {x: 360, y: 200, w: 1200, h: 520};
  // log time axis 0.3 → 300 Gyr: on this scale "now" and the crossing sit together
  const ts = new Array(160).fill(0).map((_, i) => 0.3 * 1000 ** (i / 159)); // Gyr
  const H0 = 70 / 977.792;
  const aOf = (t: number) => Math.sinh(1.5 * Math.sqrt(0.7) * H0 * t) ** (2 / 3) / Math.sinh(1.5 * Math.sqrt(0.7) * H0 * 13.8) ** (2 / 3);
  const CY_D: [number, number] = [-3, 3];
  const cs = scales(cBox, [0.3, 300], CY_D, true);
  const ratio = (t: number) => Math.log10(0.3 / 0.7 / aOf(t) ** 3); // log10(matter / dark energy)
  const rhoM = ts.filter((t) => Math.abs(ratio(t)) <= 3).map((t) => [cs.X(t), cs.Y(ratio(t))] as [number, number]);
  const rhoL = ts.map((t) => [cs.X(t), cs.Y(0)] as [number, number]);
  // the window where the two are within a factor of 3 of each other
  const within = ts.filter((t) => Math.abs(ratio(t)) <= Math.log10(3));
  const win = [within[0], within[within.length - 1]];

  return (
    <AbsoluteFill>
      {/* beat 0: 1998, a supernova */}
      <Show from={0} until={1}>
        <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute'}}>
          <defs>
            <radialGradient id="sn">
              <stop offset="0%" stopColor="#fff" />
              <stop offset="20%" stopColor="#fff2c4" stopOpacity={0.9} />
              <stop offset="100%" stopColor={C.gold} stopOpacity={0} />
            </radialGradient>
          </defs>
          <ellipse cx={1150} cy={520} rx={260} ry={90} fill="#9cc9ff" opacity={0.12} transform="rotate(-18 1150 520)" />
          <ellipse cx={1150} cy={520} rx={130} ry={40} fill="#cfe4ff" opacity={0.25} transform="rotate(-18 1150 520)" />
          <circle cx={1150} cy={520} r={14} fill="#fff" opacity={0.8} />
          <circle cx={1260} cy={480} r={40 + 160 * flash} fill="url(#sn)" opacity={flash} />
        </svg>
        <At x={560} y={430}>
          <Title size={170} color={C.gold}>
            1998
          </Title>
        </At>
        <At x={560} y={580}>
          <Label size={36} color={C.ink} style={{textAlign: 'center', width: 640}}>
            Type Ia supernovae: exploding stars with a known brightness
          </Label>
        </At>
      </Show>

      {/* beats 1-2: real Pantheon+ SNe vs the slowing universe, then Λ */}
      {showChart && (
        <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute', opacity: chartO}}>
          <Axes
            box={BOX}
            xd={XD}
            yd={YD}
            logX
            xticks={[0.01, 0.1, 1, 2]}
            yticks={[-0.2, 0, 0.2, 0.4, 0.6]}
            fmtY={(v) => (v > 0 ? `+${v}` : `${v}`)}
            xlabel="redshift z  (how far back)"
            ylabel="extra dimness (mag)"
          />
          {bins.map((p, i) =>
            i / bins.length < dotsP ? (
              <g key={i}>
                <line x1={p.x} x2={p.x} y1={p.y - p.e} y2={p.y + p.e} stroke="#c9d4f5" strokeWidth={3} opacity={0.7} />
                <circle cx={p.x} cy={p.y} r={9} fill="#c9d4f5" />
              </g>
            ) : null,
          )}
          <path d={pathOf(partial(edsPts, edsP))} stroke={C.eds} strokeWidth={6} fill="none" strokeDasharray="14 10" />
          {b.on(2) && <path d={pathOf(partial(lcdmPts, lcdmP))} stroke={C.lcdm} strokeWidth={6} fill="none" />}
          <LegendItem x={BOX.x + 40} y={BOX.y + 30} color="#c9d4f5" text="Real supernovae (Pantheon+, binned)" />
          <LegendItem x={BOX.x + 40} y={BOX.y + 80} color={C.eds} dashed text="Slowing universe (no dark energy)" opacity={edsP} />
          <LegendItem x={BOX.x + 40} y={BOX.y + 130} color={C.lcdm} text="With Λ (ΛCDM)" opacity={lcdmP} />
          {b.on(1, 50) && !b.on(2) && (
            <text x={X(0.025)} y={Y(0.4)} fill={C.gold} fontFamily={FONT.display} fontWeight={800} fontSize={40} opacity={b.p(1, 15, 50)}>
              ↑ too dim = too far!
            </text>
          )}
        </svg>
      )}
      <Show from={2} until={3} offset={10}>
        <At x={1690} y={480}>
          <Title size={220} color={C.lcdm} style={{textShadow: `0 0 50px ${C.lcdm}`}}>
            Λ
          </Title>
        </At>
        <At x={1690} y={640}>
          <Label size={30} color={C.ink} style={{textAlign: 'center', width: 360}}>
            energy of empty space
          </Label>
        </At>
      </Show>

      {/* beat 3: what ΛCDM means */}
      <Show from={3} until={4}>
        <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', flexDirection: 'column', gap: 40}}>
          <Label size={36} color={C.dim} style={{fontFamily: FONT.display, fontWeight: 800, letterSpacing: 2}}>
            THE STANDARD MODEL OF THE UNIVERSE
          </Label>
          <Title size={170} style={{letterSpacing: 4}}>
            <span style={{color: C.lcdm}}>Λ</span>
            <span style={{color: '#8fa3d8'}}>CDM</span>
          </Title>
          <div style={{display: 'flex', gap: 120, opacity: b.p(3, 20, 25)}}>
            <div style={{textAlign: 'center'}}>
              <Title size={64} color={C.lcdm}>Λ</Title>
              <Label size={34} color={C.ink}>dark energy</Label>
              <Label size={26}>(“Lambda”)</Label>
            </div>
            <div style={{textAlign: 'center'}}>
              <Title size={64} color="#8fa3d8">CDM</Title>
              <Label size={34} color={C.ink}>cold dark matter</Label>
              <Label size={26}>(invisible, slow-moving matter)</Label>
            </div>
          </div>
        </AbsoluteFill>
      </Show>

      {/* beat 4: two problems */}
      <Show from={4} until={5}>
        <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', flexDirection: 'column', gap: 30}}>
          <Title size={120}>It works. But…</Title>
          <Title size={70} color={C.hot}>
            2 embarrassing problems
          </Title>
        </AbsoluteFill>
      </Show>

      {/* beat 4: 10^120 */}
      <Show from={5} until={6}>
        <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', flexDirection: 'column', gap: 10}}>
          <Label size={44} color={C.dim} style={{fontFamily: FONT.display, fontWeight: 800}}>
            PROBLEM #1 · FINE-TUNING
          </Label>
          <div style={{display: 'flex', alignItems: 'baseline', gap: 30, transform: `scale(${0.7 + 0.3 * b.s(5, 8)})`}}>
            <Tex tex={'10^{120}'} size={250} color={C.hot} />
            <Title size={90}>× too big</Title>
          </div>
          <div
            style={{
              fontFamily: FONT.mono,
              fontSize: 30,
              color: C.faint,
              width: 1500,
              wordBreak: 'break-all',
              textAlign: 'center',
              lineHeight: 1.25,
            }}
          >
            1{'0'.repeat(Math.round(120 * b.p(5, 60, 10)))}
          </div>
        </AbsoluteFill>
      </Show>

      {/* beat 5: coincidence */}
      <Show from={6} until={7}>
        <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute'}}>
          <rect x={cs.X(win[0])} y={cBox.y} width={cs.X(win[1]) - cs.X(win[0])} height={cBox.h} fill={C.gold} opacity={0.1 * b.p(6, 15, 40)} />
          <Axes box={cBox} xd={[0.3, 300]} yd={CY_D} logX xticks={[1, 13.8, 100]} fmtX={(v) => (v === 13.8 ? 'now' : `${v}`)} xlabel="cosmic time (billion years, log scale)" ylabel="density (log)" grid={false} />
          <path d={pathOf(partial(rhoM, b.p(6, 50)))} stroke={C.gold} strokeWidth={7} fill="none" />
          <path d={pathOf(partial(rhoL, b.p(6, 50)))} stroke={C.node} strokeWidth={7} fill="none" />
          <text x={cs.X(0.45)} y={cs.Y(2.3)} fill={C.gold} fontFamily={FONT.body} fontWeight={700} fontSize={32}>matter (thins out)</text>
          <text x={cs.X(40)} y={cs.Y(0) - 22} fill={C.node} fontFamily={FONT.body} fontWeight={700} fontSize={32}>dark energy (constant)</text>
          <text x={(cs.X(win[0]) + cs.X(win[1])) / 2} y={cBox.y + cBox.h - 24} fill={C.gold} fontFamily={FONT.body} fontWeight={700} fontSize={26} textAnchor="middle" opacity={b.p(6, 15, 40)}>
            comparable only here
          </text>
          <g opacity={b.p(6, 15, 55)}>
            <line x1={cs.X(13.8)} x2={cs.X(13.8)} y1={cBox.y} y2={cBox.y + cBox.h} stroke={C.ink} strokeWidth={3} strokeDasharray="8 8" />
            <circle cx={cs.X(13.8)} cy={cs.Y(Math.log10(0.3 / 0.7))} r={16} fill="none" stroke={C.hot} strokeWidth={5} />
            <text x={cs.X(13.8) + 26} y={cBox.y + 40} fill={C.hot} fontFamily={FONT.display} fontWeight={800} fontSize={40}>us. right now?!</text>
          </g>
        </svg>
        <At x={WIDTH / 2} y={130}>
          <Label size={44} color={C.dim} style={{fontFamily: FONT.display, fontWeight: 800}}>
            PROBLEM #2 · THE COINCIDENCE
          </Label>
        </At>
      </Show>

      {/* beat 6 */}
      <Show from={7}>
        <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', flexDirection: 'column', gap: 20}}>
          <Title size={110}>Fitting</Title>
          <Title size={140} color={C.hot}>≠</Title>
          <Title size={110} color={C.gold}>Explaining</Title>
        </AbsoluteFill>
      </Show>
    </AbsoluteFill>
  );
};
