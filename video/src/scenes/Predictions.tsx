import React from 'react';
import {AbsoluteFill, interpolate} from 'remotion';
import {useBeats} from '../beats';
import {Axes, LegendItem, scales} from '../components/Chart';
import {Show} from '../components/Show';
import {Arrow, At, NodeDot, pathOf, partial, SvgDefs, Title} from '../components/ui';
import {C, FONT, HEIGHT, WIDTH} from '../theme';

const CX = WIDTH / 2;
const SKY = {cx: CX, cy: 470, rx: 560, ry: 280};
const AXIS = -0.45; // radians: the shared preferred direction on the map
/** Where the shared axis points, OUTSIDE the sky map: our nearest, heaviest neighbours. */
const MARK = {x: SKY.cx + Math.cos(AXIS) * 700, y: SKY.cy + Math.sin(AXIS) * 700};
/** Where the dark flow converges on the map: inside the red region, on the axis toward MARK. */
const FLOW = {x: SKY.cx + Math.cos(AXIS) * 470, y: SKY.cy + Math.sin(AXIS) * 230};
// Hubble tension: Planck 2018 (CMB, whole sky) and SH0ES (Riess et al. 2022, distance ladder)
const H0 = [
  {v: 67.4, e: 0.5, c: C.lcdm, l: 'whole sky (CMB, Planck)'},
  {v: 73.04, e: 1.04, c: C.hot, l: 'nearby galaxies (SH0ES)'},
];

/** Beats: 0 fingerprint · 1 dipole · 2-3 Hubble tension · 4 tension expected here (anisotropy) ·
 *  5-6 Axis of Evil · 7 dark flow · 8 phantom: pull grows as we drift (accelerating) ·
 *  9 phantom: w < −1 · 10 all the same way (attractor marker outside the map, on the axis) */
export const Predictions: React.FC = () => {
  const b = useBeats();
  const f = b.frame;
  const ax = Math.cos(AXIS);
  const ay = Math.sin(AXIS);
  // sky map visible except during the tension (2-3) and phantom (7-8) beats
  const skyO = Math.min(
    1 - b.p(2, 12) + b.p(5, 12),
    1 - b.p(8, 12) + b.p(10, 12),
  );
  const dip = b.p(1, 30);
  const multi = b.p(5, 30);
  const flow = b.p(7, 30);
  const final = b.p(10, 30);

  // tension number line
  const tBox = {x: 1080, y: 330, w: 700, h: 260};
  const ts = scales(tBox, [64, 76], [0, 1]);
  const U = {x: 540, y: 520, r: 300};
  // beat 2 reveals, in order: us, the whole sky (blue, both sides at once), nearby galaxies (red)
  const usP = b.p(2, 12, 6);
  const blueP = interpolate(b.through(2), [0.12, 0.22], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const redP = interpolate(b.through(2), [0.55, 0.65], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const tensionO = Math.min(b.p(2, 15), 1 - b.p(5, 12));

  // phantom: drift toward the attractor (beat 7), then w(t) (beat 8)
  // starts slow and speeds up as the pull grows: the speed-up speeds up
  const drift = interpolate(f, [b.at(8), b.at(9)], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: (t) => t ** 2.6});
  const driftO = Math.min(b.p(8, 15), 1 - b.p(9, 12));
  const bx = 520 + 520 * drift;
  const nodeX = 1500;
  const pull = 60 + 260 / ((nodeX - bx) / 400) ** 2 * 0.25;
  const wBox = {x: 520, y: 200, w: 880, h: 480};
  const ws = scales(wBox, [0, 1], [-1.35, -0.85]);
  const xs = new Array(60).fill(0).map((_, i) => i / 59);
  const wModel = xs.map((u) => [ws.X(u), ws.Y(-1 - 0.25 * u ** 2.2)] as [number, number]);
  const wO = Math.min(b.p(9, 15), 1 - b.p(10, 12));
  return (
    <AbsoluteFill>
      {/* ---- sky map ---- */}
      <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute', opacity: Math.max(0, skyO)}}>
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
          {[-2, -1, 0, 1, 2].map((k) => (
            <ellipse key={k} cx={SKY.cx} cy={SKY.cy} rx={Math.abs(k) * SKY.rx * 0.33 || 1} ry={SKY.ry} fill="none" stroke={C.faint} strokeWidth={1.5} />
          ))}
          {[-0.66, -0.33, 0.33, 0.66].map((k) => (
            <line key={k} x1={SKY.cx - SKY.rx} x2={SKY.cx + SKY.rx} y1={SKY.cy + k * SKY.ry} y2={SKY.cy + k * SKY.ry} stroke={C.faint} strokeWidth={1.5} />
          ))}
          <rect x={SKY.cx - SKY.rx} y={SKY.cy - SKY.ry} width={SKY.rx * 2} height={SKY.ry * 2} fill="url(#dipole)" opacity={dip} />
          <g opacity={multi} transform={`rotate(${(AXIS * 180) / Math.PI} ${SKY.cx} ${SKY.cy})`}>
            {[0, 1].map((k) => (
              <ellipse key={`q${k}`} cx={SKY.cx + (k ? 1 : -1) * 230} cy={SKY.cy} rx={150} ry={70} fill="none" stroke={C.gold} strokeWidth={5} />
            ))}
            {[-1, 0, 1].map((k) => (
              <ellipse key={`o${k}`} cx={SKY.cx + k * 150} cy={SKY.cy} rx={50} ry={120} fill="none" stroke={C.node} strokeWidth={4} strokeDasharray="10 8" />
            ))}
            <line x1={SKY.cx - 600} x2={SKY.cx + 600} y1={SKY.cy} y2={SKY.cy} stroke={C.gold} strokeWidth={3} strokeDasharray="14 10" />
          </g>
          {flow > 0 &&
            [-3, -2, -1, 0, 1, 2, 3].flatMap((i) =>
              [-1, 0, 1].map((j) => {
                const x = SKY.cx + i * 140;
                const y = SKY.cy + j * 110;
                if (Math.hypot(FLOW.x - x, FLOW.y - y) < 150 || ((x - SKY.cx) / SKY.rx) ** 2 + ((y - SKY.cy) / SKY.ry) ** 2 > 0.85) return null;
                const L = 70 * flow;
                const tt = ((f * 2 + i * 20 + j * 37) % 60) / 60;
                // every region drifts toward the same place: our nearest, heaviest neighbours
                const dl = Math.hypot(FLOW.x - x, FLOW.y - y);
                const ux = (FLOW.x - x) / dl;
                const uy = (FLOW.y - y) / dl;
                return <Arrow key={`${i}${j}`} x1={x + ux * L * tt} y1={y + uy * L * tt} x2={x + ux * L * (tt + 0.8)} y2={y + uy * L * (tt + 0.8)} color={C.eds} width={4} head={14} opacity={flow} />;
              }),
            )}
        </g>
        {final > 0 && (
          <g opacity={final}>
            <line x1={SKY.cx} y1={SKY.cy} x2={MARK.x} y2={MARK.y} stroke={C.gold} strokeWidth={4} strokeDasharray="14 10" />
            <circle cx={MARK.x} cy={MARK.y} r={46 + 8 * Math.sin(f * 0.2)} fill="none" stroke={C.gold} strokeWidth={6} />
            <text x={MARK.x} y={MARK.y + 100} fill={C.gold} fontFamily={FONT.body} fontWeight={800} fontSize={28} textAnchor="middle">
              nearest, heaviest neighbours
            </text>
            <SvgDefs />
            <NodeDot x={MARK.x} y={MARK.y} r={26} />
          </g>
        )}
      </svg>

      {/* ---- Hubble tension: our universe on the left, the two measurements on the right ---- */}
      {tensionO > 0 && (
        <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute', opacity: tensionO}}>
          <SvgDefs />
          {/* our universe, "us" in the middle */}
          <circle cx={U.x} cy={U.y} r={U.r} fill="url(#bubble)" stroke={C.bubble} strokeWidth={3} />
          {/* whole sky at once: the cosmic microwave background, all around the edge */}
          <circle cx={U.x} cy={U.y} r={U.r - 6} fill="none" stroke={C.lcdm} strokeWidth={10} opacity={0.75 * blueP} />
          {[0, 1, 2, 3, 4, 5].map((k) => {
            const a = (k / 6) * Math.PI * 2 + 0.2;
            return <circle key={k} cx={U.x + Math.cos(a) * (U.r - 6)} cy={U.y + Math.sin(a) * (U.r - 6)} r={14} fill={C.lcdm} opacity={blueP} />;
          })}
          {/* nearby galaxies (distance ladder), on both sides */}
          {[[-95, -40], [-70, 55], [-120, 10], [90, -55], [115, 25], [70, 60]].map(([dx, dy], k) => (
            <circle key={k} cx={U.x + dx} cy={U.y + dy} r={11} fill={C.hot} opacity={redP} />
          ))}
          <circle cx={U.x} cy={U.y} r={12} fill="#fff" opacity={usP} />
          <text x={U.x} y={U.y + 42} fill="#fff" fontFamily={FONT.body} fontWeight={800} fontSize={28} textAnchor="middle" opacity={usP}>
            us
          </text>
          {/* beat 4: a lopsided (anisotropic) expansion: faster toward one side */}
          {b.on(4) &&
            [0, 1, 2, 3, 4, 5, 6, 7].map((k) => {
              const a = (k / 8) * Math.PI * 2;
              const L = (90 + 70 * Math.max(0, Math.cos(a - AXIS))) * b.p(4, 30, 30);
              return <Arrow key={k} x1={U.x + Math.cos(a) * 30} y1={U.y + Math.sin(a) * 30} x2={U.x + Math.cos(a) * (30 + L)} y2={U.y + Math.sin(a) * (30 + L)} color={C.gold} width={5} head={16} />;
            })}

          {/* the two measurements */}
          <Axes box={tBox} xd={[64, 76]} yd={[0, 1]} xticks={[64, 68, 72, 76]} xlabel="expansion rate H₀ (km/s/Mpc)" grid={false} />
          {H0.map((m, i) => {
            const y = tBox.y + 70 + i * 100;
            const p = i ? redP : blueP;
            return (
              <g key={i} opacity={p}>
                <line x1={ts.X(m.v - m.e)} x2={ts.X(m.v + m.e)} y1={y} y2={y} stroke={m.c} strokeWidth={6} />
                <circle cx={ts.X(m.v)} cy={y} r={14} fill={m.c} />
                <text x={ts.X(m.v)} y={y - 26} fill={m.c} fontFamily={FONT.display} fontWeight={900} fontSize={46} textAnchor="middle">
                  {Math.round(m.v)}
                </text>
                <text x={ts.X(m.v)} y={y + 42} fill={C.ink} fontFamily={FONT.body} fontWeight={700} fontSize={22} textAnchor="middle">
                  {m.l}
                </text>
              </g>
            );
          })}
          <g opacity={b.p(3, 20)}>
            <rect x={ts.X(H0[0].v)} y={tBox.y + 10} width={ts.X(H0[1].v) - ts.X(H0[0].v)} height={tBox.h - 20} fill={C.gold} opacity={0.08} />
            <text x={tBox.x + tBox.w / 2} y={tBox.y - 30} fill={C.gold} fontFamily={FONT.display} fontWeight={800} fontSize={34} textAnchor="middle">
              ≈ 8% apart: the Hubble tension
            </text>
          </g>
          <g opacity={b.p(4, 20)} fontFamily={FONT.display} fontWeight={800}>
            <text x={tBox.x + tBox.w / 2} y={tBox.y + tBox.h + 150} fill={C.dim} fontSize={30} textAnchor="middle">
              standard cosmology: a headache
            </text>
            <text x={tBox.x + tBox.w / 2} y={tBox.y + tBox.h + 200} fill={C.gold} fontSize={36} textAnchor="middle" opacity={b.p(4, 20, 50)}>
              this model: expected (anisotropy)
            </text>
          </g>
        </svg>
      )}

      {/* ---- phantom: the pull grows as we drift closer ---- */}
      {driftO > 0 && (
        <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute', opacity: driftO}}>
          <SvgDefs />
          <NodeDot x={nodeX} y={480} r={70} />
          <circle cx={bx} cy={480} r={110} fill="url(#bubble)" stroke={C.bubble} strokeWidth={3} />
          <Arrow x1={bx + 120} y1={480} x2={bx + 120 + Math.min(pull, nodeX - bx - 220)} y2={480} color={C.node} width={10} head={30} />
          <text x={bx} y={660} fill={C.ink} fontFamily={FONT.body} fontWeight={700} fontSize={30} textAnchor="middle">
            closer → stronger pull
          </text>
        </svg>
      )}
      {wO > 0 && (
        <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute', opacity: wO}}>
          <Axes box={wBox} xd={[0, 1]} yd={[-1.35, -0.85]} xticks={[0, 1]} fmtX={(v) => (v ? 'later' : 'earlier')} yticks={[-1.3, -1.2, -1.1, -1, -0.9]} fmtY={(v) => v.toFixed(1)} xlabel="time →" ylabel="w (dark-energy “type”)" />
          <line x1={wBox.x} x2={wBox.x + wBox.w} y1={ws.Y(-1)} y2={ws.Y(-1)} stroke={C.lcdm} strokeWidth={6} />
          <path d={pathOf(partial(wModel, b.p(9, 50, 10)))} stroke={C.model} strokeWidth={7} fill="none" />
          <LegendItem x={wBox.x + 30} y={wBox.y + 30} color={C.lcdm} text="cosmological constant: w = −1, forever" />
          <LegendItem x={wBox.x + 30} y={wBox.y + 80} color={C.model} text="attractors: w drifts below −1 (shape illustrative)" />
          <text x={ws.X(0.62)} y={ws.Y(-1.21)} fill={C.model} fontFamily={FONT.display} fontWeight={800} fontSize={36}>
            phantom
          </text>
        </svg>
      )}

      {/* ---- captions over the visuals ---- */}
      <Show from={0} until={1}>
        <At x={CX} y={830}>
          <Title size={54}>
            the fingerprint: a <span style={{color: C.gold}}>lopsided</span> expansion
          </Title>
        </At>
      </Show>
      <Show from={1} until={2}>
        <At x={CX} y={830}>
          <Title size={46}>
            <span style={{color: C.gold}}>#1</span> Hubble dipole: <span style={{color: C.hot}}>faster</span> one way, <span style={{color: C.lcdm}}>slower</span> the other
          </Title>
        </At>
      </Show>
      <Show from={2} until={5}>
        <At x={CX} y={150}>
          <Title size={50}>Two ways to measure the expansion rate</Title>
        </At>
      </Show>
      <Show from={5} until={7}>
        <At x={CX} y={830}>
          <Title size={46}>
            <span style={{color: C.gold}}>#2</span> CMB “Axis of Evil”: biggest patterns share one axis
          </Title>
        </At>
      </Show>
      <Show from={7} until={8}>
        <At x={CX} y={830}>
          <Title size={46}>
            <span style={{color: C.gold}}>#3</span> Dark flow: galaxies drifting in bulk
          </Title>
        </At>
      </Show>
      <Show from={8} until={10}>
        <At x={CX} y={130}>
          <Title size={46}>
            <span style={{color: C.gold}}>#4</span> Phantom acceleration: the speed-up speeds up
          </Title>
        </At>
      </Show>
      <Show from={10}>
        <At x={CX} y={830}>
          <Title size={50}>
            all pointing <span style={{color: C.gold}}>the same way</span>
          </Title>
        </At>
      </Show>
    </AbsoluteFill>
  );
};
