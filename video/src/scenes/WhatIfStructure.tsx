import React, {useMemo} from 'react';
import {AbsoluteFill, interpolate, random} from 'remotion';
import {useBeats} from '../beats';
import {Axes, scales} from '../components/Chart';
import {Show} from '../components/Show';
import {cloudAt, project, Space3D, V3} from '../components/Space3D';
import {At, Card, gly, Label, NodeDot, SvgDefs, Title} from '../components/ui';
import hero from '../../public/data/cloud_hero.json';
import knot from '../../public/data/cloud_knot.json';
import verdict from '../../public/data/verdict.json';
import hubble from '../../public/data/hubble.json';
import {NOISE} from './Data';
import {C, FONT, HEIGHT, WIDTH} from '../theme';

type Cell = {cm: number; seed: number; bo: number | null; knot: number | null; ok: boolean};
const CELLS = verdict.cells as Cell[];
const CMS = [...new Set(CELLS.map((c) => c.cm))].sort((a, b) => a - b);
const SEEDS = [...new Set(CELLS.map((c) => c.seed))].sort((a, b) => a - b);

// ---------------------------------------------------------------------------------------------
export const WhatIfObserver: React.FC = () => {
  const b = useBeats();
  const f = b.frame;
  const frames = hero.frames as number[][][];
  const pts = frames[frames.length - 1] as V3[];
  const cam = {yaw: f * 0.004, pitch: 0.3, distance: 60, scale: 36};
  const W = 1100;
  // scanning cursor (beat 1): hop between random particles, land on one
  const hops = 14;
  const scanP = interpolate(f, [b.at(1, 10), b.at(2) - 30], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const hop = Math.min(hops - 1, Math.floor(scanP * hops));
  const idx = Math.floor(random(`hop${hop}`) * pts.length);
  const q = project(pts[idx], cam, W, W);
  const tested = Math.round(scanP * 500);
  const goodCells = CELLS.filter((c) => c.bo !== null);
  const box = {x: 1180, y: 300, w: 600, h: 380};
  const sc = scales(box, [0.42, 0.47], [0, 1]);
  return (
    <AbsoluteFill>
      <div style={{position: 'absolute', left: (b.on(2) ? 520 : WIDTH / 2) - W / 2, top: -10}}>
        <Space3D
          width={W}
          height={W}
          camera={cam}
          layers={[{kind: 'points', pts, color: (i) => (i === idx && b.on(1) ? C.gold : '#bcd7ff'), size: 2.1, alpha: (i) => (i === idx && b.on(1) ? 1 : 0.7)}]}
        />
        <svg width={W} height={W} style={{position: 'absolute', left: 0, top: 0}}>
          {!b.on(1) && (
            <g opacity={b.p(0, 20, 20)}>
              <circle cx={W / 2} cy={W / 2} r={26} fill="none" stroke={C.hot} strokeWidth={5} />
              <line x1={W / 2 - 40} y1={W / 2 - 40} x2={W / 2 + 40} y2={W / 2 + 40} stroke={C.hot} strokeWidth={6} />
              <text x={W / 2} y={W / 2 + 80} fill={C.hot} fontFamily={FONT.display} fontWeight={800} fontSize={34} textAnchor="middle">
                the exact centre? unlikely
              </text>
            </g>
          )}
          {b.on(1) && q && (
            <g>
              <circle cx={q.x} cy={q.y} r={22} fill="none" stroke={C.gold} strokeWidth={4} />
            </g>
          )}
        </svg>
      </div>
      <Show from={1} until={2}>
        <At x={WIDTH / 2} y={140}>
          <Label size={30} color={C.ink} style={{fontWeight: 700}}>
            vantage points tested: <span style={{color: C.gold, fontFamily: FONT.mono}}>{tested}</span>
          </Label>
        </At>
      </Show>
      <Show from={1} until={2} offset={Math.max(0, b.at(2) - b.at(1) - 60)}>
        <At x={1560} y={540}>
          <Card accent={C.gold} style={{textAlign: 'center'}}>
            <Label size={26} color={C.gold} style={{fontFamily: FONT.display, fontWeight: 800}}>
              BEST VANTAGE IN THIS RUN
            </Label>
            <Title size={60}>{hubble.chi2.model.toFixed(3)}</Title>
            <Label size={24}>ΛCDM: {hubble.chi2.lcdm.toFixed(3)}</Label>
          </Card>
        </At>
      </Show>
      <Show from={2}>
        <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute'}}>
          <text x={box.x} y={box.y - 70} fill={C.ink} fontFamily={FONT.display} fontWeight={800} fontSize={34}>
            Best vantage in {goodCells.length} configurations
          </text>
          <Axes box={box} xd={[0.42, 0.47]} yd={[0, 1]} xticks={[0.42, 0.44, 0.46]} fmtX={(v) => v.toFixed(2)} xlabel="fit score (χ²/dof)" grid={false} />
          <rect x={sc.X(hubble.chi2.lcdm - NOISE)} y={box.y} width={sc.X(hubble.chi2.lcdm + NOISE) - sc.X(hubble.chi2.lcdm - NOISE)} height={box.h} fill={C.lcdm} opacity={0.14} />
          <text x={sc.X(hubble.chi2.lcdm)} y={box.y + box.h - 12} fill={C.lcdm} fontFamily={FONT.body} fontWeight={700} fontSize={20} textAnchor="middle" opacity={0.9}>
            noise ±{NOISE}
          </text>
          <line x1={sc.X(hubble.chi2.lcdm)} x2={sc.X(hubble.chi2.lcdm)} y1={box.y} y2={box.y + box.h} stroke={C.lcdm} strokeWidth={4} />
          <text x={sc.X(hubble.chi2.lcdm) - 10} y={box.y - 14} fill={C.lcdm} fontFamily={FONT.body} fontWeight={800} fontSize={24} textAnchor="end">
            ΛCDM {hubble.chi2.lcdm}
          </text>
          {goodCells.map((c, i) => {
            const x = sc.X(Math.min(0.47, c.bo!));
            const y = box.y + box.h - 20 - random(`jy${i}`) * (box.h - 50);
            return <circle key={i} cx={x} cy={y} r={9} fill={C.model} opacity={b.p(2, 20, i * 0.8)} />;
          })}
        </svg>
      </Show>
    </AbsoluteFill>
  );
};

// ---------------------------------------------------------------------------------------------
const CloudPanel: React.FC<{data: typeof hero; cx: number; p: number; yaw: number; title: string; sub: string; good: boolean; w?: number}> = ({data, cx, p, yaw, title, sub, good, w = 820}) => {
  const frames = data.frames as number[][][];
  const pts = cloudAt(frames, p * (frames.length - 1));
  const core = data.core as boolean[];
  return (
    <>
      <div style={{position: 'absolute', left: cx - w / 2, top: 50}}>
        <Space3D
          width={w}
          height={w}
          camera={{yaw, pitch: 0.3, distance: 60, scale: 30}}
          layers={[
            {kind: 'points', pts, color: (i) => (core[i] ? (good ? C.gold : C.hot) : '#8fb4f0'), size: 2.2, alpha: (i) => (core[i] ? 1 : 0.35)},
          ]}
        />
      </div>
      <At x={cx} y={815}>
        <div style={{textAlign: 'center'}}>
          <Title size={44} color={good ? C.gold : C.hot}>
            {title}
          </Title>
          <Label size={28} color={C.ink}>
            {sub}
          </Label>
        </div>
      </At>
    </>
  );
};

export const WhatIfClose: React.FC = () => {
  const b = useBeats();
  const f = b.frame;
  const CX = WIDTH / 2;
  const CY = HEIGHT / 2 - 20;
  const close = b.p(0, 40);
  const run = interpolate(f, [b.at(1), b.at(2)], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const simP = interpolate(f, [b.at(2), b.at(4) - 10], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const split = b.p(3, 25);
  const yaw = f * 0.005;
  const parts = useMemo(() => new Array(220).fill(0).map((_, i) => ({a: random(`ra${i}`) * Math.PI * 2, r: Math.sqrt(random(`rr${i}`))})), []);
  const checks = [
    ['fits the supernovae', 'best-observer χ²/dof ≤ 0.46'],
    ['grows by the right amount', 'within 20% of ΛCDM'],
    ['core doesn’t collapse', 'knot ratio ≥ 0.9'],
    ['no attractor inside the horizon', `nearest attractor > ${gly(15)}`],
  ];
  const nOk = CELLS.filter((c) => c.ok).length;
  return (
    <AbsoluteFill>
      {/* beats 0-1: nodes too close -> runaway */}
      {!b.on(2) && (
        <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute', opacity: 1 - b.p(2, 12)}}>
          <SvgDefs />
          {[-1, 1].map((s) => (
            <NodeDot key={s} x={CX + s * (720 - 330 * close)} y={CY} r={46} />
          ))}
          {/* the cloud is stretched toward both attractors, then torn apart */}
          <ellipse cx={CX} cy={CY} rx={110 * (1 + 2.6 * run ** 2)} ry={110 * (1 - 0.35 * run)} fill="url(#bubble)" stroke={C.bubble} strokeWidth={3} opacity={1 - run} />
          {parts.map((p, i) => {
            const x0 = CX + Math.cos(p.a) * p.r * 105;
            const y0 = CY + Math.sin(p.a) * p.r * 105;
            const side = Math.cos(p.a) >= 0 ? 1 : -1;
            const nx = CX + side * (720 - 330 * close);
            const u = run ** 2 * (1.1 + 0.5 * random(`ru${i}`)); // accelerating, each at its own pace
            if (random(`rf${i}`) < 0.62) {
              // swallowed: falls straight into the attractor
              const k = Math.min(1, u);
              return k >= 0.98 ? null : <circle key={i} cx={x0 + (nx - x0) * k} cy={y0 + (CY - y0) * k} r={2.6} fill="#cfe4ff" />;
            }
            // slingshot: whips past the attractor and is flung away on a bent path
            const off = (Math.sin(p.a) >= 0 ? 1 : -1) * (40 + 40 * random(`ro${i}`));
            if (u < 1) return <circle key={i} cx={x0 + (nx - x0) * u} cy={y0 + (CY + off - y0) * u} r={2.6} fill="#ffd9a8" />;
            const t = u - 1;
            const ang = (off > 0 ? 1 : -1) * (0.6 + 1.2 * t);
            return <circle key={i} cx={nx + side * Math.cos(ang) * t * 700} cy={CY + off + Math.sin(ang) * t * 700} r={2.6} fill="#ffd9a8" />;
          })}
        </svg>
      )}
      <Show from={1} until={2} offset={10}>
        <At x={CX} y={160}>
          <Title size={90} color={C.hot}>
            RUNAWAY
          </Title>
        </At>
        <At x={CX} y={240}>
          <Label size={32} color={C.ink} style={{fontWeight: 700}}>
            stretched · swallowed · slingshot away
          </Label>
        </At>
      </Show>

      {/* beats 2-3: real runs */}
      {b.on(2) && !b.on(4) && (
        <AbsoluteFill style={{opacity: b.p(2, 15) * (1 - b.p(4, 12))}}>
          <CloudPanel
            data={knot as unknown as typeof hero}
            cx={interpolate(split, [0, 1], [CX, 500])}
            p={simP}
            yaw={yaw}
            good={false}
            title={`core ×${knot.knot.toFixed(2)}: a KNOT`}
            sub={`attractors ~${gly(knot.S)} apart · fit ${knot.best_observer_chi2.toFixed(3)}`}
          />
          {split > 0 && (
            <div style={{opacity: split}}>
              <CloudPanel
                data={hero}
                cx={1420}
                p={simP}
                yaw={yaw + 1}
                good
                title={`core ×${hero.knot.toFixed(2)}: healthy`}
                sub={`attractors ~${gly(hero.S)} apart · fit ${hero.best_observer_chi2.toFixed(3)}`}
              />
            </div>
          )}
          <At x={CX} y={120}>
            <Label size={30} color={C.dim} style={{fontFamily: FONT.display, fontWeight: 800, letterSpacing: 2}}>
              REAL SIMULATIONS · CORE PARTICLES HIGHLIGHTED · {knot.t_Gyr[0]} → 13.8 Gyr
            </Label>
          </At>
        </AbsoluteFill>
      )}

      {/* beats 4-6: checklist */}
      <Show from={4} until={7}>
        <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
          <Card accent={C.gold} style={{padding: '40px 60px'}}>
            <Title size={56} style={{marginBottom: 26}}>
              Universe-like? <span style={{color: C.gold}}>Checklist</span>
            </Title>
            {checks.map(([a, d], i) => {
              const p = b.p(5, 15, i * Math.max(12, (b.at(6) - b.at(5)) / 5));
              return (
                <div key={i} style={{display: 'flex', alignItems: 'center', gap: 24, margin: '14px 0', opacity: 0.3 + 0.7 * p}}>
                  <div style={{width: 54, height: 54, borderRadius: 12, border: `4px solid ${C.eds}`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 40, color: C.eds, fontWeight: 900}}>
                    {p > 0.5 ? '✓' : ''}
                  </div>
                  <div>
                    <Title size={42}>{a}</Title>
                    <Label size={26}>{d}</Label>
                  </div>
                </div>
              );
            })}
          </Card>
        </AbsoluteFill>
      </Show>
      <Show from={6} until={7} offset={20}>
        <At x={CX} y={160}>
          <Title size={44} color={C.hot}>
            fit alone → picks the collapsing ones
          </Title>
        </At>
      </Show>

      {/* beat 7: 72-cell grid */}
      <Show from={7}>
        <AbsoluteFill style={{alignItems: 'center', paddingTop: 120}}>
          <Title size={50}>
            {CMS.length} progenitor masses × {SEEDS.length} random realizations
          </Title>
          <div style={{display: 'grid', gridTemplateColumns: `140px repeat(${CMS.length}, 104px)`, gap: 10, marginTop: 40}}>
            <div />
            {CMS.map((cm) => (
              <div key={cm} style={{textAlign: 'center', fontFamily: FONT.body, fontWeight: 700, fontSize: 24, color: C.dim}}>
                {cm}
              </div>
            ))}
            {SEEDS.map((seed, r) => (
              <React.Fragment key={seed}>
                <div style={{fontFamily: FONT.body, fontWeight: 700, fontSize: 22, color: C.dim, textAlign: 'right', paddingRight: 10, lineHeight: '50px'}}>seed {seed}</div>
                {CMS.map((cm, k) => {
                  const c = CELLS.find((x) => x.cm === cm && x.seed === seed);
                  const p = b.p(7, 10, 8 + (r * CMS.length + k) * 0.9);
                  return (
                    <div
                      key={cm}
                      style={{
                        height: 50,
                        borderRadius: 8,
                        background: !c ? C.faint : c.ok ? C.eds : C.hot,
                        opacity: p,
                        transform: `scale(${0.6 + 0.4 * p})`,
                      }}
                    />
                  );
                })}
              </React.Fragment>
            ))}
          </div>
          <div style={{display: 'flex', gap: 40, alignItems: 'baseline', marginTop: 30}}>
            <Label size={26}>progenitor mass (× our universe) →</Label>
            <Title size={80} color={C.eds}>
              {Math.round(nOk * b.p(7, 60, 30))} / {CELLS.length}
            </Title>
            <Label size={30} color={C.ink}>universe-like</Label>
          </div>
        </AbsoluteFill>
      </Show>
    </AbsoluteFill>
  );
};

// ---------------------------------------------------------------------------------------------
export const WhatIfMass: React.FC = () => {
  const b = useBeats();
  const f = b.frame;
  const CX = WIDTH / 2;
  const grow = interpolate(f, [b.at(0), b.at(1)], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const size = 8 * Math.pow(18, Math.sin(grow * Math.PI * 1.5) * 0.5 + 0.5);
  const top = {x: 380, y: 210, w: 1200, h: 240};
  const bot = {x: 380, y: 540, w: 1200, h: 240};
  const sTop = scales(top, [0.7, 4500], [0.42, 0.47], true);
  const sBot = scales(bot, [0.7, 4500], [0.4, 1.6], true);
  const typical = CMS[CMS.length - 1];
  return (
    <AbsoluteFill>
      {!b.on(1) && (
        <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute', opacity: 1 - b.p(1, 12)}}>
          <circle cx={CX} cy={HEIGHT / 2} r={size * 3} fill={C.gold} opacity={0.15} />
          <circle cx={CX} cy={HEIGHT / 2} r={size} fill="#120a00" stroke={C.gold} strokeWidth={5} />
        </svg>
      )}
      {b.on(1) && (
        <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute', opacity: b.p(1, 15)}}>
          <Axes box={top} xd={[0.7, 4500]} yd={[0.42, 0.47]} logX xticks={CMS} yticks={[0.43, 0.45, 0.47]} fmtY={(v) => v.toFixed(2)} ylabel="fit" />
          <Axes box={bot} xd={[0.7, 4500]} yd={[0.4, 1.6]} logX xticks={CMS} yticks={[0.6, 1, 1.4]} xlabel="progenitor mass (× our observable universe)" ylabel="core ratio" />
          <line x1={top.x} x2={top.x + top.w} y1={sTop.Y(hubble.chi2.lcdm)} y2={sTop.Y(hubble.chi2.lcdm)} stroke={C.lcdm} strokeWidth={3} strokeDasharray="8 8" />
          <line x1={bot.x} x2={bot.x + bot.w} y1={sBot.Y(0.9)} y2={sBot.Y(0.9)} stroke={C.hot} strokeWidth={3} strokeDasharray="8 8" />
          <text x={bot.x + bot.w + 10} y={sBot.Y(0.9) + 8} fill={C.hot} fontFamily={FONT.body} fontWeight={700} fontSize={22}>collapse</text>
          <text x={top.x + top.w + 10} y={sTop.Y(hubble.chi2.lcdm) + 8} fill={C.lcdm} fontFamily={FONT.body} fontWeight={700} fontSize={22}>ΛCDM</text>
          {CELLS.map((c, i) => {
            const jx = Math.pow(1.12, random(`mx${i}`) * 2 - 1);
            const p = b.p(2, 15, (i % 20) * 1.5);
            return (
              <g key={i} opacity={p}>
                {c.bo !== null && <circle cx={sTop.X(c.cm * jx)} cy={sTop.Y(Math.min(0.47, c.bo))} r={8} fill={C.model} opacity={0.85} />}
                {c.knot !== null && <circle cx={sBot.X(c.cm * jx)} cy={sBot.Y(Math.max(0.4, Math.min(1.6, c.knot)))} r={8} fill={c.knot >= 0.9 ? C.eds : C.hot} opacity={0.85} />}
              </g>
            );
          })}
          {b.on(3) && (
            <g opacity={b.p(3, 20)}>
              <rect x={sTop.X(typical) - 40} y={top.y - 20} width={80} height={bot.y + bot.h - top.y + 40} rx={14} fill="none" stroke={C.gold} strokeWidth={5} />
            </g>
          )}
        </svg>
      )}
      <Show from={2}>
        <At x={1700} y={330}>
          <Label size={30} color={C.model} style={{fontWeight: 800}}>
            fit: flat
          </Label>
        </At>
        <At x={1720} y={660}>
          <Label size={30} color={C.eds} style={{fontWeight: 800}}>
            structure: varies
          </Label>
        </At>
      </Show>
      <Show from={3}>
        <At x={Math.min(sTop.X(typical), WIDTH - 470)} y={150}>
          <Card accent={C.gold} style={{padding: '12px 24px'}}>
            <Title size={36} color={C.gold}>
              typical member (as heavy as its neighbours) · most robust
            </Title>
          </Card>
        </At>
      </Show>
    </AbsoluteFill>
  );
};
