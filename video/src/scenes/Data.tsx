import React from 'react';
import {AbsoluteFill} from 'remotion';
import {interpolate} from '../interp';
import {useBeats} from '../beats';
import {Axes, LegendItem, scales} from '../components/Chart';
import {Show} from '../components/Show';
import {At, Card, Label, pathOf, partial, Tex, Title} from '../components/ui';
import hubble from '../../public/data/hubble.json';
import verdict from '../../public/data/verdict.json';
import {C, FONT, HEIGHT, WIDTH} from '../theme';
import {useFmt, useT} from '../i18n';

const BOX = {x: 330, y: 190, w: 1260, h: 560};
/** Best-observer χ²/dof moves ~±0.005 with particle count / observer sampling (lode PF28). */
export const NOISE = 0.005;
/** Best best-observer fit among the 68 universe-like cells of the 72-cell campaign. */
const BEST_RUN = Math.min(...(verdict.cells as {bo: number | null; ok: boolean}[]).filter((c) => c.ok && c.bo !== null).map((c) => c.bo as number));
const XD: [number, number] = [0.01, 2.3];

export const Data: React.FC = () => {
  const tr = useT();
  const fmt = useFmt();
  const b = useBeats();
  const f = b.frame;
  const h = hubble;
  // ---- μ(z) view (beats 1-2)
  const muView = scales(BOX, XD, [33, 47], true);
  const dotsP = b.p(1, 50);
  const nShown = Math.round(h.n_sne * dotsP);
  const modelMuP = b.p(3, 50, 20);
  // ---- residual view vs ΛCDM (beats 3+)
  const RD: [number, number] = [-0.75, 0.3];
  const rv = scales(BOX, XD, RD, true);
  const resO = b.p(4, 20);
  const muO = 1 - resO;
  const edsP = b.p(4, 40, 20);
  const lcdmP = b.p(5, 30);
  const modelP = interpolate(f, [b.at(6, 25), b.at(7, 10)], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const edsRes = h.grid.z.map((z, i) => [rv.X(z), rv.Y(h.resid.eds[i])] as [number, number]);
  const modelRes = h.model.z.map((z, i) => [rv.X(z), rv.Y(h.model.resid[i])] as [number, number]);
  const chartGone = b.p(8, 15);

  const bars = [
    {l: tr('data.bar_eds'), v: h.chi2.eds, c: C.eds, beat: 8},
    {l: tr('data.bar_lcdm'), v: h.chi2.lcdm, c: C.lcdm, beat: 8},
    {l: tr('data.bar_model'), v: h.chi2.model, c: C.model, beat: 8},
    {l: tr('data.bar_best'), v: BEST_RUN, c: '#ffc27a', beat: 9},
  ];
  return (
    <AbsoluteFill>
      <Show from={0} until={1}>
        <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
          <div style={{transform: `rotate(-3deg) scale(${0.6 + 0.4 * b.s(0)})`, background: C.gold, padding: '10px 50px'}}>
            <Title size={160} color={C.bg}>
              {tr('data.real_data')}</Title>
          </div>
        </AbsoluteFill>
      </Show>

      {b.on(1) && chartGone < 1 && (
        <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute', opacity: 1 - chartGone}}>
          {/* μ(z) */}
          <g opacity={muO}>
            <Axes box={BOX} xd={XD} yd={[33, 47]} logX xticks={[0.01, 0.1, 1, 2]} yticks={[34, 38, 42, 46]} xlabel={tr('data.redshift_z')} ylabel={tr('data.distance_modulus')} />
            {h.sne.z.slice(0, nShown).map((z, i) => (
              <circle key={i} cx={muView.X(z)} cy={muView.Y(h.sne.mu[i])} r={3.4} fill="#c9d4f5" opacity={0.55} />
            ))}
            {/* beat 2: this is the data Λ was fitted to */}
            {b.on(2) && (
              <path d={pathOf(partial(h.grid.z.map((z, i) => [muView.X(z), muView.Y(h.grid.lcdm[i])] as [number, number]), b.p(2, 40, 10)))} stroke={C.lcdm} strokeWidth={6} fill="none" opacity={1 - b.p(3, 15)} />
            )}
            <path d={pathOf(partial(h.model.z.map((z, i) => [muView.X(z), muView.Y(h.model.mu[i])] as [number, number]), modelMuP))} stroke={C.model} strokeWidth={6} fill="none" />
            {b.on(2) && !b.on(3) && (
              <text x={muView.X(0.3)} y={muView.Y(37.5)} fill={C.lcdm} fontFamily={FONT.display} fontWeight={800} fontSize={36} opacity={b.p(2, 15, 40)}>
                {tr('data.cdm_fitted_to_this')}</text>
            )}
            {b.on(3) && !b.on(4) && (
              <text x={muView.X(0.3)} y={muView.Y(37.5)} fill={C.model} fontFamily={FONT.display} fontWeight={800} fontSize={36} opacity={b.p(3, 15, 40)}>
                {tr('data.our_simulation_same_test')}</text>
            )}
          </g>
          {/* Δμ vs ΛCDM */}
          {b.on(4) && (
            <g opacity={resO}>
              <Axes box={BOX} xd={XD} yd={RD} logX xticks={[0.01, 0.1, 1, 2]} yticks={[-0.6, -0.4, -0.2, 0, 0.2]} fmtY={(v) => (v > 0 ? '+' : '') + fmt(v)} xlabel={tr('data.redshift_z')} ylabel={tr('data.difference_from_cdm_mag')} />
              {h.resid.bins_z.map((z, i) => {
                const e = (h.resid.bins_err[i] / (RD[1] - RD[0])) * BOX.h;
                const y = rv.Y(h.resid.bins_d[i]);
                return (
                  <g key={i}>
                    <line x1={rv.X(z)} x2={rv.X(z)} y1={y - e} y2={y + e} stroke="#c9d4f5" strokeWidth={3} opacity={0.7} />
                    <circle cx={rv.X(z)} cy={y} r={8} fill="#c9d4f5" />
                  </g>
                );
              })}
              <path d={pathOf(partial(edsRes, edsP))} stroke={C.eds} strokeWidth={6} strokeDasharray="14 10" fill="none" />
              <line x1={BOX.x} x2={BOX.x + BOX.w * lcdmP} y1={rv.Y(0)} y2={rv.Y(0)} stroke={C.lcdm} strokeWidth={7} />
              {modelP > 0 && <path d={pathOf(partial(modelRes, modelP))} stroke={C.model} strokeWidth={7} fill="none" />}
              <LegendItem x={BOX.x + 30} y={BOX.y + 520} color="#c9d4f5" text={tr('data.legend_sne', {n: fmt(h.n_sne)})} />
              <LegendItem x={BOX.x + 30} y={BOX.y + 470} color={C.eds} dashed text={tr('data.no_dark_energy')} opacity={edsP} />
              <LegendItem x={BOX.x + 700} y={BOX.y + 470} color={C.lcdm} text={tr('data.cdm')} opacity={lcdmP} />
              <LegendItem x={BOX.x + 700} y={BOX.y + 520} color={C.model} text={tr('data.attractor_model_from_the')} opacity={modelP > 0 ? 1 : 0} />
            </g>
          )}
          {b.on(1) && !b.on(4) && (
            <text x={BOX.x + 40} y={BOX.y + 40} fill={C.ink} fontFamily={FONT.display} fontWeight={800} fontSize={40}>
              {fmt(nShown)} {tr('data.supernovae')}</text>
          )}
        </svg>
      )}
      <Show from={3} until={4} offset={10}>
        <At x={1300} y={640}>
          <Card accent={C.model} style={{padding: '18px 26px'}}>
            <Tex tex={String.raw`\mu(z)=5\log_{10}\frac{(1+z)\,c\int dt/a(t)}{10\,\mathrm{pc}}`} size={40} />
            <Label size={24} style={{textAlign: 'center'}}>{tr('data.a_t_straight_from')}</Label>
          </Card>
        </At>
      </Show>
      <Show from={4} until={5} offset={45}>
        <At x={1250} y={680}>
          <Title size={48} color={C.eds}>
            {tr('data.misses')}</Title>
        </At>
      </Show>
      <Show from={7} until={8}>
        <At x={WIDTH / 2} y={120}>
          <Title size={64} color={C.model}>
            {tr('data.right_on_top')}</Title>
        </At>
      </Show>

      {/* χ² bars */}
      <Show from={8} until={11}>
        <AbsoluteFill style={{alignItems: 'center', paddingTop: 150}}>
          <Title size={58}>
            {tr('data.fit_score')}<span style={{color: C.dim, fontSize: 36}}>{tr('data.2_dof_lower_is')}</span>
          </Title>
          <div style={{marginTop: 50, display: 'flex', flexDirection: 'column', gap: 26, width: 1400}}>
            {bars.map((bar, i) => {
              const p = bar.beat === 8 ? b.p(8, 30, 25 + i * 35) : b.p(bar.beat, 30, 10);
              return (
                <div key={i} style={{display: 'flex', alignItems: 'center', gap: 30}}>
                  <div style={{width: 470, textAlign: 'right', fontFamily: FONT.body, fontWeight: 700, fontSize: 34, color: C.ink}}>{bar.l}</div>
                  <div style={{height: 64, width: 800 * (bar.v / 0.9) * p, background: bar.c, borderRadius: 10, boxShadow: `0 0 30px ${bar.c}55`}} />
                  <div style={{fontFamily: FONT.mono, fontWeight: 500, fontSize: 48, color: bar.c, opacity: p}}>{fmt(bar.v, 3)}</div>
                </div>
              );
            })}
          </div>
        </AbsoluteFill>
      </Show>
      <Show from={9} until={10} offset={40}>
        <At x={WIDTH / 2} y={760}>
          <Card accent={C.gold} style={{textAlign: 'center'}}>
            <Title size={46}>
              {tr('data.it')}<span style={{color: C.gold}}>{tr('data.matches')}</span> {tr('data.cdm')}</Title>
            <Label size={28}>{tr('data.gaps_below')}{fmt(NOISE)} {tr('data.are_simulation_noise_converged')}</Label>
          </Card>
        </At>
      </Show>
      <Show from={10} until={11}>
        <At x={WIDTH / 2} y={760}>
          <Card accent={C.hot} style={{textAlign: 'center'}}>
            <Title size={50}>
              {tr('data.more_tuning_better_score')}<span style={{color: C.hot}}>{tr('data.proves_nothing')}</span>
            </Title>
          </Card>
        </At>
      </Show>
      <Show from={11} until={12}>
        <Overfit />
      </Show>
      <Show from={12}>
        <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', flexDirection: 'column', gap: 30}}>
          <Title size={110}>
            {tr('data.fitting')}<span style={{color: C.hot}}>≠</span> {tr('data.explaining')}</Title>
          <Label size={38} color={C.ink} style={{fontWeight: 700}}>
            {tr('data.the_fit_is_only')}<span style={{color: C.gold}}>{tr('data.first')}</span> {tr('data.test')}</Label>
          <Label size={30} style={{opacity: b.p(12, 20, 60)}}>{tr('data.next_does_the_simulated')}</Label>
        </AbsoluteFill>
      </Show>
    </AbsoluteFill>
  );
};

/** Overfitting, illustrated: an honest trend line vs. a curve that threads every noisy point. */
const Overfit: React.FC = () => {
  const tr = useT();
  const b = useBeats();
  const box = {x: 380, y: 230, w: 1160, h: 480};
  const pts = new Array(11).fill(0).map((_, i) => {
    const x = box.x + 40 + (i / 10) * (box.w - 80);
    const trend = box.y + box.h - 80 - (i / 10) * (box.h - 180);
    const noise = [30, -42, 18, 50, -35, 8, -55, 40, -20, 45, -30][i];
    return [x, trend + noise] as [number, number];
  });
  const honest = `M${pts[0][0]},${box.y + box.h - 80} L${pts[10][0]},${box.y + 100}`;
  // a cubic through every point, with control handles swung far up/down: fits all, means nothing
  let wiggle = `M${pts[0][0]},${pts[0][1]}`;
  for (let i = 0; i < 10; i++) {
    const [x0, y0] = pts[i];
    const [x1, y1] = pts[i + 1];
    const s = i % 2 ? 1 : -1;
    wiggle += ` C${x0 + (x1 - x0) / 3},${y0 + s * 110} ${x1 - (x1 - x0) / 3},${y1 - s * 110} ${x1},${y1}`;
  }
  const honestP = b.p(11, 30, 20);
  const wiggleP = b.p(11, 50, 60);
  return (
    <AbsoluteFill>
      <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute'}}>
        <line x1={box.x} y1={box.y + box.h} x2={box.x + box.w} y2={box.y + box.h} stroke={C.dim} strokeWidth={2.5} />
        <line x1={box.x} y1={box.y} x2={box.x} y2={box.y + box.h} stroke={C.dim} strokeWidth={2.5} />
        <path d={honest} stroke={C.eds} strokeWidth={6} fill="none" strokeDasharray="2000" strokeDashoffset={2000 * (1 - honestP)} />
        <path d={wiggle} stroke={C.hot} strokeWidth={5} fill="none" strokeDasharray="6000" strokeDashoffset={6000 * (1 - wiggleP)} />
        {pts.map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r={11} fill="#c9d4f5" />
        ))}
        <LegendItem x={box.x + 40} y={box.y - 20} color={C.eds} text={tr('data.simple_model_captures_the')} opacity={honestP} />
        <LegendItem x={box.x + 640} y={box.y - 20} color={C.hot} text={tr('data.too_many_knobs_hits')} opacity={wiggleP} />
      </svg>
      <At x={WIDTH / 2} y={120}>
        <Title size={64}>
          {tr('data.over')}<span style={{color: C.hot}}>{tr('data.fitting_2')}</span>
        </Title>
      </At>
      <At x={WIDTH / 2} y={800}>
        <Label size={32} color={C.ink} style={{fontWeight: 700, opacity: b.p(11, 20, 90)}}>
          {tr('data.a_perfect_fit_can')}</Label>
      </At>
    </AbsoluteFill>
  );
};
