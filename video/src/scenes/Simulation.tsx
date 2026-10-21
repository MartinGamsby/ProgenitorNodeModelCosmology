import React from 'react';
import {AbsoluteFill, interpolate} from 'remotion';
import {useBeats} from '../beats';
import {Axes, LegendItem, scales} from '../components/Chart';
import {Show} from '../components/Show';
import {cloudAt, Space3D} from '../components/Space3D';
import {Arrow, At, Card, Label, pathOf, partial, Title} from '../components/ui';
import hero from '../../public/data/cloud_hero.json';
import growth from '../../public/data/growth.json';
import {C, FONT, HEIGHT, WIDTH} from '../theme';
import {useFmt, useGly, useT} from '../i18n';

const CLOUD_W = 1000;
const CLOUD_CX = 520;

export const Simulation: React.FC = () => {
  const tr = useT();
  const fmt = useFmt();
  const gly = useGly();
  const b = useBeats();
  const f = b.frame;
  const frames = hero.frames as number[][][];
  const t = hero.t_Gyr as number[];
  // the sim plays from beat 3 to the end of the scene
  const simP = interpolate(f, [b.at(3), b.dur - 20], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const fi = simP * (frames.length - 1);
  const pts = cloudAt(frames, fi);
  const tNow = interpolate(fi, [0, frames.length - 1], [t[0], t[t.length - 1]]);
  const cloudO = b.p(1, 25);
  const yaw = f * 0.006;

  // growth chart (right side)
  const box = {x: 1180, y: 250, w: 620, h: 420};
  const {X, Y} = scales(box, [2.9, 13.8], [1, 3.6]);
  const g = growth;
  const line = (arr: number[]) => g.t.map((tt, i) => [X(tt), Y(arr[i])] as [number, number]);
  const chartO = b.p(4, 20);
  const edsP = b.p(4, 40, 10);
  const zeroDots = b.p(5, 40);
  const modelP = interpolate(f, [b.at(6), b.dur - 20], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const lcdmO = b.p(6, 20);
  const last = (arr: number[]) => arr[arr.length - 1];
  return (
    <AbsoluteFill>
      <div style={{position: 'absolute', left: CLOUD_CX - CLOUD_W / 2, top: 40, opacity: cloudO}}>
        <Space3D
          width={CLOUD_W}
          height={CLOUD_W}
          camera={{yaw, pitch: 0.3, distance: 60, scale: 36}}
          layers={[{kind: 'points', pts, color: (i) => (i % 5 ? '#bcd7ff' : '#ffd9a8'), size: 2.1, alpha: 0.85}]}
        />
      </div>
      {/* forces: inward self-gravity, outward node pulls */}
      <Show from={2} until={3}>
        <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute'}}>
          {[0, 1, 2, 3, 4, 5].map((k) => {
            const a = (k / 6) * Math.PI * 2 + 0.3;
            const cx = CLOUD_CX;
            const cy = 540;
            return (
              <g key={k}>
                <Arrow x1={cx + Math.cos(a) * 190} y1={cy + Math.sin(a) * 190} x2={cx + Math.cos(a) * 120} y2={cy + Math.sin(a) * 120} color={C.lcdm} width={6} head={18} p={b.p(2, 15)} />
                <Arrow x1={cx + Math.cos(a + 0.5) * 230} y1={cy + Math.sin(a + 0.5) * 230} x2={cx + Math.cos(a + 0.5) * 400} y2={cy + Math.sin(a + 0.5) * 400} color={C.node} width={7} head={22} p={b.p(2, 15, 40)} />
              </g>
            );
          })}
        </svg>
        <At x={CLOUD_CX} y={130}>
          <Label size={32} color={C.lcdm} style={{fontWeight: 800}}>
            {tr('simulation.self_gravity_brakes')}</Label>
        </At>
        <At x={CLOUD_CX} y={870}>
          <Label size={32} color={C.node} style={{fontWeight: 800}}>
            {tr('simulation.distant_attractors_pull_apart')}</Label>
        </At>
      </Show>
      <Show from={1} until={2}>
        <At x={1460} y={420}>
          <Card>
            <Title size={50}>{tr('simulation.n_body_simulation')}</Title>
            <Label size={30} color={C.ink}>
              {fmt(hero.n_inner)} {tr('simulation.particles_plain_newtonian_gravity')}</Label>
            <Label size={26}>{tr('simulation.pull_from_300_attractors')}{gly(hero.S)} {tr('simulation.apart')}</Label>
          </Card>
        </At>
      </Show>

      {/* clock */}
      <Show from={3}>
        <div style={{position: 'absolute', left: 80, top: 120}}>
          <Label size={26} style={{fontFamily: FONT.display, fontWeight: 800, letterSpacing: 2}}>
            {tr('simulation.age_of_the_universe')}</Label>
          <Title size={72} color={C.gold} style={{fontFamily: FONT.mono}}>
            {fmt(tNow, 1)} {tr('simulation.gyr')}</Title>
        </div>
        <div style={{position: 'absolute', left: 84, top: 230}}>
          <svg width={400} height={60}>
            <line x1={0} x2={36 * 5} y1={20} y2={20} stroke={C.dim} strokeWidth={4} />
            <line x1={0} x2={0} y1={10} y2={30} stroke={C.dim} strokeWidth={4} />
            <line x1={180} x2={180} y1={10} y2={30} stroke={C.dim} strokeWidth={4} />
            <text x={200} y={30} fill={C.dim} fontFamily={FONT.body} fontWeight={700} fontSize={26}>
              {gly(5, true)}
            </text>
          </svg>
        </div>
      </Show>

      {/* growth chart */}
      {b.on(4) && (
        <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute', opacity: chartO}}>
          <rect x={box.x - 150} y={box.y - 110} width={box.w + 200} height={box.h + 230} rx={22} fill="rgba(10,16,34,0.85)" stroke={C.faint} strokeWidth={2} />
          <text x={box.x - 120} y={box.y - 60} fill={C.ink} fontFamily={FONT.display} fontWeight={800} fontSize={34}>
            {tr('simulation.how_much_the_cloud')}</text>
          <Axes box={box} xd={[2.9, 13.8]} yd={[1, 3.6]} xticks={[2.9, 6, 10, 13.8]} yticks={[1, 2, 3]} fmtY={(v) => `×${v}`} xlabel={tr('simulation.time_gyr')} />
          <path d={pathOf(partial(line(g.eds), edsP))} stroke={C.eds} strokeWidth={6} strokeDasharray="12 9" fill="none" />
          {g.t.map((tt, i) =>
            i % 8 === 0 && i / g.t.length < zeroDots ? <circle key={i} cx={X(tt)} cy={Y(g.eds[i])} r={8} fill="none" stroke="#fff" strokeWidth={3} /> : null,
          )}
          <path d={pathOf(line(g.lcdm))} stroke={C.lcdm} strokeWidth={6} fill="none" opacity={lcdmO} />
          {b.on(6) && <path d={pathOf(partial(line(g.model), modelP))} stroke={C.model} strokeWidth={7} fill="none" />}
          <LegendItem x={box.x + 20} y={box.y + 10} color={C.eds} dashed text={tr('simulation.no_dark_energy_eds')} />
          <LegendItem x={box.x + 20} y={box.y + 55} color="#ffffff" text={tr('simulation.sim_with_m_0')} opacity={zeroDots} />
          <LegendItem x={box.x + 20} y={box.y + 100} color={C.lcdm} text={tr('simulation.cdm')} opacity={lcdmO} />
          <LegendItem x={box.x + 20} y={box.y + 145} color={C.model} text={tr('simulation.sim_with_attractors')} opacity={modelP > 0 ? 1 : 0} />
          <g opacity={b.p(7, 20)} fontFamily={FONT.body} fontWeight={800} fontSize={26}>
            <text x={X(13.8) + 12} y={Y(last(g.lcdm)) - 12} fill={C.lcdm}>{tr('simulation.x')}{fmt(last(g.lcdm), 2)}</text>
            <text x={X(13.8) + 12} y={Y(last(g.model)) + 22} fill={C.model}>{tr('simulation.x')}{fmt(last(g.model), 2)}</text>
            <text x={X(13.8) + 12} y={Y(last(g.eds)) + 10} fill={C.eds}>{tr('simulation.x')}{fmt(last(g.eds), 2)}</text>
          </g>
        </svg>
      )}
      <Show from={5} until={6} offset={30}>
        <At x={1490} y={850}>
          <Title size={40} color={C.eds}>
            {tr('simulation.m_0_exactly_einstein')}</Title>
        </At>
      </Show>
    </AbsoluteFill>
  );
};
