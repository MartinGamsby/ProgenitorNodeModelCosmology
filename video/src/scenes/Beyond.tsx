import React, {useMemo} from 'react';
import {AbsoluteFill, interpolate, random} from 'remotion';
import {useBeats} from '../beats';
import {Show} from '../components/Show';
import {Camera, project, Space3D, V3} from '../components/Space3D';
import {At, Card, Label, Title} from '../components/ui';
import medium from '../../public/data/medium.json';
import {C, FONT, HEIGHT, WIDTH} from '../theme';
import {useT} from '../i18n';

const R_OBS = 14; // Gpc

/** Camera at a given zoom: `scale` px per Gpc, gentle perspective. */
export const camAt = (scale: number, yaw: number, pitch = 0.32): Camera => ({
  yaw,
  pitch,
  scale,
  distance: (3 * 540) / scale,
});

export const useMedium = () =>
  useMemo(() => ({pos: medium.pos as V3[], mass: medium.mass as number[], S: medium.S}), []);

export const Beyond: React.FC = () => {
  const tr = useT();
  const b = useBeats();
  const f = b.frame;
  const {pos, mass} = useMedium();
  const dust = useMemo(
    () =>
      new Array(900).fill(0).map((_, i): V3 => {
        const u = random(`d${i}`) * 2 - 1;
        const ph = random(`p${i}`) * Math.PI * 2;
        const r = R_OBS * Math.cbrt(random(`r${i}`)) * 0.97;
        const s = Math.sqrt(1 - u * u);
        return [r * s * Math.cos(ph), r * u, r * s * Math.sin(ph)];
      }),
    [],
  );
  // the standard view: beyond our horizon, more of the same (galaxies like ours, everywhere),
  // drawn exactly like the galaxies inside our bubble, just continuing outward
  const grid = useMemo(
    () =>
      new Array(5200).fill(0).map((_, i): V3 => [
        (random(`ux${i}`) * 2 - 1) * 100,
        (random(`uy${i}`) * 2 - 1) * 100,
        (random(`uz${i}`) * 2 - 1) * 100,
      ]).filter((p) => Math.hypot(...p) > R_OBS),
    [],
  );
  const gridR = useMemo(() => grid.map((p) => Math.hypot(...p)), [grid]);

  // zoom path (px per Gpc), keyed to beats
  const scale = interpolate(
    f,
    [0, b.at(1), b.at(2), b.at(4), b.at(5), b.at(8), b.at(8, 60)],
    [40, 26, 6, 4.2, 1.15, 1.0, 3.2],
    {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
  );
  const cam = camAt(scale, f * 0.0035 + 0.3);
  // standard view (beat 2): more of the same, forever. This model (beat 3+): more of the same
  // only out to the edge of the Big Bang's matter (illustrative radius), then empty space.
  const R_BANG = 3 * R_OBS;
  const insideO = b.p(2, 20) * 0.55;
  const outsideO = Math.min(b.p(2, 20), 1 - b.p(3, 25, 10)) * 0.55;
  const edgeO = b.p(3, 25, 20);
  // our universe turns on its own axis too, so it never looks frozen while the camera orbits
  const spin = -f * 0.012; // same sense as the camera orbit
  const spunDust = dust.map(([x, y, z]): V3 => [x * Math.cos(spin) - z * Math.sin(spin), y, x * Math.sin(spin) + z * Math.cos(spin)]);
  const nodeO = b.p(4, 40);
  const nodes = pos;

  // HMEA label: pick a bright node in front
  const labelNode = useMemo(() => {
    let best = 0;
    nodes.forEach((p, i) => {
      if (mass[i] > mass[best] && Math.hypot(...p) < 260) best = i;
    });
    return best;
  }, [nodes, mass]);
  const lp = project(nodes[labelNode], cam, WIDTH, HEIGHT);
  const bubbleP = project([0, 0, 0], cam, WIDTH, HEIGHT)!;

  return (
    <AbsoluteFill>
      <Space3D
        width={WIDTH}
        height={HEIGHT}
        camera={cam}
        layers={[
          {kind: 'points', pts: grid, color: (i) => (i % 3 ? '#cfe4ff' : '#ffd9a8'), size: Math.max(1.1, scale * 0.05), alpha: (i) => (gridR[i] <= R_BANG ? insideO : outsideO)},
          {kind: 'sphere', center: [0, 0, 0], radius: R_BANG, color: C.gold, alpha: edgeO * 0.8, outline: true},
          {kind: 'sphere', center: [0, 0, 0], radius: R_OBS, color: C.bubble, alpha: Math.min(1, b.p(0, 30, 10) + 0.0)},
          {kind: 'points', pts: spunDust, color: (i) => (i % 3 ? '#cfe4ff' : '#ffd9a8'), size: Math.max(1.1, scale * 0.05), alpha: interpolate(scale, [1, 8], [0.6, 0.9], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'})},
          {kind: 'glows', pts: nodes, radius: (i) => (5 + 6 * Math.cbrt(mass[i])) * Math.max(1, scale / 1.3) ** 0.5, color: C.nodeGlow, core: C.node, alpha: nodeO},
        ]}
        style={{position: 'absolute'}}
      />

      <Show from={1} until={2}>
        <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute'}}>
          <line x1={bubbleP.x} y1={bubbleP.y} x2={bubbleP.x + R_OBS * scale} y2={bubbleP.y} stroke={C.gold} strokeWidth={4} strokeDasharray="10 8" />
        </svg>
        <At x={bubbleP.x + R_OBS * scale + 250} y={bubbleP.y}>
          <Card accent={C.bubble}>
            <Title size={50} color={C.bubble}>
              {tr('beyond.radius_46_billion')}</Title>
            <Title size={50} color={C.bubble}>
              {tr('beyond.light_years')}</Title>
            <Label size={26}>{tr('beyond.the_observable_universe')}</Label>
          </Card>
        </At>
      </Show>

      <Show from={2} until={3}>
        <At x={WIDTH / 2} y={170}>
          <Title size={60} style={{textAlign: 'center'}}>
            {tr('beyond.standard_view_more_of')}</Title>
        </At>
      </Show>
      <Show from={3} until={4}>
        <At x={WIDTH / 2} y={170}>
          <Title size={46} color={C.gold} style={{textAlign: 'center'}}>
            {tr('beyond.this_model_more_of')}</Title>
        </At>
      </Show>

      <Show from={5} until={7}>
        {lp && (
          <>
            <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute'}}>
              <line x1={lp.x} y1={lp.y} x2={lp.x + 140} y2={lp.y - 110} stroke={C.node} strokeWidth={3} />
            </svg>
            <div style={{position: 'absolute', left: lp.x + 150, top: lp.y - 200}}>
              <Card accent={C.node} style={{padding: '20px 28px'}}>
                <Title size={58} color={C.node}>
                  {tr('beyond.hmea')}</Title>
                <Label size={28} color={C.ink}>
                  {tr('beyond.hyper_massive_external_attractor')}</Label>
                <Label size={24}>{tr('beyond.mass_of_our_whole')}</Label>
              </Card>
            </div>
          </>
        )}
      </Show>
      <Show from={6} until={7}>
        <At x={WIDTH / 2} y={150}>
          <Card style={{textAlign: 'center'}}>
            <Title size={44} color={C.ink}>
              {tr('beyond.a_lumpy')}<span style={{color: C.gold}}>{tr('beyond.settled_swarm')}</span>
            </Title>
            <Label size={28}>{tr('beyond.like_galaxies_in_a')}</Label>
          </Card>
        </At>
      </Show>
      <Show from={7} until={8}>
        <CrystalVsSwarm />
      </Show>
      <Show from={8} offset={50}>
        <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute'}}>
          <circle cx={bubbleP.x} cy={bubbleP.y} r={R_OBS * scale + 18} fill="none" stroke={C.gold} strokeWidth={4} strokeDasharray="12 10" />
        </svg>
        <At x={bubbleP.x} y={bubbleP.y + R_OBS * scale + 60}>
          <Label size={36} color={C.gold} style={{fontFamily: FONT.display, fontWeight: 900}}>
            {tr('beyond.you_are_here')}</Label>
        </At>
      </Show>
    </AbsoluteFill>
  );
};

/** "Virialized", explained, in 3D at the same scale. Left: a cube of ionic crystal (two node
 *  sizes, like table salt), frozen. Right: the REAL relaxed swarm from the simulation (all 300
 *  nodes of medium.json). Every member jiggles around its spot, lighter ones a bit more, but the
 *  overall shape never changes. Both rotate together. */
const BOX = 537; // half-size of the relaxed medium box (Gpc)
const CRYSTAL: {p: V3; m: number}[] = (() => {
  const n = 6;
  const step = (2 * BOX) / (n - 1);
  const out: {p: V3; m: number}[] = [];
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      for (let k = 0; k < n; k++) out.push({p: [-BOX + i * step, -BOX + j * step, -BOX + k * step], m: (i + j + k) % 2 ? 0.45 : 1.6});
  return out;
})();

const CrystalVsSwarm: React.FC = () => {
  const tr = useT();
  const b = useBeats();
  const f = b.frame;
  const W = 840;
  const cam = {yaw: 0.5 + f * 0.004, pitch: 0.35, distance: 3200, scale: 0.38};
  const pos = medium.pos as V3[];
  const mass = medium.mass as number[];
  const swarm = useMemo(
    () => pos.map((_, i) => ({ph: random(`vj${i}`) * Math.PI * 2, ax: random(`vx${i}`) * 2 - 1, ay: random(`vy${i}`) * 2 - 1, az: random(`vz${i}`) * 2 - 1})),
    [pos],
  );
  const jig = 0.14 * medium.S;
  const moving = pos.map((q, i): V3 => {
    const s = swarm[i];
    const a = s.ph + f * (0.035 / Math.sqrt(mass[i]));
    const amp = jig / Math.sqrt(mass[i]);
    return [q[0] + Math.cos(a) * amp * s.ax, q[1] + Math.sin(a * 1.3) * amp * s.ay, q[2] + Math.cos(a * 0.7) * amp * s.az];
  });
  const r = (m: number) => 2.6 + 3.4 * Math.cbrt(m);
  return (
    <AbsoluteFill style={{background: 'rgba(4,6,13,0.9)'}}>
      <div style={{position: 'absolute', left: 540 - W / 2, top: 500 - W / 2}}>
        <Space3D width={W} height={W} camera={cam} layers={[{kind: 'glows', pts: CRYSTAL.map((c) => c.p), radius: (i) => r(CRYSTAL[i].m), color: C.nodeGlow, core: C.node, alpha: 0.75}]} />
      </div>
      <div style={{position: 'absolute', left: 1380 - W / 2, top: 500 - W / 2}}>
        <Space3D width={W} height={W} camera={cam} layers={[{kind: 'glows', pts: moving, radius: (i) => r(mass[i]), color: C.nodeGlow, core: C.node, alpha: 0.75}]} />
      </div>
      <At x={540} y={130}>
        <Title size={52} color={C.dim}>
          {tr('beyond.crystal')}</Title>
      </At>
      <At x={540} y={790}>
        <Label size={30} color={C.ink} style={{fontWeight: 700}}>
          {tr('beyond.frozen_in_place')}</Label>
      </At>
      <At x={1380} y={130}>
        <Title size={52} color={C.gold}>
          {tr('beyond.virialized_swarm')}</Title>
      </At>
      <At x={1380} y={790}>
        <Label size={30} color={C.ink} style={{fontWeight: 700}}>
          {tr('beyond.always_moving_keeps_its')}</Label>
      </At>
      <At x={1380} y={828}>
        <Label size={22}>{tr('beyond.the_relaxed_swarm_from')}</Label>
      </At>
    </AbsoluteFill>
  );
};
