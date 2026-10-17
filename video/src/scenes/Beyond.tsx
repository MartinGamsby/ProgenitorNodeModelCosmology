import React, {useMemo} from 'react';
import {AbsoluteFill, interpolate, random} from 'remotion';
import {useBeats} from '../beats';
import {Show} from '../components/Show';
import {Camera, project, Space3D, V3} from '../components/Space3D';
import {At, Card, Label, Title} from '../components/ui';
import medium from '../../public/data/medium.json';
import {C, FONT, HEIGHT, WIDTH} from '../theme';

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
  const grid = useMemo(() => {
    const g: V3[] = [];
    for (let x = -6; x <= 6; x++) for (let y = -6; y <= 6; y++) for (let z = -6; z <= 6; z++) if (x || y || z) g.push([x * 30, y * 30, z * 30]);
    return g;
  }, []);

  // zoom path (px per Gpc), keyed to beats
  const scale = interpolate(
    f,
    [0, b.at(1), b.at(2), b.at(4), b.at(5), b.at(8), b.at(8, 60)],
    [40, 26, 6, 4.2, 1.15, 1.0, 3.2],
    {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
  );
  const cam = camAt(scale, f * 0.0035 + 0.3);
  const gridO = Math.min(b.p(2, 20), 1 - b.p(3, 25, 10));
  // our universe turns on its own axis too, so it never looks frozen while the camera orbits
  const spin = f * 0.012;
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
          {kind: 'points', pts: grid, color: '#9fb2e6', size: 2.2, alpha: gridO * 0.6},
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
              radius ≈ 46 billion
            </Title>
            <Title size={50} color={C.bubble}>
              light-years
            </Title>
            <Label size={26}>the observable universe</Label>
          </Card>
        </At>
      </Show>

      <Show from={2} until={3}>
        <At x={WIDTH / 2} y={170}>
          <Title size={60} style={{textAlign: 'center'}}>
            “Same out there as in here. Forever.”
          </Title>
        </At>
      </Show>
      <Show from={3} until={4}>
        <At x={WIDTH / 2} y={170}>
          <Title size={100} color={C.gold}>
            …or not?
          </Title>
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
                  HMEA
                </Title>
                <Label size={28} color={C.ink}>
                  Hyper-Massive External Attractor
                </Label>
                <Label size={24}>≫ mass of our whole universe · dark</Label>
              </Card>
            </div>
          </>
        )}
      </Show>
      <Show from={6} until={7}>
        <At x={WIDTH / 2} y={150}>
          <Card style={{textAlign: 'center'}}>
            <Title size={44} color={C.ink}>
              a lumpy, <span style={{color: C.gold}}>settled swarm</span>
            </Title>
            <Label size={28}>like galaxies in a relaxed cluster, only absurdly bigger</Label>
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
            YOU ARE HERE
          </Label>
        </At>
      </Show>
    </AbsoluteFill>
  );
};

/** "Virialized", explained: a frozen crystal vs. a swarm that keeps moving but keeps its shape. */
const CrystalVsSwarm: React.FC = () => {
  const b = useBeats();
  const f = b.frame;
  const swarm = useMemo(
    () =>
      new Array(36).fill(0).map((_, i) => ({
        r: 40 + random(`sw${i}`) * 150,
        e: 0.4 + random(`se${i}`) * 0.6,
        tilt: random(`st${i}`) * Math.PI,
        w: (0.012 + random(`sv${i}`) * 0.02) * (random(`sd${i}`) > 0.5 ? 1 : -1),
        ph: random(`sp${i}`) * Math.PI * 2,
        m: 0.6 + random(`sm${i}`) * 1.2,
      })),
    [],
  );
  const L = {x: 540, y: 500};
  const R = {x: 1380, y: 500};
  return (
    <AbsoluteFill style={{background: 'rgba(4,6,13,0.86)'}}>
      <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute'}}>
        {[-2, -1, 0, 1, 2].flatMap((i) =>
          [-2, -1, 0, 1, 2].map((j) => <circle key={`${i}${j}`} cx={L.x + i * 70} cy={L.y + j * 70} r={13} fill={C.node} opacity={0.85} />),
        )}
        {swarm.map((s, i) => {
          const a = s.ph + f * s.w;
          const x0 = Math.cos(a) * s.r;
          const y0 = Math.sin(a) * s.r * s.e;
          const x = R.x + x0 * Math.cos(s.tilt) - y0 * Math.sin(s.tilt);
          const y = R.y + x0 * Math.sin(s.tilt) + y0 * Math.cos(s.tilt);
          return <circle key={i} cx={x} cy={y} r={9 * s.m} fill={C.node} opacity={0.85} />;
        })}
        <circle cx={R.x} cy={R.y} r={200} fill="none" stroke={C.faint} strokeWidth={2} strokeDasharray="6 10" />
      </svg>
      <At x={L.x} y={190}>
        <Title size={52} color={C.dim}>
          crystal
        </Title>
      </At>
      <At x={L.x} y={790}>
        <Label size={30} color={C.ink} style={{fontWeight: 700}}>
          frozen in place
        </Label>
      </At>
      <At x={R.x} y={190}>
        <Title size={52} color={C.gold}>
          virialized swarm
        </Title>
      </At>
      <At x={R.x} y={790}>
        <Label size={30} color={C.ink} style={{fontWeight: 700}}>
          always moving · keeps its shape
        </Label>
      </At>
    </AbsoluteFill>
  );
};
