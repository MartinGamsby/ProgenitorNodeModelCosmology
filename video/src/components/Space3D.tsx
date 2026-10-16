import React, {useLayoutEffect, useRef} from 'react';

export type V3 = [number, number, number];

export type PointLayer = {
  kind: 'points';
  pts: V3[];
  color: string | ((i: number) => string);
  size: number; // px at depth 1
  alpha?: number | ((i: number) => number);
};

export type GlowLayer = {
  kind: 'glows';
  pts: V3[];
  radius: (i: number) => number; // px at depth 1
  color: string;
  core?: string;
  alpha?: number | ((i: number) => number);
};

export type SphereLayer = {kind: 'sphere'; center: V3; radius: number; color: string; alpha?: number; label?: string};

export type Layer = PointLayer | GlowLayer | SphereLayer;

export type Camera = {
  yaw: number; // radians
  pitch: number;
  distance: number; // in scene units (e.g. Gpc)
  scale: number; // px per scene unit at the focal plane
  target?: V3;
};

export const project = (p: V3, cam: Camera, w: number, h: number) => {
  const t = cam.target ?? [0, 0, 0];
  const x0 = p[0] - t[0];
  const y0 = p[1] - t[1];
  const z0 = p[2] - t[2];
  const cy = Math.cos(cam.yaw);
  const sy = Math.sin(cam.yaw);
  const x1 = cy * x0 + sy * z0;
  const z1 = -sy * x0 + cy * z0;
  const cp = Math.cos(cam.pitch);
  const sp = Math.sin(cam.pitch);
  const y2 = cp * y0 - sp * z1;
  const z2 = sp * y0 + cp * z1;
  const depth = cam.distance + z2;
  if (depth <= cam.distance * 0.05) return null;
  const k = cam.distance / depth;
  return {x: w / 2 + x1 * cam.scale * k, y: h / 2 - y2 * cam.scale * k, k, depth};
};

const hexA = (hex: string, a: number) => {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
};

/** Canvas 3D scatter renderer: deterministic, fast enough for a few thousand points. */
export const Space3D: React.FC<{
  width: number;
  height: number;
  camera: Camera;
  layers: Layer[];
  style?: React.CSSProperties;
}> = ({width, height, camera, layers, style}) => {
  const ref = useRef<HTMLCanvasElement>(null);
  useLayoutEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext('2d')!;
    ctx.clearRect(0, 0, width, height);
    for (const L of layers) {
      if (L.kind === 'sphere') {
        const c = project(L.center, camera, width, height);
        if (!c) continue;
        const r = L.radius * camera.scale * c.k;
        const g = ctx.createRadialGradient(c.x - r * 0.3, c.y - r * 0.3, r * 0.1, c.x, c.y, r);
        const a = L.alpha ?? 1;
        g.addColorStop(0, hexA(L.color, 0.35 * a));
        g.addColorStop(0.85, hexA(L.color, 0.18 * a));
        g.addColorStop(1, hexA(L.color, 0.0));
        ctx.globalCompositeOperation = 'source-over';
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(c.x, c.y, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = hexA(L.color, 0.8 * a);
        ctx.lineWidth = 2.5;
        ctx.stroke();
        continue;
      }
      const proj = L.pts.map((p, i) => ({i, q: project(p, camera, width, height)}));
      proj.sort((a, b) => (b.q?.depth ?? 0) - (a.q?.depth ?? 0));
      ctx.globalCompositeOperation = 'lighter';
      for (const {i, q} of proj) {
        if (!q) continue;
        const a = typeof L.alpha === 'function' ? L.alpha(i) : (L.alpha ?? 1);
        if (a <= 0.003) continue;
        if (L.kind === 'points') {
          const col = typeof L.color === 'function' ? L.color(i) : L.color;
          const s = Math.max(0.6, L.size * q.k);
          ctx.fillStyle = hexA(col, Math.min(1, a * Math.min(1.2, q.k)));
          ctx.beginPath();
          ctx.arc(q.x, q.y, s, 0, Math.PI * 2);
          ctx.fill();
        } else {
          const r = Math.max(1, L.radius(i) * q.k);
          const g = ctx.createRadialGradient(q.x, q.y, 0, q.x, q.y, r * 3.2);
          g.addColorStop(0, hexA(L.core ?? '#ffffff', 0.95 * a));
          g.addColorStop(0.18, hexA(L.color, 0.75 * a));
          g.addColorStop(0.45, hexA(L.color, 0.18 * a));
          g.addColorStop(1, hexA(L.color, 0));
          ctx.fillStyle = g;
          ctx.beginPath();
          ctx.arc(q.x, q.y, r * 3.2, 0, Math.PI * 2);
          ctx.fill();
          ctx.globalCompositeOperation = 'source-over';
          ctx.fillStyle = hexA('#05030a', a);
          ctx.beginPath();
          ctx.arc(q.x, q.y, r * 0.42, 0, Math.PI * 2);
          ctx.fill();
          ctx.globalCompositeOperation = 'lighter';
        }
      }
    }
    ctx.globalCompositeOperation = 'source-over';
  });
  return <canvas ref={ref} width={width} height={height} style={style} />;
};

/** Interpolate a snapshot cloud at fractional frame index f. */
export const cloudAt = (frames: number[][][], f: number): V3[] => {
  const n = frames.length - 1;
  const ff = Math.max(0, Math.min(n, f));
  const a = Math.floor(ff);
  const b = Math.min(n, a + 1);
  const u = ff - a;
  const A = frames[a];
  const B = frames[b];
  return A.map((p, i) => [
    p[0] + (B[i][0] - p[0]) * u,
    p[1] + (B[i][1] - p[1]) * u,
    p[2] + (B[i][2] - p[2]) * u,
  ]);
};
