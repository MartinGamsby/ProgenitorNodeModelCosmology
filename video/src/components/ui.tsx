import katex from 'katex';
import 'katex/dist/katex.min.css';
import './tex.css';
import React from 'react';
import {C, FONT} from '../theme';

/** KaTeX equation. */
export const Tex: React.FC<{tex: string; size?: number; color?: string; style?: React.CSSProperties}> = ({
  tex,
  size = 64,
  color = C.ink,
  style,
}) => (
  <div
    className="tex"
    style={{fontSize: size, color, whiteSpace: 'nowrap', ...style}}
    dangerouslySetInnerHTML={{__html: katex.renderToString(tex, {throwOnError: false, displayMode: true})}}
  />
);

export const Title: React.FC<{children: React.ReactNode; size?: number; color?: string; style?: React.CSSProperties}> = ({
  children,
  size = 80,
  color = C.ink,
  style,
}) => (
  <div style={{fontFamily: FONT.display, fontWeight: 900, fontSize: size, color, lineHeight: 1.05, letterSpacing: -1, ...style}}>
    {children}
  </div>
);

export const Label: React.FC<{children: React.ReactNode; size?: number; color?: string; style?: React.CSSProperties}> = ({
  children,
  size = 32,
  color = C.dim,
  style,
}) => <div style={{fontFamily: FONT.body, fontWeight: 600, fontSize: size, color, ...style}}>{children}</div>;

/** Absolutely positioned element centred on (x, y). */
export const At: React.FC<{x: number; y: number; children: React.ReactNode; style?: React.CSSProperties}> = ({x, y, children, style}) => (
  <div style={{position: 'absolute', left: x, top: y, transform: 'translate(-50%, -50%)', width: 'max-content', ...style}}>{children}</div>
);

export const Card: React.FC<{children: React.ReactNode; style?: React.CSSProperties; accent?: string}> = ({children, style, accent = C.faint}) => (
  <div
    style={{
      background: 'rgba(10,16,34,0.82)',
      border: `2px solid ${accent}`,
      borderRadius: 22,
      padding: '28px 36px',
      boxShadow: '0 20px 60px rgba(0,0,0,0.45)',
      ...style,
    }}
  >
    {children}
  </div>
);

/** SVG arrow from (x1,y1) to (x2,y2); `p` draws it progressively. */
export const Arrow: React.FC<{
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  color?: string;
  width?: number;
  p?: number;
  head?: number;
  opacity?: number;
}> = ({x1, y1, x2, y2, color = C.ink, width = 6, p = 1, head = 18, opacity = 1}) => {
  if (p <= 0.001) return null;
  const x = x1 + (x2 - x1) * p;
  const y = y1 + (y2 - y1) * p;
  const ang = Math.atan2(y - y1, x - x1);
  const len = Math.hypot(x - x1, y - y1);
  const h = Math.min(head, len * 0.6);
  const bx = x - Math.cos(ang) * h * 0.8;
  const by = y - Math.sin(ang) * h * 0.8;
  return (
    <g opacity={opacity}>
      <line x1={x1} y1={y1} x2={bx} y2={by} stroke={color} strokeWidth={width} strokeLinecap="round" />
      <polygon
        points={`${x},${y} ${x - Math.cos(ang - 0.45) * h},${y - Math.sin(ang - 0.45) * h} ${x - Math.cos(ang + 0.45) * h},${y - Math.sin(ang + 0.45) * h}`}
        fill={color}
      />
    </g>
  );
};

/** Little spiral-ish galaxy glyph. */
export const Galaxy: React.FC<{x: number; y: number; s: number; rot: number; hue: string; o: number}> = ({x, y, s, rot, hue, o}) => (
  <g transform={`translate(${x},${y}) rotate(${rot}) scale(${s})`} opacity={o}>
    <ellipse rx={14} ry={5} fill={hue} opacity={0.35} />
    <ellipse rx={7} ry={2.6} fill={hue} opacity={0.7} />
    <circle r={2.2} fill="#fff" />
  </g>
);

/** Glowing HMEA "dark giant" icon for 2D diagrams. */
export const NodeDot: React.FC<{x: number; y: number; r: number; opacity?: number; color?: string}> = ({
  x,
  y,
  r,
  opacity = 1,
  color = C.node,
}) => (
  <g opacity={opacity}>
    <circle cx={x} cy={y} r={r * 3} fill={`url(#nodeglow)`} />
    <circle cx={x} cy={y} r={r} fill="#06030c" stroke={color} strokeWidth={3} />
  </g>
);

export const SvgDefs: React.FC = () => (
  <defs>
    <radialGradient id="nodeglow">
      <stop offset="0%" stopColor={C.node} stopOpacity={0.9} />
      <stop offset="35%" stopColor={C.nodeGlow} stopOpacity={0.35} />
      <stop offset="100%" stopColor={C.nodeGlow} stopOpacity={0} />
    </radialGradient>
    <radialGradient id="bubble">
      <stop offset="0%" stopColor={C.bubble} stopOpacity={0.35} />
      <stop offset="85%" stopColor={C.bubble} stopOpacity={0.15} />
      <stop offset="100%" stopColor={C.bubble} stopOpacity={0} />
    </radialGradient>
    <filter id="soft" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="6" />
    </filter>
  </defs>
);

/** Counts a number up with fixed decimals. */
export const Count: React.FC<{value: number; p: number; decimals?: number; style?: React.CSSProperties}> = ({value, p, decimals = 0, style}) => (
  <span style={style}>{(value * p).toFixed(decimals)}</span>
);

/** On-screen distances are in light-years (the data is in Gpc). */
export const gly = (gpc: number) => `${Math.round(gpc * 3.2616)} billion light-years`;

/** Path string from points. */
export const pathOf = (pts: [number, number][]) =>
  pts.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ');

/** Truncate a polyline to fraction p of its points (for draw-on animations). */
export const partial = <T,>(pts: T[], p: number) => pts.slice(0, Math.max(2, Math.round(pts.length * Math.max(0, Math.min(1, p)))));
