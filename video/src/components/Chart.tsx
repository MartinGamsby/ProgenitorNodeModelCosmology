import React from 'react';
import {C, FONT} from '../theme';

export type Box = {x: number; y: number; w: number; h: number};

export const scales = (box: Box, xd: [number, number], yd: [number, number], logX = false) => {
  const fx = (v: number) => (logX ? Math.log10(v) : v);
  const X = (v: number) => box.x + ((fx(v) - fx(xd[0])) / (fx(xd[1]) - fx(xd[0]))) * box.w;
  const Y = (v: number) => box.y + box.h - ((v - yd[0]) / (yd[1] - yd[0])) * box.h;
  return {X, Y};
};

export const Axes: React.FC<{
  box: Box;
  xd: [number, number];
  yd: [number, number];
  logX?: boolean;
  xticks?: number[];
  yticks?: number[];
  xlabel?: string;
  ylabel?: string;
  fmtX?: (v: number) => string;
  fmtY?: (v: number) => string;
  opacity?: number;
  grid?: boolean;
}> = ({box, xd, yd, logX, xticks = [], yticks = [], xlabel, ylabel, fmtX = String, fmtY = String, opacity = 1, grid = true}) => {
  const {X, Y} = scales(box, xd, yd, logX);
  return (
    <g opacity={opacity} fontFamily={FONT.body}>
      {grid &&
        yticks.map((t) => <line key={`gy${t}`} x1={box.x} x2={box.x + box.w} y1={Y(t)} y2={Y(t)} stroke={C.faint} strokeWidth={1} opacity={0.5} />)}
      <line x1={box.x} y1={box.y + box.h} x2={box.x + box.w} y2={box.y + box.h} stroke={C.dim} strokeWidth={2.5} />
      <line x1={box.x} y1={box.y} x2={box.x} y2={box.y + box.h} stroke={C.dim} strokeWidth={2.5} />
      {xticks.map((t) => (
        <g key={`x${t}`}>
          <line x1={X(t)} x2={X(t)} y1={box.y + box.h} y2={box.y + box.h + 10} stroke={C.dim} strokeWidth={2} />
          <text x={X(t)} y={box.y + box.h + 42} fill={C.dim} fontSize={26} textAnchor="middle" fontWeight={600}>
            {fmtX(t)}
          </text>
        </g>
      ))}
      {yticks.map((t) => (
        <g key={`y${t}`}>
          <line x1={box.x - 10} x2={box.x} y1={Y(t)} y2={Y(t)} stroke={C.dim} strokeWidth={2} />
          <text x={box.x - 18} y={Y(t) + 9} fill={C.dim} fontSize={26} textAnchor="end" fontWeight={600}>
            {fmtY(t)}
          </text>
        </g>
      ))}
      {xlabel && (
        <text x={box.x + box.w / 2} y={box.y + box.h + 88} fill={C.ink} fontSize={30} textAnchor="middle" fontWeight={700}>
          {xlabel}
        </text>
      )}
      {ylabel && (
        <text
          x={box.x - 100}
          y={box.y + box.h / 2}
          fill={C.ink}
          fontSize={30}
          textAnchor="middle"
          fontWeight={700}
          transform={`rotate(-90 ${box.x - 100} ${box.y + box.h / 2})`}
        >
          {ylabel}
        </text>
      )}
    </g>
  );
};

/** Coloured legend chip + text. */
export const LegendItem: React.FC<{x: number; y: number; color: string; text: string; dashed?: boolean; opacity?: number}> = ({
  x,
  y,
  color,
  text,
  dashed,
  opacity = 1,
}) => (
  <g opacity={opacity} fontFamily={FONT.body}>
    <line x1={x} x2={x + 44} y1={y} y2={y} stroke={color} strokeWidth={6} strokeDasharray={dashed ? '10 8' : undefined} strokeLinecap="round" />
    <text x={x + 58} y={y + 10} fill={C.ink} fontSize={28} fontWeight={700}>
      {text}
    </text>
  </g>
);
