import React from 'react';
import {AbsoluteFill} from 'remotion';
import {useBeats} from '../../beats';
import {Show} from '../../components/Show';
import {Arrow, At, Card, Label, NodeDot, SvgDefs, Tex, Title} from '../../components/ui';
import {C, FONT, HEIGHT, WIDTH} from '../../theme';

export const Numbers: React.FC = () => {
  const b = useBeats();
  const px = 13; // px per Gpc
  const x0 = 330;
  const y0 = 770;
  const S = 39;
  const R = 14;
  const ruler = b.p(2, 40);
  return (
    <AbsoluteFill>
      <At x={WIDTH / 2} y={230}>
        <Card style={{display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
          <Tex tex={String.raw`H_0^2\,\Omega_\Lambda\approx\frac{2GM}{S^3}\;\Rightarrow\;S\approx\left(\frac{2GM}{H_0^2\,\Omega_\Lambda}\right)^{1/3}`} size={58} />
        </Card>
      </At>
      <Show from={1}>
        <At x={WIDTH / 2 - 330} y={450}>
          <Card accent={C.node} style={{textAlign: 'center'}}>
            <Label size={28} color={C.node} style={{fontFamily: FONT.display, fontWeight: 800}}>
              NODE MASS
            </Label>
            <Title size={64}>
              M ≈ {Math.round(500 * b.p(1, 40))} × M<sub style={{fontSize: 30}}>obs</sub>
            </Title>
            <Label size={28}>≈ 5 × 10⁵⁵ kg</Label>
          </Card>
        </At>
      </Show>
      <Show from={2}>
        <At x={WIDTH / 2 + 330} y={450}>
          <Card accent={C.gold} style={{textAlign: 'center'}}>
            <Label size={28} color={C.gold} style={{fontFamily: FONT.display, fontWeight: 800}}>
              SPACING
            </Label>
            <Title size={64}>S ≈ {Math.round(39 * b.p(2, 40))} Gpc</Title>
            <Label size={28}>≈ 130 billion light-years</Label>
          </Card>
        </At>
        <svg width={WIDTH} height={HEIGHT} style={{position: 'absolute'}}>
          <SvgDefs />
          <circle cx={x0} cy={y0} r={R * px} fill="url(#bubble)" stroke={C.bubble} strokeWidth={3} />
          <NodeDot x={x0 + S * px} y={y0} r={30} />
          <text x={x0 + (R * px + S * px) / 2} y={y0 - 22} fill={C.gold} fontFamily={FONT.body} fontWeight={800} fontSize={30} textAnchor="middle" opacity={ruler}>
            39 Gpc
          </text>
          <text x={x0} y={y0 + 8} fill={C.bubble} fontFamily={FONT.body} fontWeight={700} fontSize={26} textAnchor="middle">
            us (R ≈ 14 Gpc)
          </text>
          <text x={x0 + S * px} y={y0 - 70} fill={C.node} fontFamily={FONT.body} fontWeight={700} fontSize={26} textAnchor="middle">
            nearest node
          </text>
          <Arrow x1={x0 + R * px + 20} y1={y0} x2={x0 + S * px - 80} y2={y0} color={C.node} width={5} head={18} p={ruler} opacity={0.7} />
        </svg>
      </Show>
      <Show from={3}>
        <At x={1560} y={780}>
          <Card accent={C.eds} style={{textAlign: 'center'}}>
            <Title size={48}>
              <span style={{textDecoration: 'line-through', color: C.dim}}>vacuum energy</span>
            </Title>
            <Title size={56} color={C.eds}>
              just gravity
            </Title>
          </Card>
        </At>
      </Show>
    </AbsoluteFill>
  );
};
