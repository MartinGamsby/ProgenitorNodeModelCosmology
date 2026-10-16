import React from 'react';
import {AbsoluteFill} from 'remotion';
import {Starfield} from '../components/Background';
import {Space3D} from '../components/Space3D';
import {Title} from '../components/ui';
import medium from '../../public/data/medium.json';
import {C, FONT} from '../theme';

/** 1280x720 YouTube thumbnail:  npx remotion still Thumbnail out/thumbnail.png */
export const Thumbnail: React.FC = () => (
  <AbsoluteFill style={{background: C.bg}}>
    <Starfield count={120} />
    <Space3D
      width={1280}
      height={720}
      camera={{yaw: 0.9, pitch: 0.3, distance: 900, scale: 2.2, target: [-125, 10, 0]}}
      layers={[
        {kind: 'glows', pts: medium.pos as [number, number, number][], radius: (i) => 10 + 9 * Math.cbrt((medium.mass as number[])[i]), color: C.nodeGlow, core: C.node, alpha: 0.9},
        {kind: 'sphere', center: [0, 0, 0], radius: 34, color: C.bubble},
      ]}
      style={{position: 'absolute'}}
    />
    <AbsoluteFill style={{background: 'linear-gradient(90deg, rgba(4,6,13,0.95) 0%, rgba(4,6,13,0.6) 55%, rgba(4,6,13,0) 100%)'}} />
    <AbsoluteFill style={{padding: 60, justifyContent: 'center'}}>
      <Title size={92} style={{maxWidth: 760}}>
        DARK ENERGY
      </Title>
      <Title size={92} color={C.gold} style={{maxWidth: 760}}>
        = THE NEIGHBOURS?
      </Title>
      <div style={{marginTop: 28, background: C.gold, color: C.bg, fontWeight: 900, fontSize: 40, padding: '6px 20px', alignSelf: 'flex-start', transform: 'rotate(-3deg)', fontFamily: FONT.display}}>
        WHAT IF…
      </div>
    </AbsoluteFill>
  </AbsoluteFill>
);
