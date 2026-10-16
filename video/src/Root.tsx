import React from 'react';
import {Composition, Still} from 'remotion';
import {Explainer} from './Video';
import {Thumbnail} from './scenes/Thumbnail';
import {FPS, HEIGHT, WIDTH} from './theme';
import {TIMELINE} from './timing';

export const RemotionRoot: React.FC = () => (
  <>
    <Composition id="Explainer" component={Explainer} durationInFrames={TIMELINE.total} fps={FPS} width={WIDTH} height={HEIGHT} />
    <Still id="Thumbnail" component={Thumbnail} width={1280} height={720} />
  </>
);
