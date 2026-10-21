import React from 'react';
import {Composition, Still} from 'remotion';
import {CUTS} from './cuts';
import {LOCALE_IDS} from './i18n';
import {Explainer} from './Video';
import {Thumbnail} from './scenes/Thumbnail';
import {FPS} from './theme';
import {buildTimeline} from './timing';
import type {CutId} from './types';

/** One composition per cut x language: full-en, full-fr, idea-en, … (render: npx remotion render <id>). */
export const RemotionRoot: React.FC = () => (
  <>
    {(['full'] as CutId[]).flatMap((cut) =>
      LOCALE_IDS.map((locale) => {
        const vertical = CUTS[cut].format === 'vertical';
        return (
          <Composition
            key={`${cut}-${locale}`}
            id={`${cut}-${locale}`}
            component={Explainer}
            defaultProps={{cut, locale}}
            durationInFrames={buildTimeline(cut, locale).total}
            fps={FPS}
            width={vertical ? 1080 : 1920}
            height={vertical ? 1920 : 1080}
          />
        );
      }),
    )}
    <Still id="Thumbnail" component={Thumbnail} width={1280} height={720} />
  </>
);
