import React from 'react';
import {AbsoluteFill, Audio, interpolate, Sequence, staticFile, useCurrentFrame} from 'remotion';
import {BeatProvider} from './beats';
import {Starfield} from './components/Background';
import {SceneShell} from './components/SceneShell';
import {SCENE_COMPONENTS} from './scenes';
import {SCENES} from './script';
import {TIMELINE} from './timing';

const MUSIC_VOLUME = 0.16;
const MUSIC_DUCKED = 0.07;
const DUCK_RAMP = 18; // frames

/** Music ducks under the voice, swells in gaps and at the end. */
const SPEECH = TIMELINE.scenes.flatMap((s) =>
  s.beats.filter((b) => b.audio).map((b) => [s.from + b.from, s.from + b.from + b.dur - 8] as const),
);

const Music: React.FC = () => {
  return (
    <Audio
      src={staticFile('audio/music.wav')}
      loop
      volume={(f) => {
        // smooth ducking: distance (frames) to the nearest spoken line, eased over DUCK_RAMP
        let d = Infinity;
        for (const [a, b] of SPEECH) d = Math.min(d, f < a ? a - f : f > b ? f - b : 0);
        const duck = interpolate(d, [0, DUCK_RAMP], [MUSIC_DUCKED, MUSIC_VOLUME], {extrapolateRight: 'clamp'});
        const fadeIn = interpolate(f, [0, 45], [0, 1], {extrapolateRight: 'clamp'});
        const fadeOut = interpolate(f, [TIMELINE.total - 90, TIMELINE.total], [1, 0], {extrapolateLeft: 'clamp'});
        return duck * fadeIn * fadeOut;
      }}
    />
  );
};

/** Fade to black over the last ~1.6 s and hold black for the final 0.6 s. */
const EndFade: React.FC = () => {
  const frame = useCurrentFrame();
  const o = interpolate(frame, [TIMELINE.total - 48, TIMELINE.total - 18], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  return o > 0 ? <AbsoluteFill style={{background: '#000', opacity: o}} /> : null;
};

export const Explainer: React.FC = () => (
  <AbsoluteFill style={{background: '#04060d'}}>
    <Starfield count={0} />
    <Music />
    {TIMELINE.scenes.map((t, k) => {
      const def = SCENES[k];
      const Scene = SCENE_COMPONENTS[def.scene];
      if (!Scene) throw new Error(`Unknown scene component "${def.scene}" (src/scenes/index.ts)`);
      return (
        <Sequence key={t.id} from={t.from} durationInFrames={t.dur} name={t.id}>
          <BeatProvider timing={t} def={def}>
            <SceneShell def={def} timing={t}>
              <Scene />
            </SceneShell>
          </BeatProvider>
          {t.beats.map((b, i) =>
            b.audio ? (
              <Sequence key={i} from={b.from} durationInFrames={b.dur} name={`voice ${i}`}>
                <Audio src={staticFile(`voice/${b.audio}`)} />
              </Sequence>
            ) : null,
          )}
          {(def.sfx ?? []).map((s, i) => (
            <Sequence key={`sfx${i}`} from={t.beats[s.beat]?.from ?? 0} name={`sfx ${s.sound}`}>
              <Audio src={staticFile(`audio/${s.sound}.wav`)} volume={0.55} />
            </Sequence>
          ))}
          {def.whatIf && (
            <Sequence from={0} name="sfx whoosh">
              <Audio src={staticFile('audio/whoosh.wav')} volume={0.4} />
            </Sequence>
          )}
        </Sequence>
      );
    })}
    <EndFade />
  </AbsoluteFill>
);
