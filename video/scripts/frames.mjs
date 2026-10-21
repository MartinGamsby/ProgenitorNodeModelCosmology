// Print the frame of a scene/beat (for stills).  node scripts/frames.mjs [sceneId] [beat] [offsetFrames]
// With no args: lists every scene with its start frame, length and beats.
// CUT / LOCALE env pick the video (default full / en).
import {loadTs} from './load-ts.mjs';

const CUT = process.env.CUT || 'full';
const LOCALE = process.env.LOCALE || 'en';
const {buildTimeline} = await loadTs('timing');
const TL = buildTimeline(CUT, LOCALE);
const [id, beat, off] = process.argv.slice(2);
if (!id) {
  for (const s of TL.scenes) console.log(`${s.id.padEnd(18)} from ${String(s.from).padStart(6)}  dur ${String(s.dur).padStart(5)}  beats ${s.beats.map((b) => s.from + b.from).join(',')}`);
  console.log(`total ${TL.total} frames = ${(TL.total / 30 / 60).toFixed(2)} min (${CUT}-${LOCALE})`);
} else {
  const s = TL.scenes.find((x) => x.id === id);
  if (!s) throw new Error(`no scene ${id}`);
  const b = beat === undefined ? 0 : s.beats[Number(beat)].from;
  console.log(s.from + b + Number(off ?? 0));
}
