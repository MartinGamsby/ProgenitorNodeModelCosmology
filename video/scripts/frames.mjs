// Print the frame of a scene/beat (for `remotion still`).  node scripts/frames.mjs [sceneId] [beat] [offsetFrames]
// With no args: lists every scene with its start frame, length and beats.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import ts from 'typescript';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const cache = path.join(ROOT, 'node_modules', '.cache', 'frames');
fs.mkdirSync(cache, {recursive: true});
for (const f of ['script', 'markup', 'types', 'constants', 'timing']) {
  let js = ts.transpileModule(fs.readFileSync(path.join(ROOT, 'src', `${f}.ts`), 'utf8'), {
    compilerOptions: {module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022},
  }).outputText;
  js = js.replace(/from '\.\/(\w+)'/g, "from './$1.mjs'");
  js = js.replace(/import voice from '\.\/generated\/voice\.json';/, `const voice = ${fs.readFileSync(path.join(ROOT, 'src/generated/voice.json'), 'utf8')};`);
  fs.writeFileSync(path.join(cache, `${f}.mjs`), js);
}
const {TIMELINE} = await import(pathToFileURL(path.join(cache, 'timing.mjs')).href + `?t=${Date.now()}`);
const [id, beat, off] = process.argv.slice(2);
if (!id) {
  for (const s of TIMELINE.scenes) console.log(`${s.id.padEnd(18)} from ${String(s.from).padStart(6)}  dur ${String(s.dur).padStart(5)}  beats ${s.beats.map((b) => s.from + b.from).join(',')}`);
  console.log(`total ${TIMELINE.total} frames = ${(TIMELINE.total / 30 / 60).toFixed(2)} min`);
} else {
  const s = TIMELINE.scenes.find((x) => x.id === id);
  if (!s) throw new Error(`no scene ${id}`);
  const b = beat === undefined ? 0 : s.beats[Number(beat)].from;
  console.log(s.from + b + Number(off ?? 0));
}
