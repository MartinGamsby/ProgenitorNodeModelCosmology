// Render review stills with one bundle:  node scripts/stills.mjs 120 450 2300 ...
// Frames may also be "sceneId:beat[:offset]". Output: out/stills/<frame>.jpg at half size.
import {bundle} from '@remotion/bundler';
import {renderStill, selectComposition} from '@remotion/renderer';
import {execFileSync} from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const frames = process.argv.slice(2).map((a) =>
  /^\d+$/.test(a) ? Number(a) : Number(execFileSync('node', [path.join(ROOT, 'scripts/frames.mjs'), ...a.split(':')]).toString().trim()),
);
const serveUrl = await bundle({entryPoint: path.join(ROOT, 'src/index.ts')});
const composition = await selectComposition({serveUrl, id: 'Explainer'});
fs.mkdirSync(path.join(ROOT, 'out/stills'), {recursive: true});
for (const frame of frames) {
  const output = path.join(ROOT, 'out/stills', `${frame}.jpg`);
  await renderStill({composition, serveUrl, frame, output, imageFormat: 'jpeg', jpegQuality: 80, scale: 0.5});
  console.log(output);
}

// contact sheets of 6, in argument order: out/sheet-1.jpg, out/sheet-2.jpg, ...
const outs = frames.map((fr) => path.join(ROOT, 'out/stills', `${fr}.jpg`));
for (let i = 0; i * 6 < outs.length; i++) {
  execFileSync('python', [path.join(ROOT, 'scripts/sheet.py'), path.join(ROOT, 'out', `sheet-${i + 1}.jpg`), ...outs.slice(i * 6, i * 6 + 6)]);
}
console.log(`${Math.ceil(outs.length / 6)} sheet(s) in out/`);
