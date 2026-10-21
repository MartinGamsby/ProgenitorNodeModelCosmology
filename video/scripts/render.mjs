// Render one video and master its loudness:  node scripts/render.mjs <cut> <locale>
//   → out/<cut>-<locale>.mp4   (e.g. npm run render = full en)
import {execFileSync} from 'node:child_process';

const [cut = 'full', locale = 'en'] = process.argv.slice(2);
const id = `${cut}-${locale}`;
const raw = `out/${id}-raw.mp4`;
execFileSync('npx', ['remotion', 'render', id, raw], {stdio: 'inherit', shell: true});
execFileSync('ffmpeg', ['-y', '-hide_banner', '-loglevel', 'error', '-i', raw, '-c:v', 'copy', '-af', 'loudnorm=I=-14:TP=-1.5:LRA=11', '-c:a', 'aac', '-b:a', '192k', `out/${id}.mp4`], {stdio: 'inherit'});
console.log(`out/${id}.mp4`);
