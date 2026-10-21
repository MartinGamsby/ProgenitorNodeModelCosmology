// List French (or LOCALE=xx) text that is missing or doesn't match English: npm run i18n:check
import {loadTs} from './load-ts.mjs';

const target = process.env.LOCALE || 'fr';
const {LOCALES} = await loadTs('i18n/index');
const {SCENE_SPECS} = await loadTs('content/scenes');
const en = LOCALES.en;
const L = LOCALES[target];
const problems = [];
for (const [cut, scenes] of Object.entries(en.narration)) {
  for (const [id, lines] of Object.entries(scenes)) {
    const tl = L.narration[cut]?.[id];
    if (!tl) problems.push(`narration.${cut}.${id}: missing (${lines.length} lines)`);
    else if (tl.length !== lines.length) problems.push(`narration.${cut}.${id}: ${tl.length} lines, English has ${lines.length}`);
  }
}
for (const part of ['labels', 'chapters', 'whatIf', 'credits']) {
  for (const k of Object.keys(en[part])) if (!(k in L[part])) problems.push(`${part}.${k}: missing`);
}
for (const s of SCENE_SPECS) if (!en.narration.full[s.id]) problems.push(`en narration.full.${s.id}: missing`);
console.log(problems.length ? problems.join('\n') : `${target}: complete`);
console.log(`${problems.length} issue(s)`);
