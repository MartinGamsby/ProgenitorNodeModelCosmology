import fs from 'node:fs';
// Side-by-side EN | FR review sheet for the translator: npm run i18n:review -> ../lode/tmp/fr-review.md
import {loadTs} from './load-ts.mjs';
const {LOCALES} = await loadTs('i18n/index');
const {en, fr} = LOCALES;
const esc = (s) => String(s).replace(/\|/g, '\|');
let md = `# Révision — traduction française (brouillon)\n\nColonne FR = sous-titre tel qu’affiché; entre crochets, ce que la voix prononce quand c’est différent.\nCorrige directement dans \`video/src/i18n/fr.ts\` (ou dis-moi les changements).\n\n`;
md += `## Titre\n| EN | FR |\n|---|---|\n| ${esc(en.title)} | ${esc(fr.title)} |\n| ${esc(en.subtitle)} | ${esc(fr.subtitle)} |\n\n`;
md += `## Chapitres et « Et si… »\n| EN | FR |\n|---|---|\n`;
for (const k of Object.keys(en.chapters)) md += `| ${esc(en.chapters[k])} | ${esc(fr.chapters[k])} |\n`;
for (const k of Object.keys(en.whatIf)) md += `| WHAT IF… ${esc(en.whatIf[k])} | ET SI… ${esc(fr.whatIf[k])} |\n`;
const shown = (s) => s.replace(/\{([^|}]*)\|([^}]*)\}/g, '$1');
const spoken = (s) => s.replace(/\{([^|}]*)\|([^}]*)\}/g, '$2').replace(/\*/g, '');
md += `\n## Narration\n`;
for (const [id, lines] of Object.entries(en.narration.full)) {
  md += `\n### ${id}\n| # | EN | FR |\n|---|---|---|\n`;
  lines.forEach((l, i) => {
    const f = fr.narration.full[id][i];
    const sp = spoken(f), sh = shown(f).replace(/\*/g, '');
    md += `| ${i} | ${esc(shown(l))} | ${esc(shown(f))}${sp !== sh ? ` [voix : ${esc(sp)}]` : ''} |\n`;
  });
}
md += `\n## Étiquettes à l’écran\n| clé | EN | FR |\n|---|---|---|\n`;
for (const k of Object.keys(en.labels)) md += `| ${k} | ${esc(en.labels[k])} | ${esc(fr.labels[k])} |\n`;
fs.mkdirSync('../lode/tmp', {recursive: true});
fs.writeFileSync('../lode/tmp/fr-review.md', md);
console.log(md.split('\n').length, 'lines');
