// One-off codemod: move visible JSX text and text="…" props of scene files into en.labels and
// replace them with t('scene.key'). Usage: node scripts/_extract_labels.mjs src/scenes/Foo.tsx …
// Writes the new labels to out/labels-<file>.json (merged into src/i18n/en.ts afterwards).
import fs from 'node:fs';
import path from 'node:path';
import ts from 'typescript';

// React's JSX whitespace rule (babel cleanJSXElementLiteralChild)
const jsxString = (raw) => {
  const lines = raw.split(/\r\n|\n|\r/);
  let last = 0;
  lines.forEach((l, i) => { if (/[^ \t]/.test(l)) last = i; });
  let out = '';
  lines.forEach((line, i) => {
    const isFirst = i === 0, isLast = i === lines.length - 1, isLastNonEmpty = i === last;
    let t = line.replace(/\t/g, ' ');
    if (!isFirst) t = t.replace(/^[ ]+/, '');
    if (!isLast) t = t.replace(/[ ]+$/, '');
    if (t) { if (!isLastNonEmpty) t += ' '; out += t; }
  });
  return out;
};

const slug = (s) =>
  s.normalize('NFKD').replace(/[^\w\s]/g, ' ').trim().toLowerCase().split(/\s+/).slice(0, 4).join('_') || 'x';

for (const file of process.argv.slice(2)) {
  const src = fs.readFileSync(file, 'utf8');
  const sf = ts.createSourceFile(file, src, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  const prefix = path.basename(file, '.tsx').toLowerCase();
  const labels = {};
  const edits = []; // {start, end, text}
  const needT = new Set(); // function body block nodes
  const keyFor = (text) => {
    const existing = Object.entries(labels).find(([, v]) => v === text);
    if (existing) return existing[0];
    let k = `${prefix}.${slug(text)}`, n = 2;
    while (labels[k] !== undefined) k = `${prefix}.${slug(text)}_${n++}`;
    labels[k] = text;
    return k;
  };
  const enclosingBlock = (node) => {
    for (let p = node.parent; p; p = p.parent) {
      if ((ts.isArrowFunction(p) || ts.isFunctionDeclaration(p) || ts.isFunctionExpression(p)) && p.body && ts.isBlock(p.body)) {
        // only component-level functions (inside a .map callback we still want the component)
        const name = ts.isVariableDeclaration(p.parent) ? p.parent.name.getText(sf) : '';
        if (/^[A-Z]/.test(name)) return p.body;
      }
    }
    return null;
  };
  const visit = (node) => {
    if (ts.isJsxText(node)) {
      const raw = node.getText(sf);
      const text = jsxString(raw);
      if (/[A-Za-zÀ-ÿΛ…?]/.test(text) && text.trim()) {
        const block = enclosingBlock(node);
        if (block) {
          // keep a separating space where React would have kept one
          const lead = /^\s*\S/.test(raw) && text.startsWith(' ') ? '' : '';
          edits.push({start: node.getStart(sf), end: node.getEnd(), text: `${lead}{tr('${keyFor(text)}')}`});
          needT.add(block);
        }
      }
    } else if (ts.isJsxAttribute(node) && ['text', 'xlabel', 'ylabel', 'title', 'sub', 'label'].includes(node.name.getText(sf)) && node.initializer && ts.isStringLiteral(node.initializer)) {
      const block = enclosingBlock(node);
      if (block) {
        edits.push({start: node.initializer.getStart(sf), end: node.initializer.getEnd(), text: `{tr('${keyFor(node.initializer.text)}')}`});
        needT.add(block);
      }
    }
    ts.forEachChild(node, visit);
  };
  visit(sf);
  for (const block of needT) {
    const has = block.statements.some((s) => s.getText(sf).includes('useT()'));
    if (!has) edits.push({start: block.getStart(sf) + 1, end: block.getStart(sf) + 1, text: '\n  const t = useT();'});
  }
  edits.sort((a, b) => b.start - a.start);
  let out = src;
  for (const e of edits) out = out.slice(0, e.start) + e.text + out.slice(e.end);
  if (needT.size && !/import \{[^}]*useT[^}]*\} from '\.\.\/i18n'/.test(out)) {
    if (/import \{([^}]*)\} from '\.\.\/i18n';/.test(out)) out = out.replace(/import \{([^}]*)\} from '\.\.\/i18n';/, (m, g) => `import {${g.trim()}, useT} from '../i18n';`);
    else out = out.replace(/(import [^\n]+\n)(?![\s\S]*^import )/m, `$1import {useT} from '../i18n';\n`);
  }
  fs.writeFileSync(file, out);
  fs.mkdirSync('out', {recursive: true});
  fs.writeFileSync(`out/labels-${prefix}.json`, JSON.stringify(labels, null, 1));
  console.log(`${file}: ${Object.keys(labels).length} labels, ${needT.size} components`);
}
