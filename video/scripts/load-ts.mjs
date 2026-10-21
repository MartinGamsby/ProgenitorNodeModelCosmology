// Load the video's TypeScript modules (script data, cuts, i18n, timing) from Node scripts without
// a TS runtime: transpile src/**/*.ts into node_modules/.cache/ts/, fix relative imports
// (adds .mjs, resolves folder/index), inline JSON imports, then import the requested module.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import ts from 'typescript';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SRC = path.join(ROOT, 'src');
const OUT = path.join(ROOT, 'node_modules', '.cache', 'ts');

const tsFiles = (dir) =>
  fs.readdirSync(dir, {withFileTypes: true}).flatMap((e) => {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) return e.name === 'scenes' || e.name === 'components' ? [] : tsFiles(p);
    return e.name.endsWith('.ts') && !e.name.endsWith('.d.ts') ? [p] : [];
  });

let built = false;
const build = () => {
  if (built) return;
  fs.rmSync(OUT, {recursive: true, force: true});
  for (const file of tsFiles(SRC)) {
    const dir = path.dirname(file);
    let js = ts.transpileModule(fs.readFileSync(file, 'utf8'), {
      compilerOptions: {module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022},
    }).outputText;
    js = js.replace(/import (\w+) from '(\.{1,2}\/[^']+\.json)';/g, (_, name, rel) =>
      `const ${name} = ${fs.readFileSync(path.resolve(dir, rel), 'utf8')};`,
    );
    js = js.replace(/from '(\.{1,2}\/[^']+)'/g, (m, rel) => {
      const abs = path.resolve(dir, rel);
      if (fs.existsSync(abs + '.ts')) return `from '${rel}.mjs'`;
      if (fs.existsSync(path.join(abs, 'index.ts'))) return `from '${rel}/index.mjs'`;
      return m;
    });
    const out = path.join(OUT, path.relative(SRC, file)).replace(/\.ts$/, '.mjs');
    fs.mkdirSync(path.dirname(out), {recursive: true});
    fs.writeFileSync(out, js);
  }
  built = true;
};

/** import('timing') → the transpiled src/timing.ts module. */
export const loadTs = async (mod) => {
  build();
  return import(pathToFileURL(path.join(OUT, `${mod}.mjs`)).href + `?t=${Date.now()}`);
};
