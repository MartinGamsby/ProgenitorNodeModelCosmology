// Generate one voice clip per narration beat from src/script.ts.
//   npm run voice                      (Windows SAPI, offline, default)
//   VOICE_ENGINE=chatterbox npm run voice   (neural, expressive; CB_EXAGGERATION, CB_CFG, CB_PROMPT)
//   VOICE_ENGINE=kokoro npm run voice       (neural, fast; KOKORO_VOICE, KOKORO_SPEED)
//   VOICE_ENGINE=piper PIPER_MODEL=path/to/voice.onnx npm run voice
// chatterbox/kokoro run scripts/tts_worker.py inside video/.tts-venv (see README).
// Clip files carry a hash of engine + settings, so switching engines never mixes voices and an
// interrupted neural run resumes where it stopped.
//   VOICE_ENGINE=none npm run voice    (no audio; timing estimated from word count)
// Only clips whose spoken text changed are regenerated. Writes public/voice/*.wav and
// src/generated/voice.json (read by the video to time every beat).
import {execFileSync, spawnSync} from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import ts from 'typescript';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ENGINE = process.env.VOICE_ENGINE || 'sapi';
const SAPI_VOICE = process.env.SAPI_VOICE || 'Microsoft David Desktop';
const SAPI_RATE = process.env.SAPI_RATE || '1';
const ENV_KEYS = {
  sapi: ['SAPI_VOICE', 'SAPI_RATE'],
  kokoro: ['KOKORO_VOICE', 'KOKORO_SPEED'],
  chatterbox: ['CB_EXAGGERATION', 'CB_CFG', 'CB_PROMPT'],
  piper: ['PIPER_MODEL'],
  none: [],
};
const SIG = [ENGINE, ...(ENV_KEYS[ENGINE] ?? []).map((k) => `${k}=${process.env[k] ?? ''}`)].join(';');
const VENV_PY = path.join(ROOT, '.tts-venv', 'Scripts', 'python.exe');
const VOICE_DIR = path.join(ROOT, 'public', 'voice');
const MANIFEST = path.join(ROOT, 'src', 'generated', 'voice.json');

// --- load src/script.ts + src/markup.ts without a TS runtime --------------------------------
const cache = path.join(ROOT, 'node_modules', '.cache', 'voice');
fs.mkdirSync(cache, {recursive: true});
for (const f of ['script', 'markup', 'types']) {
  const src = fs.readFileSync(path.join(ROOT, 'src', `${f}.ts`), 'utf8');
  let js = ts.transpileModule(src, {
    compilerOptions: {module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022},
  }).outputText;
  js = js.replace(/from '\.\/(\w+)'/g, "from './$1.mjs'");
  fs.writeFileSync(path.join(cache, `${f}.mjs`), js);
}
const {SCENES} = await import(pathToFileURL(path.join(cache, 'script.mjs')).href + `?t=${Date.now()}`);
const {spokenText, clipKey, hash} = await import(pathToFileURL(path.join(cache, 'markup.mjs')).href);

const wavSeconds = (file) => {
  const b = fs.readFileSync(file);
  const byteRate = b.readUInt32LE(28);
  let off = 12;
  while (off < b.length - 8) {
    const id = b.toString('ascii', off, off + 4);
    const size = b.readUInt32LE(off + 4);
    if (id === 'data') return size / byteRate;
    off += 8 + size;
  }
  return (b.length - 44) / byteRate;
};

const clips = {};
const jobs = [];
for (const s of SCENES) {
  s.beats.forEach((text, i) => {
    const spoken = spokenText(text);
    if (!spoken) return;
    const key = clipKey(s.id, i, text);
    const file = `${key}-${hash(SIG)}.wav`;
    const abs = path.join(VOICE_DIR, file);
    if (ENGINE !== 'none' && fs.existsSync(abs)) {
      clips[key] = {file, seconds: wavSeconds(abs), text: spoken};
    } else if (ENGINE !== 'none') {
      jobs.push({key, text: spoken, out: abs, file});
    }
  });
}

fs.mkdirSync(VOICE_DIR, {recursive: true});
if (jobs.length && ENGINE === 'sapi') {
  const jobFile = path.join(cache, 'jobs.json');
  fs.writeFileSync(jobFile, JSON.stringify(jobs.map(({text, out}) => ({text, out}))));
  const ps = `
Add-Type -AssemblyName System.Speech
$jobs = Get-Content -Raw -Encoding UTF8 '${jobFile}' | ConvertFrom-Json
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice('${SAPI_VOICE}')
$s.Rate = ${SAPI_RATE}
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(24000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
foreach ($j in $jobs) { $s.SetOutputToWaveFile($j.out, $fmt); $s.Speak($j.text); $s.SetOutputToNull() }
$s.Dispose()`;
  execFileSync('powershell', ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', ps], {stdio: 'inherit'});
} else if (jobs.length && (ENGINE === 'kokoro' || ENGINE === 'chatterbox')) {
  const jobFile = path.join(cache, 'jobs.json');
  fs.writeFileSync(jobFile, JSON.stringify(jobs.map(({text, out}) => ({text, out}))));
  execFileSync(VENV_PY, [path.join(ROOT, 'scripts', 'tts_worker.py'), '--engine', ENGINE, '--jobs', jobFile], {stdio: 'inherit'});
} else if (jobs.length && ENGINE === 'piper') {
  const model = process.env.PIPER_MODEL;
  if (!model) throw new Error('Set PIPER_MODEL to a piper .onnx voice');
  for (const j of jobs) {
    const r = spawnSync(process.env.PIPER_BIN || 'piper', ['-m', model, '-f', j.out], {input: j.text});
    if (r.status !== 0) throw new Error(`piper failed: ${r.stderr}`);
  }
}
for (const j of jobs) clips[j.key] = {file: j.file, seconds: wavSeconds(j.out), text: j.text};

// Drop clips no beat uses any more.
const keep = new Set(Object.values(clips).map((c) => c.file));
for (const f of fs.readdirSync(VOICE_DIR)) if (f.endsWith('.wav') && !keep.has(f)) fs.unlinkSync(path.join(VOICE_DIR, f));

fs.mkdirSync(path.dirname(MANIFEST), {recursive: true});
fs.writeFileSync(MANIFEST, JSON.stringify({engine: SIG, clips}, null, 1));
const total = Object.values(clips).reduce((a, c) => a + c.seconds, 0);
console.log(`voice: ${jobs.length} generated, ${Object.keys(clips).length} clips, ${total.toFixed(1)} s of speech (${ENGINE})`);
