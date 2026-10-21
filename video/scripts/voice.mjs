// Generate one voice clip per narration line, for one language, covering every cut (full, idea,
// short): a line shared by two cuts is voiced once. Text: src/i18n/<locale>.ts.
//   LOCALE=en VOICE_ENGINE=chatterbox npm run voice   (neural; CB_EXAGGERATION, CB_CFG, CB_PROMPT)
//   LOCALE=fr VOICE_ENGINE=chatterbox CB_LANGUAGE=fr …  (Chatterbox Multilingual)
//   VOICE_ENGINE=kokoro (KOKORO_VOICE, KOKORO_SPEED) · piper (PIPER_MODEL) · sapi (default) · none
// chatterbox/kokoro run scripts/tts_worker.py inside video/.tts-venv (see README).
// Clip files carry a hash of engine + settings, so switching engines never mixes voices and an
// interrupted neural run resumes where it stopped. Only changed lines are regenerated.
// Writes public/voice/<locale>/*.wav and src/generated/voice.<locale>.json (beat timing).
import {execFileSync, spawnSync} from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {loadTs} from './load-ts.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const LOCALE = process.env.LOCALE || 'en';
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
// CB_LANGUAGE joins the signature only when set, so existing English clips keep their names
const SIG = [ENGINE, ...(ENV_KEYS[ENGINE] ?? []).map((k) => `${k}=${process.env[k] ?? ''}`), ...(process.env.CB_LANGUAGE ? [`CB_LANGUAGE=${process.env.CB_LANGUAGE}`] : [])].join(';');
const VENV_PY = path.join(ROOT, '.tts-venv', 'Scripts', 'python.exe');
const VOICE_DIR = path.join(ROOT, 'public', 'voice', LOCALE);
const MANIFEST = path.join(ROOT, 'src', 'generated', `voice.${LOCALE}.json`);
const cache = path.join(ROOT, 'node_modules', '.cache', 'voice');
fs.mkdirSync(cache, {recursive: true});

const {resolveScenes} = await loadTs('timing');
const {CUTS} = await loadTs('cuts');
const {spokenText, clipKey, hash} = await loadTs('markup');
// every scene of every cut in this language (a scene used by several cuts is listed once per text)
const SCENES = Object.keys(CUTS).flatMap((cut) => resolveScenes(cut, LOCALE).map((s) => ({...s, skip: new Set(CUTS[cut].skip?.[s.id] ?? [])})));

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
    if (!spoken || s.skip.has(i)) return;
    const key = clipKey(s.id, i, text);
    if (clips[key] || jobs.some((j) => j.key === key)) return; // already listed by another cut
    const ov = s.voice?.[i] ?? {}; // per-line overrides (exaggeration / cfg) for neural engines
    const file = `${key}-${hash(SIG + JSON.stringify(ov))}.wav`;
    const abs = path.join(VOICE_DIR, file);
    const extra = {exag: ov.exaggeration, cfg: ov.cfg};
    if (ENGINE !== 'none' && fs.existsSync(abs)) {
      clips[key] = {file, seconds: wavSeconds(abs), text: spoken, ...extra};
    } else if (ENGINE !== 'none') {
      jobs.push({key, text: spoken, out: abs, file, ...extra});
    }
  });
}

if (process.env.VOICE_DRY) {
  console.log(`dry run (${LOCALE}): ${Object.keys(clips).length} clips present, ${jobs.length} to generate`);
  for (const j of jobs.slice(0, 10)) console.log('  would voice:', j.key, '|', j.text.slice(0, 60));
  process.exit(0);
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
  fs.writeFileSync(jobFile, JSON.stringify(jobs.map(({text, out, exag, cfg}) => ({text, out, exag, cfg}))));
  execFileSync(VENV_PY, [path.join(ROOT, 'scripts', 'tts_worker.py'), '--engine', ENGINE, '--jobs', jobFile], {stdio: 'inherit'});
} else if (jobs.length && ENGINE === 'piper') {
  const model = process.env.PIPER_MODEL;
  if (!model) throw new Error('Set PIPER_MODEL to a piper .onnx voice');
  for (const j of jobs) {
    const r = spawnSync(process.env.PIPER_BIN || 'piper', ['-m', model, '-f', j.out], {input: j.text});
    if (r.status !== 0) throw new Error(`piper failed: ${r.stderr}`);
  }
}
for (const j of jobs) clips[j.key] = {file: j.file, seconds: wavSeconds(j.out), text: j.text, exag: j.exag, cfg: j.cfg};

// Drop clips no beat uses any more.
const keep = new Set(Object.values(clips).map((c) => c.file));
for (const f of fs.readdirSync(VOICE_DIR)) if (f.endsWith('.wav') && !keep.has(f)) fs.unlinkSync(path.join(VOICE_DIR, f));

fs.mkdirSync(path.dirname(MANIFEST), {recursive: true});
fs.writeFileSync(MANIFEST, JSON.stringify({engine: SIG, clips}, null, 1));
const total = Object.values(clips).reduce((a, c) => a + c.seconds, 0);
console.log(`voice: ${jobs.length} generated, ${Object.keys(clips).length} clips, ${total.toFixed(1)} s of speech (${ENGINE})`);
