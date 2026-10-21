// Caption markup shared by the video and scripts/voice.mjs (keep it dependency-free).
//   *word*          highlight in caption
//   {shown|spoken}  caption text vs. voice text

export type Token = {text: string; em: boolean};

export const spokenText = (s: string): string =>
  s.replace(/\{([^|}]*)\|([^}]*)\}/g, '$2').replace(/\*/g, '').replace(/’/g, "'").trim(); // TTS gets ASCII apostrophes

export const shownText = (s: string): string =>
  s.replace(/\{([^|}]*)\|([^}]*)\}/g, '$1').replace(/\*/g, '').trim();

/** Words of the caption with emphasis flags. */
export const captionTokens = (s: string): Token[] => {
  const shown = s.replace(/\{([^|}]*)\|([^}]*)\}/g, '$1');
  const out: Token[] = [];
  let em = false;
  for (const part of shown.split(/(\*)/)) {
    if (part === '*') {
      em = !em;
      continue;
    }
    part.split(/\s+/).forEach((w, i) => {
      if (!w) return;
      // glue punctuation-only fragments to the previous word: right after a closing * directly,
      // after a space (French "vérité :") with a no-break space so it never wraps alone
      const last = out[out.length - 1];
      if (/^[.,:;!?…)»]+$/.test(w) && last) last.text += (i > 0 ? ' ' : '') + w;
      // French elision before a highlight ("l’*énergie*") stays one word, highlighted
      else if (i === 0 && last && /[’']$/.test(last.text)) Object.assign(last, {text: last.text + w, em});
      else out.push({text: w, em});
    });
  }
  return out;
};

/** Stable short hash so voice clips regenerate only when their spoken text changes. */
export const hash = (s: string): string => {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0).toString(36);
};

export const clipKey = (sceneId: string, beat: number, text: string) =>
  `${sceneId}-${beat}-${hash(spokenText(text))}`;
