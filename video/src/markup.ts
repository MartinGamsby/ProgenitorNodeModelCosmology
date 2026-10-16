// Caption markup shared by the video and scripts/voice.mjs (keep it dependency-free).
//   *word*          highlight in caption
//   {shown|spoken}  caption text vs. voice text

export type Token = {text: string; em: boolean};

export const spokenText = (s: string): string =>
  s.replace(/\{([^|}]*)\|([^}]*)\}/g, '$2').replace(/\*/g, '').trim();

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
    for (const w of part.split(/\s+/)) {
      if (!w) continue;
      // glue punctuation-only fragments (after a closing *) to the previous word
      if (/^[.,:;!?…)]+$/.test(w) && out.length) out[out.length - 1].text += w;
      else out.push({text: w, em});
    }
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
