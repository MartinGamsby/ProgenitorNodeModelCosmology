import React, {createContext, useContext} from 'react';
import type {CutId, Locale, LocaleId} from '../types';
import {en} from './en';
import {fr} from './fr';

export const LOCALES: Record<LocaleId, Locale> = {en, fr};
export const LOCALE_IDS = Object.keys(LOCALES) as LocaleId[];

/** Narration lines of a scene for a cut + language (falls back: cut → full, fr → en). */
export const narrationFor = (cut: CutId, locale: LocaleId, sceneId: string): string[] => {
  const L = LOCALES[locale];
  return L.narration[cut]?.[sceneId] ?? L.narration.full?.[sceneId] ?? en.narration[cut]?.[sceneId] ?? en.narration.full?.[sceneId] ?? [];
};

const Ctx = createContext<{locale: LocaleId; cut: CutId}>({locale: 'en', cut: 'full'});

export const LocaleProvider: React.FC<{locale: LocaleId; cut: CutId; children: React.ReactNode}> = ({locale, cut, children}) =>
  React.createElement(Ctx.Provider, {value: {locale, cut}}, children);

export const useLocale = () => useContext(Ctx);

/** t('scene.key', {n: 3}) → the current language's label, falling back to English, then the key. */
export const useT = () => {
  const {locale} = useContext(Ctx);
  return (key: string, vars?: Record<string, string | number>) => {
    let s = LOCALES[locale].labels[key] ?? en.labels[key] ?? key;
    if (vars) for (const [k, v] of Object.entries(vars)) s = s.split(`{${k}}`).join(String(v));
    return s;
  };
};

/** Locale-aware number formatting (0.436 → 0,436 in French). */
export const fmtNum = (locale: LocaleId, v: number, digits: number, grouping = true) =>
  v.toLocaleString(locale === 'fr' ? 'fr-CA' : 'en-US', {minimumFractionDigits: digits, maximumFractionDigits: digits, useGrouping: grouping});

/** fmt(0.436, 3) → "0.436" / "0,436"; fmt(1580) → "1,580" / "1 580"; no digits = as written (0.01). */
export const useFmt = () => {
  const {locale} = useContext(Ctx);
  return (v: number, digits?: number, grouping = true) => fmtNum(locale, v, digits ?? (String(v).split('.')[1]?.length ?? 0), grouping);
};

/** Distances on screen are in light-years (data in Gpc): gly(89) → "290 billion light-years". */
export const useGly = () => {
  const tr = useT();
  return (gpc: number, short = false) => tr(short ? 'units.gly_short' : 'units.gly', {n: Math.round(gpc * 3.2616)});
};
