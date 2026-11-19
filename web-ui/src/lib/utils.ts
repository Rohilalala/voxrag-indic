import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

import { LANG_ENDONYM, LANG_LABEL, RTL_LANGS, type Lang } from './types'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// ---------------------------------------------------------------------------------------
// Language / script
// ---------------------------------------------------------------------------------------

/** Urdu is written in the Arabic script and runs right to left. */
export function isRtl(lang: Lang | null | undefined): boolean {
  return lang !== null && lang !== undefined && RTL_LANGS.has(lang)
}

/** Spread onto any element holding user- or model-generated text:
 *  `<p {...langAttrs(result.detected_lang)}>`. `lang` lets the browser pick the right
 *  font from the fallback chain and hyphenate correctly; `dir` is what stops Urdu from
 *  rendering backwards. */
export function langAttrs(lang: Lang | null | undefined): { lang?: string; dir: 'rtl' | 'ltr' } {
  return { ...(lang ? { lang } : {}), dir: isRtl(lang) ? 'rtl' : 'ltr' }
}

/** "Hindi", or the raw code if the backend ever reports one we do not know. */
export function langLabel(lang: Lang | null | undefined): string {
  if (!lang) return 'auto-detect'
  return LANG_LABEL[lang] ?? lang
}

/** The language's own name in its own script. */
export function langEndonym(lang: Lang | null | undefined): string {
  if (!lang) return ''
  return LANG_ENDONYM[lang] ?? lang
}

// ---------------------------------------------------------------------------------------
// Numbers
// ---------------------------------------------------------------------------------------

/** Milliseconds for a readout: sub-10ms keeps a decimal, anything above is an integer,
 *  and seconds take over past 10s so the column never grows a digit mid-demo. */
export function formatMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return '--'
  if (ms >= 10_000) return `${(ms / 1000).toFixed(1)}s`
  if (ms < 10) return `${ms.toFixed(1)}ms`
  return `${Math.round(ms)}ms`
}

/** A [0,1] fraction as a percentage. */
export function formatPct(fraction: number | null | undefined, digits = 0): string {
  if (fraction === null || fraction === undefined || !Number.isFinite(fraction)) return '--'
  return `${(fraction * 100).toFixed(digits)}%`
}

/** Thousands separators, e.g. 2268974 -> "2,268,974". */
export function formatCount(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return '--'
  return n.toLocaleString('en-US')
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}
