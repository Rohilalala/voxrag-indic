/**
 * Wire types for the VoxRAG API.
 *
 * Mirrors `src/voxrag/contracts.py` field for field. The backend is frozen and tested; this
 * file is the only place the frontend is allowed to know its field names. Python enums
 * serialise to their string values (`model_dump(mode="json")`), so every enum here is a
 * string union -- also required because tsconfig sets `erasableSyntaxOnly`, which bans
 * TypeScript `enum`.
 */

// ---------------------------------------------------------------------------------------
// Enumerations (contracts.py)
// ---------------------------------------------------------------------------------------

/** The 14 Indic languages in ai4bharat/MSMARCO-XI. */
export const LANGS = [
  'as',
  'bn',
  'gu',
  'hi',
  'kn',
  'ml',
  'mr',
  'ne',
  'or',
  'pa',
  'sa',
  'ta',
  'te',
  'ur',
] as const

export type IndicLang = (typeof LANGS)[number]

/** `Lang` in contracts.py: the 14 Indic languages plus the English pivot. */
export type Lang = IndicLang | 'en'

/** English name per language, for the language selector. */
export const LANG_LABEL: Record<Lang, string> = {
  as: 'Assamese',
  bn: 'Bengali',
  gu: 'Gujarati',
  hi: 'Hindi',
  kn: 'Kannada',
  ml: 'Malayalam',
  mr: 'Marathi',
  ne: 'Nepali',
  or: 'Odia',
  pa: 'Punjabi',
  sa: 'Sanskrit',
  ta: 'Tamil',
  te: 'Telugu',
  ur: 'Urdu',
  en: 'English',
}

/** The language's own name, in its own script. Doubles as a live tofu check: if these
 *  render as boxes, the font stack in index.css is not resolving. */
export const LANG_ENDONYM: Record<Lang, string> = {
  as: 'অসমীয়া',
  bn: 'বাংলা',
  gu: 'ગુજરાતી',
  hi: 'हिन्दी',
  kn: 'ಕನ್ನಡ',
  ml: 'മലയാളം',
  mr: 'मराठी',
  ne: 'नेपाली',
  or: 'ଓଡ଼ିଆ',
  pa: 'ਪੰਜਾਬੀ',
  sa: 'संस्कृतम्',
  ta: 'தமிழ்',
  te: 'తెలుగు',
  ur: 'اردو',
  en: 'English',
}

/** Unicode script per language -- mirrors LANG_SCRIPT in contracts.py. */
export const LANG_SCRIPT: Record<Lang, string> = {
  as: 'Bengali',
  bn: 'Bengali',
  gu: 'Gujarati',
  hi: 'Devanagari',
  kn: 'Kannada',
  ml: 'Malayalam',
  mr: 'Devanagari',
  ne: 'Devanagari',
  or: 'Oriya',
  pa: 'Gurmukhi',
  sa: 'Devanagari',
  ta: 'Tamil',
  te: 'Telugu',
  ur: 'Arabic',
  en: 'Latin',
}

/** Right-to-left languages. Urdu is written in the Arabic script; without `dir="rtl"` its
 *  text renders in the wrong visual order. */
export const RTL_LANGS: ReadonlySet<string> = new Set<Lang>(['ur'])

export const CHUNK_STRATEGIES = [
  'fixed',
  'fixed_overlap',
  'sentence_window',
  'semantic',
  'recursive',
  'passage_native',
  'parent_child',
  'metadata_aware',
] as const

export type ChunkStrategyName = (typeof CHUNK_STRATEGIES)[number]

/** Pipeline stages in canonical execution order. `total` and `core` are synthetic:
 *  CORE = TOTAL - STT, and CORE is what the 200ms budget is measured against. */
export const STAGES = [
  'stt',
  'input_guardrail',
  'query_embed',
  'retrieve_dense',
  'retrieve_lexical',
  'fuse',
  'rerank',
  'generate',
  'output_guardrail',
  'total',
  'core',
] as const

export type StageName = (typeof STAGES)[number]

/** The stages that make up one pass, excluding the two synthetic aggregates. */
export const PIPELINE_STAGES = STAGES.filter(
  (s): s is Exclude<StageName, 'total' | 'core'> => s !== 'total' && s !== 'core',
)

export const GENERATION_MODES = ['fast', 'quality'] as const
export type GenerationMode = (typeof GENERATION_MODES)[number]

export type RefusalReason =
  | 'off_topic'
  | 'unsafe'
  | 'prompt_injection'
  | 'no_relevant_context'
  | 'ungrounded_answer'
  | 'low_confidence'
  | 'empty_transcript'
  | 'unsupported_language'
  | 'unverified_cross_language'

// ---------------------------------------------------------------------------------------
// Core models (contracts.py)
// ---------------------------------------------------------------------------------------

/** Free-form per-stage detail: token counts, candidate counts, cache hits. */
export type Detail = Record<string, unknown>

export interface Chunk {
  chunk_id: string
  doc_id: string
  text: string
  lang: Lang
  strategy: ChunkStrategyName
  ordinal: number
  char_start: number
  char_end: number
  n_tokens: number
  title: string | null
  url: string | null
  /** Wider text handed to the generator; falls back to `text` when null. */
  context_text: string | null
  parent_id: string | null
  meta: Detail
}

export interface ScoredChunk {
  chunk: Chunk
  /** Fused final score, higher is better. Comparable only within one result set. */
  score: number
  dense_score: number | null
  lexical_score: number | null
  rerank_score: number | null
  /** 1-based rank within each retriever, for RRF diagnostics. */
  dense_rank: number | null
  lexical_rank: number | null
}

export interface Citation {
  chunk_id: string
  doc_id: string
  /** Offsets into the *answer* text this citation supports. */
  answer_start: number
  answer_end: number
  /** Offsets into the *chunk* text that provide the support. */
  source_start: number
  source_end: number
  quote: string
}

export interface StageTiming {
  stage: StageName
  ms: number
  skipped: boolean
  detail: Detail
}

export interface Transcript {
  text: string
  lang: Lang | null
  confidence: number | null
  provider: string
  duration_s: number | null
  /** True when a fallback provider served this after the primary failed. */
  from_fallback: boolean
  raw: Detail
}

export interface GuardrailVerdict {
  name: string
  /** `false` short-circuits the pipeline into a refusal. */
  allowed: boolean
  reason: RefusalReason | null
  score: number
  detail: string
  rewritten_text: string | null
}

export interface Answer {
  text: string
  mode: GenerationMode
  citations: Citation[]
  /** Fraction of answer tokens covered by at least one citation, in [0,1]. */
  groundedness: number
  refused: boolean
  refusal_reason: RefusalReason | null
  model: string
  lang: Lang | null
}

/** The complete, auditable record of one query. The response body of /api/query,
 *  /api/voice, and the `result` websocket frame. */
export interface PipelineResult {
  request_id: string
  transcript: Transcript | null
  query_text: string
  detected_lang: Lang | null
  answer: Answer
  retrieved: ScoredChunk[]
  guardrails: GuardrailVerdict[]
  timings: StageTiming[]
  /** Populated when a stage failed but the harness recovered. */
  degradations: string[]
  retries: number
}

// ---------------------------------------------------------------------------------------
// Request bodies
// ---------------------------------------------------------------------------------------

export interface QueryRequest {
  text: string
  /** null / omitted means auto-detect from the script. */
  lang?: Lang | null
  mode?: GenerationMode
}

export interface VoiceRequest {
  audio: Blob
  lang?: Lang | null
  mode?: GenerationMode
  /** Overrides the blob's own MIME type. The backend sniffs magic bytes anyway. */
  mime?: string
}

// ---------------------------------------------------------------------------------------
// GET /api/analytics
// ---------------------------------------------------------------------------------------

/** One stage's latency distribution. Percentile keys come from
 *  `settings.report_percentiles` = (50, 70, 90, 95, 99, 100). */
export interface StageSummary {
  n: number
  mean: number
  min: number
  max: number
  p50: number
  p70: number
  p90: number
  p95: number
  p99: number
  p100: number
}

export interface BudgetReport {
  /** 200.0 ms. */
  target_ms: number
  /** Number of CORE samples the percentage is computed over. */
  n: number
  within_budget_pct: number
  worst_ms: number
}

export interface AnalyticsReport {
  /** Keyed by stage name. A stage that has never run is absent, not zeroed. */
  stages: Partial<Record<StageName, StageSummary>>
  budget: BudgetReport
  degradations: number
  retries: number
  n_traces: number
}

// ---------------------------------------------------------------------------------------
// GET /api/health
// ---------------------------------------------------------------------------------------

export interface ComponentHealth {
  ok: boolean
  detail: string
}

export interface HealthReport {
  /** True only when every *required* component is ok. Always served with HTTP 200. */
  ok: boolean
  components: Record<string, ComponentHealth>
  /** Components whose failure means the service cannot answer at all. */
  required?: string[]
  warmup?: Detail
}

// ---------------------------------------------------------------------------------------
// GET /api/strategies
// ---------------------------------------------------------------------------------------

export interface StrategyInfo {
  name: ChunkStrategyName
  description: string
  /** null when the strategy was never built into the live index. */
  n_chunks: number | null
  indexed: boolean
  /** The strategy currently serving retrieval. */
  active?: boolean
}

// ---------------------------------------------------------------------------------------
// WS /ws/voice
// ---------------------------------------------------------------------------------------

/** Client -> server. The only frame the server accepts besides `ping`. */
export interface WsAudioMessage {
  type: 'audio'
  mime: string
  /** Base64, with or without a `data:` URL prefix. */
  data: string
  lang: Lang | null
  mode: GenerationMode
}

export interface WsStageFrame {
  type: 'stage'
  stage: StageName
  ms: number
}

export interface WsTranscriptFrame {
  type: 'transcript'
  text: string
  /** Empty string when the provider reported no language. */
  lang: string
}

/** Quality mode only -- an incrementally generated token. */
export interface WsTokenFrame {
  type: 'token'
  text: string
}

export interface WsResultFrame {
  type: 'result'
  result: PipelineResult
}

export interface WsErrorFrame {
  type: 'error'
  message: string
  /** Present when the failure happened inside a run rather than at the socket level. */
  request_id?: string
  error?: string
  stage?: string
}

export type WsServerFrame =
  | WsStageFrame
  | WsTranscriptFrame
  | WsTokenFrame
  | WsResultFrame
  | WsErrorFrame

// ---------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------

export type ApiErrorKind =
  | 'network' // the request never reached the server
  | 'http' // the server answered with a non-2xx
  | 'parse' // 2xx, but the body was not the shape we expect
  | 'timeout'
  | 'aborted'
  | 'socket'

/** Every failure the API layer can produce, in one shape. Nothing throws past `lib/api`. */
export interface ApiError {
  kind: ApiErrorKind
  /** Always safe to render directly. */
  message: string
  status?: number
  /** The backend stamps this on every error envelope; it finds the request in the logs. */
  requestId?: string
  /** The backend's exception class name, when it sent one. */
  error?: string
  stage?: string
}

/** Result of any API call. Success and failure are both values -- a component renders the
 *  error instead of unmounting into a blank screen. */
export type ApiResult<T> = { ok: true; data: T } | { ok: false; error: ApiError }
