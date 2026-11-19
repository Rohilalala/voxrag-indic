/**
 * LatencyWaterfall -- where one run's time went, against the 200ms budget.
 *
 * Four decisions here are about honesty rather than looks, and each one is load-bearing:
 *
 *  1. The bar plots the stages that CONSTITUTE core. `total` is not a stage, and `stt` is
 *     excluded on purpose: a speech-to-text round trip to a third-party vendor is
 *     1.5-2.7s, so including it would push the 200ms line into a sliver and make the
 *     budget meaningless. STT is still reported, in the header, labelled "excluded".
 *  2. Skipped stages are omitted from the bar instead of being drawn as zero-width
 *     slivers, and are listed as `skipped` in the legend with the backend's own reason.
 *  3. retrieve_dense and retrieve_lexical run CONCURRENTLY. They share one bracketed
 *     column whose wall-clock contribution is max(dense, bm25), not their sum. When the
 *     measured CORE is still shorter than the stage clocks add up to, the widths are
 *     normalised to CORE and the panel says so -- a bar that silently outruns its own
 *     total looks like a bug.
 *  4. Anything CORE has that the stages do not account for is drawn as `harness`
 *     (orchestration, serialisation) rather than being quietly dropped.
 *
 * Stage frames stream in over the websocket in execution order, so passing a growing
 * `timings` array animates the bar as the pipeline runs. Motion is CSS-only and collapses
 * under prefers-reduced-motion.
 */

import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import type { Detail, PipelineResult, StageName, StageTiming } from '@/lib/types'
import { cn, formatMs } from '@/lib/utils'

// ---------------------------------------------------------------------------------------
// Stage vocabulary
// ---------------------------------------------------------------------------------------

/** The stages that make up CORE, in execution order. `stt`, `total` and `core` are all
 *  deliberately absent: the first is excluded from the budget, the other two are the
 *  aggregates the bar is measured against. */
const CORE_STAGES = [
  'input_guardrail',
  'query_embed',
  'retrieve_dense',
  'retrieve_lexical',
  'fuse',
  'rerank',
  'grade',
  'rewrite',
  'generate',
  'output_guardrail',
] as const

type CoreStage = (typeof CORE_STAGES)[number]

/** One hue family swept from violet to the interface accent, so the bar reads as a
 *  progression rather than a rainbow. `harness` sits just above the track colour because
 *  it is not a real stage, and must never look like one.
 *
 *  Exported because RetrievedChunks colours its dense/bm25 arm chips from the same two
 *  entries -- the colour is how a viewer connects "this chunk came from BM25" to the
 *  BM25 slice in the waterfall. */
export const STAGE_COLORS = {
  input_guardrail: 'oklch(0.60 0.13 268)',
  query_embed: 'oklch(0.65 0.13 255)',
  retrieve_dense: 'oklch(0.70 0.13 240)',
  retrieve_lexical: 'oklch(0.74 0.13 222)',
  fuse: 'oklch(0.78 0.13 208)',
  rerank: 'oklch(0.82 0.13 198)',
  grade: 'oklch(0.83 0.13 195)',
  rewrite: 'oklch(0.85 0.13 192)',
  generate: 'oklch(0.86 0.12 190)',
  output_guardrail: 'oklch(0.90 0.10 184)',
  harness: 'oklch(0.34 0.012 250)',
} as const satisfies Record<CoreStage | 'harness', string>

const STAGE_LABEL: Record<CoreStage | 'harness', string> = {
  input_guardrail: 'in.rails',
  query_embed: 'embed',
  retrieve_dense: 'dense',
  retrieve_lexical: 'bm25',
  fuse: 'fuse',
  rerank: 'rerank',
  grade: 'grade',
  rewrite: 'rewrite',
  generate: 'generate',
  output_guardrail: 'out.rails',
  harness: 'harness',
}

/** Hover copy. A reader should not have to know the pipeline to read the bar. */
const STAGE_NOTE: Record<CoreStage | 'harness', string> = {
  input_guardrail: 'Safety, off-topic and prompt-injection checks on the query.',
  query_embed: 'Encodes the query into the multilingual dense vector.',
  retrieve_dense: 'ANN search over 2,268,974 chunks. Runs concurrently with BM25.',
  retrieve_lexical: 'Script-aware BM25. Runs concurrently with dense retrieval.',
  fuse: 'Reciprocal-rank fusion of the two candidate lists.',
  rerank: 'Cross-encoder precision pass over the fused candidates.',
  grade: 'Agentic mode: the model judges whether each passage answers the question.',
  rewrite: 'Agentic mode: the model rewrites the query when no passage answered it.',
  generate: 'Builds the cited answer from the retrieved context.',
  output_guardrail: 'Groundedness and citation checks on the answer.',
  harness: 'Orchestration and serialisation inside CORE but outside any one stage.',
}

/** One drawn column. `bars` holds two entries only for the concurrent retrieval pair. */
interface Column {
  key: string
  /** Wall-clock contribution: the stage's own ms, or max() across a concurrent pair. */
  ms: number
  bars: { stage: CoreStage; ms: number }[]
  parallel: boolean
}

const RETRIEVAL_PAIR: readonly [CoreStage, CoreStage] = ['retrieve_dense', 'retrieve_lexical']

// ---------------------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------------------

export interface LatencyWaterfallProps {
  /** Per-stage timings. Grows frame by frame while a run streams; replaced wholesale by
   *  the authoritative array on the final `result`. */
  timings?: readonly StageTiming[] | null
  /** Convenience: used when `timings` is not passed. */
  result?: PipelineResult | null
  /** A run is in flight. Only changes how the not-yet-final state is labelled. */
  running?: boolean
  /** Defaults to the backend's own `core.detail.budget_ms`, then to 200ms. */
  budgetMs?: number
  className?: string
}

export function LatencyWaterfall({
  timings,
  result,
  running = false,
  budgetMs,
  className,
}: LatencyWaterfallProps) {
  // Eleven stages at most, so the model is rebuilt inline rather than memoised.
  const model = buildModel(timings ?? result?.timings ?? [], budgetMs)
  const {
    byStage,
    columns,
    accounted,
    core,
    total,
    stt,
    budget,
    harnessMs,
    normalised,
    scale,
    barMs,
  } = model

  const state: 'ok' | 'over' | 'live' | 'idle' =
    core !== null ? (core <= budget ? 'ok' : 'over') : running ? 'live' : 'idle'

  const budgetPct = clampPct((budget / scale) * 100)
  // Past ~72% the label would run off the right edge, so it flips to the other side.
  const budgetFlip = budgetPct > 72
  const ticks = buildTicks(scale)

  const coreTone =
    state === 'ok'
      ? 'text-success'
      : state === 'over'
        ? 'text-destructive'
        : state === 'live'
          ? 'text-primary'
          : 'text-muted-foreground'

  const lineTone =
    state === 'ok'
      ? 'border-success/80'
      : state === 'over'
        ? 'border-destructive/80'
        : 'border-foreground/35'

  const hasRetrievalPair = columns.some((c) => c.parallel)

  return (
    <TooltipProvider delayDuration={80}>
      <section className={cn('flex flex-col gap-4', className)} aria-label="Latency waterfall">
        {/* ---- readouts -------------------------------------------------------------- */}
        <header className="flex flex-wrap items-end justify-between gap-x-8 gap-y-2">
          <div className="flex items-baseline gap-3">
            <span className="font-mono text-[11px] tracking-widest text-muted-foreground uppercase">
              core
            </span>
            <span
              className={cn(
                'font-mono text-3xl leading-none tabular-nums transition-colors',
                coreTone,
              )}
            >
              {core !== null || (running && barMs > 0) ? formatMs(barMs) : '--'}
            </span>
            <span
              className={cn(
                'rounded-sm border px-1.5 py-0.5 font-mono text-[10px] tracking-wide whitespace-nowrap',
                state === 'ok'
                  ? 'border-success/40 text-success'
                  : state === 'over'
                    ? 'border-destructive/40 text-destructive'
                    : 'border-border text-muted-foreground',
              )}
            >
              {state === 'ok'
                ? `within ${Math.round(budget)}ms budget`
                : state === 'over'
                  ? `over ${Math.round(budget)}ms budget`
                  : state === 'live'
                    ? 'measuring'
                    : `${Math.round(budget)}ms budget`}
            </span>
          </div>

          <div className="flex items-baseline gap-5 font-mono text-[11px] text-muted-foreground">
            <span>
              total <b className="font-medium text-foreground">{total === null ? '--' : formatMs(total)}</b>
            </span>
            <span>
              stt{' '}
              <b className="font-medium text-foreground">
                {stt === null ? '--' : stt.skipped ? 'skipped' : formatMs(stt.ms)}
              </b>{' '}
              <span className="text-muted-foreground/70">excluded from core</span>
            </span>
          </div>
        </header>

        {/* ---- bar ------------------------------------------------------------------- */}
        <div className="relative pt-5">
          <div className="relative h-12 w-full overflow-hidden rounded-md bg-secondary/40 ring-1 ring-border ring-inset">
            {ticks.map((v) => (
              <div
                key={`tick-${v}`}
                aria-hidden
                className="absolute inset-y-0 w-px bg-foreground/6"
                style={{ left: `${(v / scale) * 100}%` }}
              />
            ))}

            <div className="absolute inset-0 flex items-stretch">
              {columns.map((column) => (
                <div
                  key={column.key}
                  className={cn(
                    'flex h-full flex-col gap-px transition-[width] duration-300 ease-out',
                    'animate-in fade-in-0 slide-in-from-left-1',
                    'motion-reduce:animate-none motion-reduce:transition-none',
                    column.parallel && 'rounded-[3px] bg-foreground/6 ring-1 ring-foreground/25 ring-inset',
                  )}
                  style={{ width: `${(column.ms / scale) * 100 * normalised}%` }}
                >
                  {column.bars.map((bar) => (
                    <Tooltip key={bar.stage}>
                      <TooltipTrigger asChild>
                        <button
                          type="button"
                          aria-label={`${STAGE_LABEL[bar.stage]} ${bar.ms.toFixed(2)} milliseconds`}
                          // border-e separates neighbouring slices without widening them:
                          // box-border means the border eats into the inline width.
                          className="block min-h-[3px] min-w-[2px] flex-1 rounded-[2px] border-e border-background/60 outline-none focus-visible:ring-2 focus-visible:ring-foreground/70"
                          style={{
                            width: `${(bar.ms / column.ms) * 100}%`,
                            backgroundColor: STAGE_COLORS[bar.stage],
                          }}
                        />
                      </TooltipTrigger>
                      <TooltipContent side="top" className="max-w-64">
                        <span className="flex flex-col gap-0.5">
                          <span className="font-mono">
                            {STAGE_LABEL[bar.stage]} · {bar.ms.toFixed(2)}ms
                          </span>
                          <span className="opacity-80">{STAGE_NOTE[bar.stage]}</span>
                        </span>
                      </TooltipContent>
                    </Tooltip>
                  ))}
                </div>
              ))}

              {harnessMs > 0 && (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button
                      type="button"
                      aria-label={`harness ${harnessMs.toFixed(2)} milliseconds`}
                      className="h-full min-w-[2px] outline-none focus-visible:ring-2 focus-visible:ring-foreground/70"
                      style={{
                        width: `${(harnessMs / scale) * 100}%`,
                        backgroundColor: STAGE_COLORS.harness,
                      }}
                    />
                  </TooltipTrigger>
                  <TooltipContent side="top" className="max-w-64">
                    <span className="flex flex-col gap-0.5">
                      <span className="font-mono">harness · {harnessMs.toFixed(2)}ms</span>
                      <span className="opacity-80">{STAGE_NOTE.harness}</span>
                    </span>
                  </TooltipContent>
                </Tooltip>
              )}
            </div>

            {columns.length === 0 && (
              <p className="absolute inset-0 flex items-center px-3 font-mono text-[11px] text-muted-foreground">
                {running ? 'waiting for the first stage…' : 'no run yet'}
              </p>
            )}
          </div>

          {/* axis */}
          <div className="relative mt-1 h-4">
            {ticks.map((v) => (
              <span
                key={`axis-${v}`}
                className={cn(
                  'absolute font-mono text-[10px] text-muted-foreground tabular-nums',
                  v === 0 ? '' : '-translate-x-1/2',
                )}
                style={{ left: `${(v / scale) * 100}%` }}
              >
                {v}
              </span>
            ))}
            <span className="absolute end-0 font-mono text-[10px] text-muted-foreground">ms</span>
          </div>

          {/* budget line, drawn over everything */}
          <div
            className="pointer-events-none absolute inset-y-0 z-10"
            style={{ left: `${budgetPct}%` }}
            aria-hidden
          >
            <div className={cn('absolute top-5 bottom-4 border-l border-dashed', lineTone)} />
            <span
              className={cn(
                'absolute top-0 font-mono text-[10px] whitespace-nowrap',
                state === 'ok'
                  ? 'text-success'
                  : state === 'over'
                    ? 'text-destructive'
                    : 'text-muted-foreground',
                budgetFlip ? '-translate-x-full pe-1.5' : 'ps-1.5',
              )}
            >
              {Math.round(budget)}ms budget
            </span>
          </div>
        </div>

        {/* ---- legend ---------------------------------------------------------------- */}
        <div className="flex flex-wrap gap-x-4 gap-y-1.5 font-mono text-[11px]">
          {CORE_STAGES.map((stage) => {
            const timing = byStage.get(stage)
            if (!timing) return null
            const skipped = timing.skipped
            const why = typeof timing.detail?.why === 'string' ? timing.detail.why : ''
            return (
              <span
                key={stage}
                title={skipped && why ? `skipped: ${why}` : undefined}
                className={cn('flex items-center gap-1.5', skipped && 'text-muted-foreground/70')}
              >
                <i
                  aria-hidden
                  className={cn('size-2 shrink-0 rounded-[2px]', skipped && 'opacity-30')}
                  style={{ backgroundColor: STAGE_COLORS[stage] }}
                />
                <span className={cn(skipped ? 'text-muted-foreground/70' : 'text-muted-foreground')}>
                  {STAGE_LABEL[stage]}
                </span>
                <b className={cn('font-medium', skipped ? 'italic' : 'text-foreground')}>
                  {skipped ? 'skipped' : formatMs(timing.ms)}
                </b>
              </span>
            )
          })}
          {harnessMs > 0 && (
            <span className="flex items-center gap-1.5">
              <i
                aria-hidden
                className="size-2 shrink-0 rounded-[2px]"
                style={{ backgroundColor: STAGE_COLORS.harness }}
              />
              <span className="text-muted-foreground">harness</span>
              <b className="font-medium text-foreground">{formatMs(harnessMs)}</b>
            </span>
          )}
        </div>

        {/* ---- the caveats that keep the picture honest ------------------------------ */}
        <ul className="flex flex-col gap-1 font-mono text-[10px] leading-relaxed text-muted-foreground/80">
          {hasRetrievalPair && (
            <li>
              dense + bm25 share one bracketed column — they run concurrently, so the column is
              max(dense, bm25), not their sum.
            </li>
          )}
          {normalised < 1 && core !== null && (
            <li className="text-warning/90">
              stage clocks add up to {formatMs(accounted)} against a {formatMs(core)} wall-clock
              core; widths are scaled to core so the budget line stays true.
            </li>
          )}
          <li>skipped stages are listed above but drawn nowhere; stt is excluded from core.</li>
        </ul>
      </section>
    </TooltipProvider>
  )
}

// ---------------------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------------------

interface WaterfallModel {
  byStage: Map<StageName, StageTiming>
  columns: Column[]
  /** Sum of the drawn columns' wall-clock contributions. */
  accounted: number
  core: number | null
  total: number | null
  stt: StageTiming | null
  budget: number
  harnessMs: number
  /** <1 when the stage clocks outrun the measured core (concurrency), else exactly 1. */
  normalised: number
  scale: number
  /** What the bar represents: core once known, the running sum until then. */
  barMs: number
}

function buildModel(timings: readonly StageTiming[], budgetOverride?: number): WaterfallModel {
  const byStage = new Map<StageName, StageTiming>()
  for (const timing of timings) {
    if (!timing || !Number.isFinite(timing.ms)) continue
    // Last write wins: a retried stage reports twice, and the later figure is the real one.
    byStage.set(timing.stage, timing)
  }

  const columns: Column[] = []
  for (const stage of CORE_STAGES) {
    if (stage === RETRIEVAL_PAIR[1]) continue // drawn inside the pair's column
    if (stage === RETRIEVAL_PAIR[0]) {
      const bars = RETRIEVAL_PAIR.map((s) => ({ stage: s, timing: byStage.get(s) }))
        .filter((b) => b.timing && !b.timing.skipped && b.timing.ms > 0)
        .map((b) => ({ stage: b.stage, ms: b.timing!.ms }))
      if (bars.length === 0) continue
      columns.push({
        key: 'retrieval',
        // Concurrent: wall clock is the slower arm, not the two added together.
        ms: Math.max(...bars.map((b) => b.ms)),
        bars,
        parallel: bars.length > 1,
      })
      continue
    }
    const timing = byStage.get(stage)
    if (!timing || timing.skipped || timing.ms <= 0) continue
    columns.push({ key: stage, ms: timing.ms, bars: [{ stage, ms: timing.ms }], parallel: false })
  }

  const accounted = columns.reduce((sum, column) => sum + column.ms, 0)
  const core = numeric(byStage.get('core')?.ms)
  const total = numeric(byStage.get('total')?.ms)
  const stt = byStage.get('stt') ?? null

  const budget = budgetOverride ?? detailBudget(byStage.get('core')?.detail) ?? 200

  const harnessMs = core !== null && core > accounted + 0.05 ? core - accounted : 0
  const normalised = core !== null && core > 0 && accounted > core + 0.05 ? core / accounted : 1

  const barMs = core ?? accounted
  const scale = Math.max(budget * 1.15, barMs * 1.08, 1)

  return { byStage, columns, accounted, core, total, stt, budget, harnessMs, normalised, scale, barMs }
}

function numeric(value: number | undefined): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

/** The backend stamps the real budget onto the CORE timing; never invent one. */
function detailBudget(detail: Detail | undefined): number | null {
  const value = detail?.budget_ms
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : null
}

function buildTicks(scale: number): number[] {
  const step = tickStep(scale)
  const ticks: number[] = []
  for (let v = 0; v <= scale - step * 0.4; v += step) ticks.push(Math.round(v))
  return ticks
}

function tickStep(scale: number): number {
  if (scale <= 60) return 10
  if (scale <= 130) return 20
  if (scale <= 320) return 50
  if (scale <= 700) return 100
  if (scale <= 1600) return 250
  if (scale <= 4000) return 500
  return 1000
}

function clampPct(value: number): number {
  return Math.min(100, Math.max(0, value))
}

export default LatencyWaterfall
