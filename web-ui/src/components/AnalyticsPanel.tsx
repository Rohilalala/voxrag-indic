/**
 * Live latency analytics, polled from /api/analytics.
 *
 * Everything on this panel is measured by the running process -- there is no benchmark
 * artefact behind it and no number here is typed in by hand. CORE (= TOTAL - STT) is the
 * figure the 200ms budget grades, so CORE P50 / P70 / P100 are the three readouts that get
 * the space; the per-stage table underneath is where the budget actually goes.
 */

import { useEffect, useRef, useState } from 'react'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { analytics } from '@/lib/api'
import type { AnalyticsReport, ApiError, StageName, StageSummary } from '@/lib/types'
import { cn, formatCount, formatMs } from '@/lib/utils'

const POLL_MS = 3_000

/** Newest N CORE samples drawn in the sparkline. */
const SPARK_POINTS = 60

/** Display order in the per-stage table: execution order, then the two aggregates. */
const TABLE_ORDER: StageName[] = [
  'stt',
  'input_guardrail',
  'query_embed',
  'retrieve_dense',
  'retrieve_lexical',
  'fuse',
  'rerank',
  'generate',
  'output_guardrail',
  'core',
  'total',
]

const STAGE_LABEL: Record<StageName, string> = {
  stt: 'stt',
  input_guardrail: 'in.rails',
  query_embed: 'embed',
  retrieve_dense: 'dense',
  retrieve_lexical: 'bm25',
  fuse: 'fuse',
  rerank: 'rerank',
  generate: 'generate',
  output_guardrail: 'out.rails',
  core: 'CORE',
  total: 'total',
}

/** One decimal below a second: the difference between 108.3ms and 130.3ms is the story. */
function heroMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return '--'
  if (ms >= 1000) return (ms / 1000).toFixed(2)
  return ms.toFixed(1)
}

function plural(n: number, singular: string, many = `${singular}s`): string {
  return `${formatCount(n)} ${n === 1 ? singular : many}`
}

function heroUnit(ms: number | null | undefined): string {
  return ms !== null && ms !== undefined && Number.isFinite(ms) && ms >= 1000 ? 's' : 'ms'
}

interface HeroStatProps {
  label: string
  value: number | null | undefined
  budgetMs: number
  emphasis?: boolean
}

function HeroStat({ label, value, budgetMs, emphasis }: HeroStatProps) {
  const known = value !== null && value !== undefined && Number.isFinite(value)
  const over = known && value > budgetMs
  return (
    <div className="flex flex-col gap-1 border-s border-border ps-3 first:border-s-0 first:ps-0">
      <span className="font-mono text-[11px] tracking-widest text-muted-foreground uppercase">
        {label}
      </span>
      <span className="flex items-baseline gap-1">
        <span
          className={cn(
            'font-mono text-3xl leading-none tabular-nums',
            !known && 'text-muted-foreground',
            known && over && 'text-destructive',
            known && !over && (emphasis ? 'text-primary' : 'text-foreground'),
          )}
        >
          {heroMs(value)}
        </span>
        <span className="font-mono text-xs text-muted-foreground">{heroUnit(value)}</span>
      </span>
    </div>
  )
}

export interface AnalyticsPanelProps {
  /** CORE ms per request, this session, oldest first. The endpoint reports percentiles, not
   *  a series, so the shell keeps the series and the panel labels it as session-local. */
  history: number[]
  /** Bumped by the shell when a request lands, to refresh without waiting out the interval. */
  refreshSignal?: number
  className?: string
}

export function AnalyticsPanel({ history, refreshSignal = 0, className }: AnalyticsPanelProps) {
  const [report, setReport] = useState<AnalyticsReport | null>(null)
  const [error, setError] = useState<ApiError | null>(null)

  const controllerRef = useRef<AbortController | null>(null)
  const aliveRef = useRef(true)

  useEffect(() => {
    aliveRef.current = true
    const controller = new AbortController()
    controllerRef.current = controller

    const run = async () => {
      const result = await analytics(controller.signal)
      if (!aliveRef.current) return
      if (result.ok) {
        setReport(result.data)
        setError(null)
      } else if (result.error.kind !== 'aborted') {
        // Keep the last good report on screen and mark it stale rather than blanking the
        // panel: an empty analytics pane looks like "the pipeline measured nothing".
        setError(result.error)
      }
    }

    // The first poll runs even when the document reports hidden -- some capture contexts
    // report that for a foreground page, and the panel would never fill in.
    void run()
    const timer = setInterval(() => {
      // Paused while backgrounded: a demo parked on another tab must not keep loading the
      // backend it is measuring.
      if (document.visibilityState !== 'hidden') void run()
    }, POLL_MS)
    const onVisible = () => {
      if (document.visibilityState === 'visible') void run()
    }
    document.addEventListener('visibilitychange', onVisible)

    return () => {
      aliveRef.current = false
      controller.abort()
      clearInterval(timer)
      document.removeEventListener('visibilitychange', onVisible)
    }
  }, [])

  // A request just finished: pull fresh percentiles now instead of up to POLL_MS later.
  const firstSignal = useRef(true)
  useEffect(() => {
    if (firstSignal.current) {
      firstSignal.current = false
      return
    }
    const controller = controllerRef.current
    if (!controller) return
    void analytics(controller.signal).then((result) => {
      if (!aliveRef.current) return
      if (result.ok) {
        setReport(result.data)
        setError(null)
      } else if (result.error.kind !== 'aborted') {
        setError(result.error)
      }
    })
  }, [refreshSignal])

  const budgetMs = report?.budget.target_ms ?? 200
  const core: StageSummary | undefined = report?.stages.core
  const nCore = core?.n ?? 0
  const withinPct = report && report.budget.n > 0 ? report.budget.within_budget_pct : null

  const rows = TABLE_ORDER.map((stage) => ({ stage, summary: report?.stages[stage] })).filter(
    (row): row is { stage: StageName; summary: StageSummary } =>
      row.summary !== undefined && row.summary.n > 0,
  )

  const series = history.slice(-SPARK_POINTS)
  const peak = series.reduce((max, ms) => Math.max(max, ms), 0)
  const scale = Math.max(budgetMs * 1.2, peak * 1.1, 1)

  return (
    <Card className={cn('gap-3', className)}>
      <CardHeader className="grid-cols-[1fr_auto] items-center gap-2">
        <CardTitle className="font-mono text-[11px] tracking-widest text-muted-foreground uppercase">
          latency analytics
        </CardTitle>
        <div className="flex items-center gap-3 font-mono text-[11px] text-muted-foreground">
          {error ? (
            <span className="text-warning" title={error.message}>
              {report ? 'stale' : 'unreachable'}
            </span>
          ) : null}
          <span>{plural(report?.n_traces ?? 0, 'trace')}</span>
          <span className={cn((report?.retries ?? 0) > 0 && 'text-warning')}>
            {plural(report?.retries ?? 0, 'retry', 'retries')}
          </span>
          <span className={cn((report?.degradations ?? 0) > 0 && 'text-warning')}>
            {plural(report?.degradations ?? 0, 'degradation')}
          </span>
        </div>
      </CardHeader>

      <CardContent className="flex flex-col gap-4">
        {!report && error ? (
          <p className="font-mono text-xs text-destructive">
            GET /api/analytics failed: {error.message}
          </p>
        ) : null}

        <div className="flex items-end justify-between gap-4">
          <div className="flex gap-5">
            <HeroStat label="core p50" value={nCore ? core?.p50 : null} budgetMs={budgetMs} emphasis />
            <HeroStat label="core p70" value={nCore ? core?.p70 : null} budgetMs={budgetMs} />
            <HeroStat label="core p100" value={nCore ? core?.p100 : null} budgetMs={budgetMs} />
          </div>
          <p className="max-w-[15rem] text-end text-[11px] leading-snug text-muted-foreground">
            CORE = TOTAL &minus; STT, measured over {formatCount(nCore)} request
            {nCore === 1 ? '' : 's'} served by this process.
          </p>
        </div>

        <div className="flex flex-col gap-1.5">
          <div className="flex items-baseline justify-between font-mono text-xs">
            <span className="text-muted-foreground">within {Math.round(budgetMs)}ms budget</span>
            <span className="flex items-baseline gap-3">
              <span className="text-muted-foreground">
                worst {report && report.budget.n > 0 ? formatMs(report.budget.worst_ms) : '--'}
              </span>
              <span
                className={cn(
                  'text-base tabular-nums',
                  withinPct === null && 'text-muted-foreground',
                  withinPct !== null && withinPct >= 95 && 'text-success',
                  withinPct !== null && withinPct < 95 && 'text-warning',
                )}
              >
                {withinPct === null ? '--' : `${withinPct.toFixed(1)}%`}
              </span>
            </span>
          </div>
          <Progress
            value={withinPct ?? 0}
            className={cn(
              'h-1.5',
              withinPct !== null && withinPct >= 95 && '*:data-[slot=progress-indicator]:bg-success',
              withinPct !== null && withinPct < 95 && '*:data-[slot=progress-indicator]:bg-warning',
            )}
          />
        </div>

        <div className="flex flex-col gap-1.5">
          <div className="relative flex h-16 items-end gap-px" aria-hidden="true">
            {series.length === 0 ? (
              <div className="flex h-full w-full items-center justify-center font-mono text-[11px] text-muted-foreground">
                no requests yet this session
              </div>
            ) : (
              series.map((ms, index) => (
                <div
                  key={`${index}-${ms}`}
                  className={cn(
                    'min-h-[2px] shrink-0 rounded-t-[1px]',
                    ms > budgetMs ? 'bg-destructive/85' : 'bg-primary/70',
                  )}
                  // Fixed slot per sample rather than flex-1: with two requests on the
                  // board, equal shares would draw two half-width slabs and imply a
                  // history that is not there.
                  style={{
                    width: `calc(${(100 / SPARK_POINTS).toFixed(3)}% - 1px)`,
                    height: `${Math.min(100, (ms / scale) * 100)}%`,
                  }}
                  title={`${formatMs(ms)} core`}
                />
              ))
            )}
            <div
              className="pointer-events-none absolute inset-x-0 border-t border-dashed border-border"
              style={{ bottom: `${Math.min(100, (budgetMs / scale) * 100)}%` }}
            />
          </div>
          <p className="font-mono text-[10px] text-muted-foreground">
            recent CORE latency, this session &middot; dashed line = {Math.round(budgetMs)}ms budget
          </p>
        </div>

        <div className="flex flex-col gap-1">
          <div className="grid grid-cols-[1fr_repeat(4,4.5rem)] gap-2 border-b border-border pb-1 font-mono text-[10px] tracking-widest text-muted-foreground uppercase">
            <span>stage</span>
            <span className="text-end">n</span>
            <span className="text-end">p50</span>
            <span className="text-end">p90</span>
            <span className="text-end">p100</span>
          </div>
          {rows.length === 0 ? (
            <p className="py-2 font-mono text-xs text-muted-foreground">
              No traces yet &mdash; ask something and the table fills in.
            </p>
          ) : (
            rows.map(({ stage, summary }) => {
              const aggregate = stage === 'core' || stage === 'total'
              return (
                <div
                  key={stage}
                  className={cn(
                    'grid grid-cols-[1fr_repeat(4,4.5rem)] gap-2 py-0.5 font-mono text-xs tabular-nums',
                    aggregate && 'border-t border-border pt-1',
                    stage === 'core' ? 'text-foreground' : 'text-muted-foreground',
                  )}
                >
                  <span className={cn(stage === 'core' && 'text-primary')}>
                    {STAGE_LABEL[stage]}
                    {stage === 'stt' ? (
                      <span className="ms-2 text-[10px] text-muted-foreground">excluded</span>
                    ) : null}
                  </span>
                  <span className="text-end">{formatCount(summary.n)}</span>
                  <span className="text-end">{formatMs(summary.p50)}</span>
                  <span className="text-end">{formatMs(summary.p90)}</span>
                  <span className="text-end">{formatMs(summary.p100)}</span>
                </div>
              )
            })
          )}
        </div>
      </CardContent>
    </Card>
  )
}

export default AnalyticsPanel
