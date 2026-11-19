/**
 * Live component health, as a strip of indicators in the header.
 *
 * The point of this surface is that the pipeline degrades on purpose: a dead BM25 arm falls
 * back to dense-only retrieval, a dead Ollama falls back to the extractive generator, and a
 * keyless STT provider falls back to local Whisper. A component in that state is amber and
 * stays on screen -- hiding it would hide the harness, which is the thing worth showing.
 */

import { useEffect, useRef, useState } from 'react'

import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { health } from '@/lib/api'
import type { ApiError, ApiResult, ComponentHealth, HealthReport } from '@/lib/types'
import { cn } from '@/lib/utils'

const POLL_MS = 10_000

/** Probe order in routes.get_health. Anything the backend adds later is appended after. */
const KNOWN_ORDER = ['harness', 'embedder', 'vector_store', 'lexical', 'generator', 'stt']

/** Mirrors `_REQUIRED_COMPONENTS` in routes.py, used only if the payload omits `required`. */
const REQUIRED_FALLBACK = ['harness', 'embedder', 'vector_store']

/** Short labels for the strip; the tooltip always shows the exact key the API reported. */
const SHORT: Record<string, string> = {
  harness: 'harness',
  embedder: 'embed',
  vector_store: 'vectors',
  lexical: 'bm25',
  generator: 'gen',
  stt: 'stt',
}

/** What a failure of each component actually costs, in one line. */
const CONSEQUENCE: Record<string, string> = {
  harness: 'Orchestrates every stage. Down means no query can run at all.',
  embedder: 'Encodes the query. Down means no dense retrieval, so no answer.',
  vector_store: 'The dense index. Empty or unreachable means every query refuses.',
  lexical: 'The BM25 arm. Down degrades hybrid retrieval to dense-only.',
  generator: 'The LLM. Down degrades quality mode to the extractive fast path.',
  stt: 'Speech-to-text. Down leaves the text box, which runs the same pipeline.',
}

type Level = 'ok' | 'degraded' | 'down' | 'unknown'

// Three states, three colours, matching the latency waterfall: live, degraded, dead.
const DOT: Record<Level, string> = {
  ok: 'bg-success',
  degraded: 'bg-warning',
  down: 'bg-destructive',
  unknown: 'bg-muted-foreground',
}

const TEXT: Record<Level, string> = {
  ok: 'text-foreground',
  degraded: 'text-warning',
  down: 'text-destructive',
  unknown: 'text-muted-foreground',
}

function levelOf(component: ComponentHealth, required: boolean): Level {
  if (component.ok) return 'ok'
  return required ? 'down' : 'degraded'
}

export interface HealthBadgeProps {
  /** Every poll, success or failure. The shell uses it to raise the empty-index and
   *  backend-down banners without opening a second poller against the same endpoint. */
  onReport?: (result: ApiResult<HealthReport>) => void
  className?: string
}

export function HealthBadge({ onReport, className }: HealthBadgeProps) {
  const [report, setReport] = useState<HealthReport | null>(null)
  const [error, setError] = useState<ApiError | null>(null)

  // Kept in a ref so a caller passing an inline arrow does not restart the poll loop.
  const onReportRef = useRef(onReport)
  onReportRef.current = onReport

  useEffect(() => {
    const controller = new AbortController()
    let alive = true

    const run = async () => {
      const result = await health(controller.signal)
      if (!alive) return
      if (result.ok) {
        setReport(result.data)
        setError(null)
      } else if (result.error.kind !== 'aborted') {
        // The last good report is kept and marked stale: a blank strip mid-demo reads as
        // "nothing is running", which is a stronger claim than "the poll failed".
        setError(result.error)
      }
      onReportRef.current?.(result)
    }

    // The opening poll ignores visibility. Some screen-capture and embedded contexts report
    // `hidden` for a page that is plainly on the projector, and gating the first poll there
    // would leave the strip permanently blank.
    void run()
    const timer = setInterval(() => {
      if (document.visibilityState !== 'hidden') void run()
    }, POLL_MS)
    const onVisible = () => {
      if (document.visibilityState === 'visible') void run()
    }
    document.addEventListener('visibilitychange', onVisible)

    return () => {
      alive = false
      controller.abort()
      clearInterval(timer)
      document.removeEventListener('visibilitychange', onVisible)
    }
  }, [])

  const components = report?.components ?? {}
  const required = new Set(report?.required ?? REQUIRED_FALLBACK)
  const names = [
    ...KNOWN_ORDER.filter((name) => name in components),
    ...Object.keys(components).filter((name) => !KNOWN_ORDER.includes(name)),
  ]

  const rows = names.map((name) => {
    const component = components[name] ?? { ok: false, detail: 'not reported' }
    return { name, component, level: levelOf(component, required.has(name)) }
  })

  const down = rows.filter((row) => row.level === 'down').length
  const degraded = rows.filter((row) => row.level === 'degraded').length
  const overall: Level = !report ? 'unknown' : down > 0 ? 'down' : degraded > 0 ? 'degraded' : 'ok'

  const summary = !report
    ? error
      ? 'backend unreachable'
      : 'probing'
    : down > 0
      ? `${down} down`
      : degraded > 0
        ? `${degraded} degraded`
        : 'all systems live'

  return (
    <div className={cn('flex items-center gap-3', className)} aria-label="Component health">
      <Tooltip>
        <TooltipTrigger
          className="flex items-center gap-2 rounded-md px-1 py-0.5 focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none"
          aria-label={`System health: ${summary}`}
        >
          <span
            className={cn('size-2 rounded-full', DOT[overall], overall === 'ok' && 'animate-pulse')}
            aria-hidden="true"
          />
          <span className={cn('font-mono text-xs whitespace-nowrap', TEXT[overall])}>{summary}</span>
        </TooltipTrigger>
        <TooltipContent side="bottom">
          {error
            ? `GET /api/health: ${error.message}`
            : report
              ? `GET /api/health · polled every ${POLL_MS / 1000}s · required: ${[...required].join(', ')}`
              : 'waiting for the first health probe'}
        </TooltipContent>
      </Tooltip>

      {error && report ? (
        <span className="font-mono text-[11px] text-warning" title={error.message}>
          stale
        </span>
      ) : null}

      <div className="flex items-center gap-1">
        {rows.length === 0
          ? KNOWN_ORDER.map((name) => (
              <span
                key={name}
                className="size-1.5 rounded-full bg-muted-foreground/40"
                aria-hidden="true"
              />
            ))
          : rows.map(({ name, component, level }) => (
              <Tooltip key={name}>
                <TooltipTrigger
                  className="flex items-center gap-1.5 rounded-md border border-border/70 px-2 py-1 focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none"
                  aria-label={`${name}: ${component.ok ? 'ok' : 'not ok'}. ${component.detail}`}
                >
                  <span className={cn('size-1.5 rounded-full', DOT[level])} aria-hidden="true" />
                  <span
                    className={cn(
                      'hidden font-mono text-[11px] xl:inline',
                      level === 'ok' ? 'text-muted-foreground' : TEXT[level],
                    )}
                  >
                    {SHORT[name] ?? name}
                  </span>
                </TooltipTrigger>
                <TooltipContent side="bottom" className="max-w-sm flex-col items-start gap-1">
                  <span className="font-mono text-[11px]">
                    {name} · {component.ok ? 'ok' : required.has(name) ? 'down' : 'degraded'}
                  </span>
                  <span className="opacity-80">{component.detail || 'no detail reported'}</span>
                  {!component.ok && CONSEQUENCE[name] ? (
                    <span className="opacity-80">{CONSEQUENCE[name]}</span>
                  ) : null}
                </TooltipContent>
              </Tooltip>
            ))}
      </div>
    </div>
  )
}

export default HealthBadge
