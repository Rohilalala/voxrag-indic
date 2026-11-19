/**
 * AnswerCard -- the answer, what it is grounded in, and when the system chose not to give
 * one.
 *
 * Two things here are graded requirements rather than decoration:
 *
 *  1. Citations are clickable and land on the exact supporting span. A Citation carries
 *     source_start/source_end offsets into its chunk's text, so the click hands those
 *     offsets to RetrievedChunks instead of searching for the quote string -- the same
 *     sentence can appear twice in one chunk, and a string search would highlight the
 *     wrong one.
 *  2. A refusal renders as its own thing: not an answer, not an error. It names the rail
 *     that fired and repeats the rail's own `detail`, because "knowing when NOT to answer"
 *     only counts if the user can see the decision being made.
 *
 * Urdu is written in the Arabic script and runs right to left, so the answer body takes
 * its `dir` from the answer's language. Without it the text renders backwards.
 */

import { useEffect, useMemo, useState } from 'react'
import { CircleAlert, Quote, ShieldAlert } from 'lucide-react'

import { setCitationFocus, useCitationFocus } from '@/components/RetrievedChunks'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import type { Answer, Citation, GuardrailVerdict, PipelineResult, RefusalReason } from '@/lib/types'
import { clamp, cn, formatPct, langAttrs, langEndonym, langLabel } from '@/lib/utils'

/** Mirrors `settings.min_groundedness` in config.py. There is no config endpoint in the
 *  frozen API contract, so the threshold marker on the meter is a constant here. */
const MIN_GROUNDEDNESS = 0.55

const REFUSAL_LABEL: Record<RefusalReason, string> = {
  off_topic: 'off topic',
  unsafe: 'unsafe request',
  prompt_injection: 'prompt injection',
  no_relevant_context: 'no relevant context',
  ungrounded_answer: 'ungrounded answer',
  low_confidence: 'low confidence',
  empty_transcript: 'empty transcript',
  unsupported_language: 'unsupported language',
  // The most common refusal a visitor will see: the corpus holds no English passages, so
  // every English question is a cross-language pair. Missing from this map AND from the
  // RefusalReason union, the two gaps masked each other -- Record<RefusalReason, string>
  // only demands the keys the union declares, so tsc stayed green while the badge fell
  // through to `?? reason` and rendered the raw enum string.
  unverified_cross_language: 'unverified across languages',
}

export interface AnswerCardProps {
  result?: PipelineResult | null
  /** Quality mode streams tokens ahead of the final result; render them as they arrive. */
  streamingText?: string
  /** A run is in flight. */
  pending?: boolean
  /** Fires on a citation click, e.g. so a tabbed layout can reveal the chunks panel. */
  onCitationFocus?: (citation: Citation) => void
  className?: string
}

export function AnswerCard({
  result,
  streamingText,
  pending = false,
  onCitationFocus,
  className,
}: AnswerCardProps) {
  const [active, setActive] = useState<number | null>(null)
  const focus = useCitationFocus()

  const answer = result?.answer ?? null
  const citations = useMemo(() => answer?.citations ?? [], [answer])
  const annotated = useMemo(() => annotate(answer?.text ?? '', citations), [answer, citations])
  const retrievedIds = useMemo(
    () => new Set((result?.retrieved ?? []).map((item) => item.chunk.chunk_id)),
    [result],
  )

  // A new run clears the selection here and the highlight over in the chunks panel; so
  // does anything else that clears the shared focus.
  useEffect(() => {
    setActive(null)
    setCitationFocus(null)
  }, [result?.request_id])
  useEffect(() => {
    if (focus === null) setActive(null)
  }, [focus])

  const select = (citation: Citation, index: number) => {
    setActive(index)
    setCitationFocus({
      chunkId: citation.chunk_id,
      sourceStart: citation.source_start,
      sourceEnd: citation.source_end,
    })
    onCitationFocus?.(citation)
  }

  const activeCitation = active === null ? null : (annotated.ordered[active] ?? null)
  const danglingCitation =
    activeCitation && !retrievedIds.has(activeCitation.chunk_id) ? activeCitation : null

  const lang = answer?.lang ?? result?.detected_lang ?? null

  return (
    <section className={cn('flex min-h-0 flex-col gap-3', className)} aria-label="Answer">
      <header className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
        <span className="font-mono text-[11px] tracking-widest text-muted-foreground uppercase">
          {answer?.refused ? 'refusal' : 'answer'}
        </span>
        {answer && (
          <div className="flex flex-wrap items-center gap-1.5">
            <Badge variant="outline" className="font-mono text-[10px]">
              {answer.mode}
            </Badge>
            {answer.model && (
              <Badge variant="outline" className="max-w-48 truncate font-mono text-[10px]">
                {answer.model}
              </Badge>
            )}
            {lang && (
              <Badge variant="outline" className="font-mono text-[10px]">
                <span className="indic-text" {...langAttrs(lang)}>
                  {langEndonym(lang)}
                </span>
                <span className="text-muted-foreground">{langLabel(lang)}</span>
              </Badge>
            )}
          </div>
        )}
      </header>

      {!result && pending && <PendingBody streamingText={streamingText} />}

      {!result && !pending && (
        <p className="rounded-md border border-dashed border-border px-3 py-8 text-center font-mono text-[11px] text-muted-foreground">
          ask in any of the 14 languages — the answer, its citations and the rails that ran
          appear here
        </p>
      )}

      {result && answer?.refused && <Refusal answer={answer} guardrails={result.guardrails} />}

      {result && answer && !answer.refused && (
        <>
          <div
            className="indic-text text-[15px] leading-relaxed text-foreground"
            {...langAttrs(lang)}
          >
            {annotated.pieces.length === 0 ? (
              <span className="text-muted-foreground">the generator returned no text</span>
            ) : (
              annotated.pieces.map((piece, index) => {
                const marker = piece.marker
                if (marker === null) return <span key={index}>{piece.text}</span>
                const citationIndex = marker - 1
                return (
                  <button
                    key={index}
                    type="button"
                    onClick={() => select(annotated.ordered[citationIndex], citationIndex)}
                    aria-label={`Citation ${marker}, highlight the supporting source`}
                    className={cn(
                      'inline cursor-pointer rounded-sm px-0.5 text-start underline decoration-primary/50 decoration-dotted underline-offset-4 transition-colors',
                      active === citationIndex
                        ? 'bg-primary/30 text-foreground'
                        : 'bg-primary/10 hover:bg-primary/20',
                    )}
                  >
                    {piece.text}
                    <sup className="ms-0.5 font-mono text-[10px] text-primary">{marker}</sup>
                  </button>
                )
              })
            )}
          </div>

          <Groundedness value={answer.groundedness} citations={citations.length} />

          {annotated.ordered.length > 0 && (
            <ul className="flex flex-col gap-1">
              {annotated.ordered.map((citation, index) => (
                <li key={`${citation.chunk_id}-${index}`}>
                  <button
                    type="button"
                    onClick={() => select(citation, index)}
                    className={cn(
                      'flex w-full items-baseline gap-2 rounded-sm border px-2 py-1 text-start transition-colors',
                      active === index
                        ? 'border-primary/60 bg-primary/10'
                        : 'border-border hover:bg-muted/60',
                    )}
                  >
                    <span className="shrink-0 font-mono text-[10px] text-primary">
                      [{index + 1}]
                    </span>
                    <Quote aria-hidden className="mt-0.5 size-3 shrink-0 text-muted-foreground" />
                    <span className="indic-text min-w-0 flex-1 text-[12px] text-muted-foreground">
                      {citation.quote || citation.chunk_id}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}

          {danglingCitation && (
            <Alert variant="destructive">
              <CircleAlert />
              <AlertTitle>Citation points outside the returned context</AlertTitle>
              <AlertDescription className="font-mono text-[11px]">
                chunk {danglingCitation.chunk_id} was cited but is not in the retrieved set.
              </AlertDescription>
            </Alert>
          )}
        </>
      )}
    </section>
  )
}

// ---------------------------------------------------------------------------------------
// States
// ---------------------------------------------------------------------------------------

function PendingBody({ streamingText }: { streamingText?: string }) {
  if (streamingText) {
    return (
      <p className="indic-text text-[15px] leading-relaxed text-foreground">
        {streamingText}
        <span
          aria-hidden
          className="ms-0.5 inline-block h-4 w-[2px] translate-y-0.5 animate-pulse bg-primary align-baseline motion-reduce:animate-none"
        />
      </p>
    )
  }
  return (
    <div className="flex flex-col gap-2" aria-label="running the pipeline">
      <Skeleton className="h-4 w-11/12" />
      <Skeleton className="h-4 w-9/12" />
      <Skeleton className="h-4 w-5/12" />
    </div>
  )
}

/** A refusal is a decision, so it is framed as one: the rail that fired, in its own words,
 *  in the warning tone rather than the error tone. */
function Refusal({ answer, guardrails }: { answer: Answer; guardrails: GuardrailVerdict[] }) {
  const blocking = guardrails.filter((rail) => !rail.allowed)
  const reason = answer.refusal_reason ?? blocking[0]?.reason ?? null

  return (
    <div className="flex flex-col gap-3 rounded-md border border-warning/40 bg-warning/5 px-3 py-3">
      <div className="flex flex-wrap items-center gap-2">
        <ShieldAlert aria-hidden className="size-4 text-warning" />
        <span className="font-medium text-warning">declined to answer</span>
        {reason && (
          <span className="rounded-sm border border-warning/40 px-1.5 py-0.5 font-mono text-[10px] text-warning">
            {REFUSAL_LABEL[reason] ?? reason}
          </span>
        )}
      </div>

      <p
        className="indic-text text-[15px] leading-relaxed text-foreground"
        {...langAttrs(answer.lang)}
      >
        {answer.text || 'The system declined to answer.'}
      </p>

      <div className="flex flex-col gap-1.5">
        {blocking.length > 0 ? (
          blocking.map((rail, index) => (
            <div
              key={`${rail.name}-${index}`}
              className="flex flex-wrap items-baseline gap-x-2 gap-y-1 rounded-sm bg-background/60 px-2 py-1.5"
            >
              <span className="font-mono text-[11px] text-warning">{rail.name}</span>
              <span className="min-w-0 flex-1 text-[12px] text-muted-foreground">
                {rail.detail || 'blocked'}
              </span>
              {rail.score > 0 && (
                <span className="font-mono text-[10px] text-muted-foreground tabular-nums">
                  confidence {rail.score.toFixed(2)}
                </span>
              )}
            </div>
          ))
        ) : (
          <div className="rounded-sm bg-background/60 px-2 py-1.5 text-[12px] text-muted-foreground">
            Declined at generation time; no input or output rail blocked.
          </div>
        )}
      </div>

      <p className="font-mono text-[10px] text-muted-foreground/80">
        the pipeline stopped on purpose — this is a policy decision, not a failure
      </p>
    </div>
  )
}

function Groundedness({ value, citations }: { value: number; citations: number }) {
  const score = clamp(Number.isFinite(value) ? value : 0, 0, 1)
  const ok = score >= MIN_GROUNDEDNESS
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between font-mono text-[10px]">
        <span className="tracking-widest text-muted-foreground uppercase">groundedness</span>
        <span className="flex items-baseline gap-3">
          <span className="text-muted-foreground">
            {citations} citation{citations === 1 ? '' : 's'}
          </span>
          <span className={cn('tabular-nums', ok ? 'text-success' : 'text-warning')}>
            {formatPct(score)}
          </span>
        </span>
      </div>
      <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-secondary">
        <div
          className={cn(
            'absolute inset-y-0 start-0 transition-[width] duration-300 motion-reduce:transition-none',
            ok ? 'bg-success' : 'bg-warning',
          )}
          style={{ width: `${score * 100}%` }}
        />
        {/* The threshold the output rail enforces; below it the answer would be refused. */}
        <div
          aria-hidden
          className="absolute inset-y-0 w-px bg-foreground/60"
          style={{ left: `${MIN_GROUNDEDNESS * 100}%` }}
        />
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------------------
// Citation markup
// ---------------------------------------------------------------------------------------

interface Piece {
  text: string
  /** 1-based citation number, or null for the plain text between citations. */
  marker: number | null
}

interface Annotated {
  pieces: Piece[]
  /** Citations in the order their markers appear, followed by any that produced no marker. */
  ordered: Citation[]
}

/**
 * Splits the answer into plain runs and cited runs using each citation's answer offsets.
 * Citations are sorted and overlaps dropped: two rails can cite the same sentence, and
 * nested markers would corrupt the offsets. A citation whose answer offsets are degenerate
 * (a generator can emit all-zero spans) still reaches the chip list below the answer, so
 * nothing the backend returned is silently discarded.
 */
function annotate(text: string, citations: readonly Citation[]): Annotated {
  const pieces: Piece[] = []
  const ordered: Citation[] = []
  if (!text) return { pieces, ordered }

  const usable = citations
    .filter(
      (citation) =>
        Number.isFinite(citation.answer_start) &&
        Number.isFinite(citation.answer_end) &&
        citation.answer_end > citation.answer_start,
    )
    .slice()
    .sort((a, b) => a.answer_start - b.answer_start)

  let cursor = 0
  for (const citation of usable) {
    const from = clamp(Math.trunc(citation.answer_start), cursor, text.length)
    const to = clamp(Math.trunc(citation.answer_end), from, text.length)
    if (to <= from) continue
    if (from > cursor) pieces.push({ text: text.slice(cursor, from), marker: null })
    ordered.push(citation)
    pieces.push({ text: text.slice(from, to), marker: ordered.length })
    cursor = to
  }
  if (cursor < text.length) pieces.push({ text: text.slice(cursor), marker: null })

  for (const citation of citations) {
    if (!ordered.includes(citation)) ordered.push(citation)
  }
  return { pieces, ordered }
}

export default AnswerCard
