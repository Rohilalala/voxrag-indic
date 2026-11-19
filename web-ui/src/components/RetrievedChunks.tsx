/**
 * RetrievedChunks -- the evidence the answer was built from, with its provenance intact.
 *
 * The point of this panel is that a viewer can SEE hybrid retrieval working. Every chunk
 * carries two arms: dense (ANN over the embedding index) and lexical (script-aware BM25).
 * A chunk found by only one of them has `dense_score` or `lexical_score` null, and that is
 * exactly the case worth staring at -- it is the chunk the other arm would have missed
 * entirely. So the missing arm is drawn as an explicit dashed "not found" slot rather than
 * being left out, and single-arm chunks are flagged in the accent colour.
 *
 * This module also owns the citation-focus store. AnswerCard sets it when a citation is
 * clicked; this panel highlights the exact supporting span using the citation's
 * source_start/source_end offsets into the chunk text -- offsets, never a string search,
 * because the same quote can occur twice in one chunk.
 */

import { useEffect, useRef, useState, useSyncExternalStore } from 'react'

import { STAGE_COLORS } from '@/components/LatencyWaterfall'
import { ScrollArea } from '@/components/ui/scroll-area'
import type { PipelineResult, ScoredChunk } from '@/lib/types'
import { clamp, cn, langAttrs, langLabel } from '@/lib/utils'

// ---------------------------------------------------------------------------------------
// Citation focus: a two-component conversation the parent does not have to mediate
// ---------------------------------------------------------------------------------------

export interface CitationFocus {
  chunkId: string
  /** Character offsets into the chunk's own text. */
  sourceStart: number
  sourceEnd: number
  /** Bumped on every set, so clicking the same citation twice still re-scrolls to it. */
  nonce: number
}

let focusState: CitationFocus | null = null
let focusNonce = 0
const focusListeners = new Set<() => void>()

/** Highlight a citation's supporting span, or pass null to clear. */
export function setCitationFocus(next: Omit<CitationFocus, 'nonce'> | null): void {
  if (next === null && focusState === null) return
  focusState = next === null ? null : { ...next, nonce: ++focusNonce }
  for (const listener of focusListeners) listener()
}

function subscribeFocus(listener: () => void): () => void {
  focusListeners.add(listener)
  return () => {
    focusListeners.delete(listener)
  }
}

export function useCitationFocus(): CitationFocus | null {
  return useSyncExternalStore(
    subscribeFocus,
    () => focusState,
    () => null,
  )
}

// ---------------------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------------------

/** Longer than this and the chunk is clamped with a toggle; the cited span may sit past
 *  the clamp, so a focused chunk always expands regardless. */
const CLAMP_CHARS = 260

export interface RetrievedChunksProps {
  chunks?: readonly ScoredChunk[]
  /** Convenience: used when `chunks` is not passed. */
  result?: PipelineResult | null
  className?: string
}

export function RetrievedChunks({ chunks, result, className }: RetrievedChunksProps) {
  const scored: readonly ScoredChunk[] = chunks ?? result?.retrieved ?? []
  const focus = useCitationFocus()

  const [expanded, setExpanded] = useState<ReadonlySet<string>>(() => new Set())
  const cards = useRef(new Map<string, HTMLLIElement>())

  // Compared by value, not identity: a parent that rebuilds the array on every render
  // (`chunks={result?.retrieved ?? []}`) must not wipe the highlight a click just set.
  const signature = scored.map((s) => s.chunk.chunk_id).join('|')

  useEffect(() => {
    setExpanded(new Set())
    setCitationFocus(null)
  }, [signature])

  // A cited chunk opens itself and scrolls into view inside the panel.
  useEffect(() => {
    if (!focus) return
    setExpanded((prev) => (prev.has(focus.chunkId) ? prev : new Set(prev).add(focus.chunkId)))
    const node = cards.current.get(focus.chunkId)
    if (!node) return
    node.scrollIntoView({
      block: 'nearest',
      behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
    })
  }, [focus])

  const arms = countArms(scored)

  return (
    <section
      className={cn('flex min-h-0 flex-col gap-3', className)}
      aria-label="Retrieved context"
    >
      <header className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
        <div className="flex items-baseline gap-3">
          <span className="font-mono text-[11px] tracking-widest text-muted-foreground uppercase">
            retrieved context
          </span>
          <span className="font-mono text-sm text-foreground tabular-nums">{scored.length}</span>
        </div>
        {scored.length > 0 && (
          <div className="flex items-baseline gap-3 font-mono text-[10px] text-muted-foreground">
            <span>
              both arms <b className="font-medium text-foreground">{arms.both}</b>
            </span>
            <span className={cn(arms.denseOnly > 0 && 'text-primary')}>
              dense only <b className="font-medium">{arms.denseOnly}</b>
            </span>
            <span className={cn(arms.lexicalOnly > 0 && 'text-primary')}>
              bm25 only <b className="font-medium">{arms.lexicalOnly}</b>
            </span>
          </div>
        )}
      </header>

      {scored.length === 0 ? (
        <p className="rounded-md border border-dashed border-border px-3 py-6 text-center font-mono text-[11px] text-muted-foreground">
          no chunks retrieved yet
        </p>
      ) : (
        /* Radix's viewport wraps its children in an inline `display:table` div, which
           shrink-wraps to the widest chunk id and makes the panel scroll sideways. Only
           `!important` beats an inline style. */
        <ScrollArea className="min-h-0 flex-1 [&_[data-slot=scroll-area-viewport]>div]:block!">
          <ol className="flex flex-col gap-2 pe-3">
            {scored.map((item, index) => (
              <ChunkRow
                key={`${item.chunk.chunk_id}-${index}`}
                item={item}
                rank={index + 1}
                focus={focus && focus.chunkId === item.chunk.chunk_id ? focus : null}
                expanded={expanded.has(item.chunk.chunk_id)}
                onToggle={() =>
                  setExpanded((prev) => {
                    const next = new Set(prev)
                    if (next.has(item.chunk.chunk_id)) next.delete(item.chunk.chunk_id)
                    else next.add(item.chunk.chunk_id)
                    return next
                  })
                }
                registerCard={(node) => {
                  if (node) cards.current.set(item.chunk.chunk_id, node)
                  else cards.current.delete(item.chunk.chunk_id)
                }}
              />
            ))}
          </ol>
        </ScrollArea>
      )}
    </section>
  )
}

// ---------------------------------------------------------------------------------------
// One chunk
// ---------------------------------------------------------------------------------------

interface ChunkRowProps {
  item: ScoredChunk
  rank: number
  focus: CitationFocus | null
  expanded: boolean
  onToggle: () => void
  registerCard: (node: HTMLLIElement | null) => void
}

function ChunkRow({ item, rank, focus, expanded, onToggle, registerCard }: ChunkRowProps) {
  const chunk = item.chunk
  const text = chunk.text ?? ''
  const dense = numeric(item.dense_score)
  const lexical = numeric(item.lexical_score)
  const rerank = numeric(item.rerank_score)
  const onlyArm = dense !== null && lexical === null ? 'dense' : lexical !== null && dense === null ? 'bm25' : null

  const showAll = expanded || focus !== null
  const clampable = text.length > CLAMP_CHARS
  const parent = chunk.context_text && chunk.context_text !== text ? chunk.context_text : null

  return (
    <li
      ref={registerCard}
      className={cn(
        'rounded-md bg-card px-3 py-2.5 ring-1 ring-inset transition-colors',
        focus ? 'ring-primary/70' : 'ring-border',
      )}
    >
      <div className="flex min-w-0 items-baseline gap-2">
        <span className="font-mono text-[11px] text-muted-foreground tabular-nums">
          {String(rank).padStart(2, '0')}
        </span>
        <span
          className="min-w-0 flex-1 truncate font-mono text-[11px] text-foreground"
          title={`${chunk.chunk_id} · doc ${chunk.doc_id}`}
        >
          {chunk.chunk_id}
        </span>
        <span className="shrink-0 rounded-sm border border-border px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
          {chunk.strategy}
        </span>
        <span className="shrink-0 rounded-sm border border-border px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
          {langLabel(chunk.lang)}
        </span>
      </div>

      {chunk.title && (
        <p className="indic-text mt-1.5 text-[13px] text-muted-foreground" {...langAttrs(chunk.lang)}>
          {chunk.title}
        </p>
      )}

      <p
        className={cn('indic-text mt-2 text-[13px] text-foreground', !showAll && 'line-clamp-4')}
        {...langAttrs(chunk.lang)}
      >
        {focus ? withHighlight(text, focus.sourceStart, focus.sourceEnd) : text}
      </p>

      {clampable && (
        <button
          type="button"
          onClick={onToggle}
          className="mt-1.5 font-mono text-[10px] text-muted-foreground underline underline-offset-4 hover:text-foreground"
        >
          {showAll ? 'collapse' : 'show full chunk'}
        </button>
      )}

      {parent && (
        <details className="mt-2">
          <summary className="cursor-pointer font-mono text-[10px] text-muted-foreground marker:text-muted-foreground hover:text-foreground">
            parent context sent to the generator
          </summary>
          <p className="indic-text mt-1.5 text-[12px] text-muted-foreground" {...langAttrs(chunk.lang)}>
            {parent}
          </p>
        </details>
      )}

      {/* Retrieval provenance. Both arms are always drawn -- the empty one is the point. */}
      <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
        <ArmChip
          label="dense"
          color={STAGE_COLORS.retrieve_dense}
          score={dense}
          rank={item.dense_rank}
          digits={3}
        />
        <ArmChip
          label="bm25"
          color={STAGE_COLORS.retrieve_lexical}
          score={lexical}
          rank={item.lexical_rank}
          digits={2}
        />
        {onlyArm && (
          <span className="rounded-sm border border-primary/50 px-1.5 py-0.5 font-mono text-[10px] text-primary">
            {onlyArm} only
          </span>
        )}
        <span className="ms-auto flex items-baseline gap-3 font-mono text-[10px] text-muted-foreground">
          {rerank !== null && (
            <span>
              rerank <b className="font-medium text-foreground">{rerank.toFixed(3)}</b>
            </span>
          )}
          <span>
            fused <b className="font-medium text-foreground">{numeric(item.score)?.toFixed(3) ?? '--'}</b>
          </span>
        </span>
      </div>
    </li>
  )
}

interface ArmChipProps {
  label: string
  color: string
  score: number | null
  rank: number | null
  digits: number
}

function ArmChip({ label, color, score, rank, digits }: ArmChipProps) {
  const found = score !== null
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-sm border px-1.5 py-0.5 font-mono text-[10px]',
        found
          ? 'border-foreground/15 bg-foreground/5 text-foreground'
          : 'border-dashed border-border text-muted-foreground/60',
      )}
    >
      <i
        aria-hidden
        className="size-1.5 shrink-0 rounded-full"
        style={{ backgroundColor: color, opacity: found ? 1 : 0.25 }}
      />
      {label}
      {found ? (
        <>
          <b className="font-medium">{score.toFixed(digits)}</b>
          {rank !== null && rank > 0 && <span className="text-muted-foreground">#{rank}</span>}
        </>
      ) : (
        <span>not found</span>
      )}
    </span>
  )
}

// ---------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------

/** Splits the chunk text on the citation's offsets. Offsets are trusted but clamped: a
 *  generator can emit a span that runs past the text it cited, and a crash here would take
 *  the whole evidence panel down. */
function withHighlight(text: string, start: number, end: number) {
  const from = clamp(Math.trunc(start), 0, text.length)
  const to = clamp(Math.trunc(end), from, text.length)
  if (to <= from) return text
  return (
    <>
      {text.slice(0, from)}
      <mark className="rounded-sm bg-primary/25 px-0.5 text-foreground">{text.slice(from, to)}</mark>
      {text.slice(to)}
    </>
  )
}

function countArms(scored: readonly ScoredChunk[]) {
  let both = 0
  let denseOnly = 0
  let lexicalOnly = 0
  for (const item of scored) {
    const hasDense = numeric(item.dense_score) !== null
    const hasLexical = numeric(item.lexical_score) !== null
    if (hasDense && hasLexical) both += 1
    else if (hasDense) denseOnly += 1
    else if (hasLexical) lexicalOnly += 1
  }
  return { both, denseOnly, lexicalOnly }
}

function numeric(value: number | null | undefined): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

export default RetrievedChunks
