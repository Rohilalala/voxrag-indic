/**
 * Every guardrail verdict from the last request.
 *
 * Blocking rails come first because they are why an answer was refused, but the rails that
 * passed stay on screen too: the user should be able to see that the safety, injection and
 * grounding checks ran on a perfectly ordinary query, not only when one of them fires.
 */

import { ShieldAlert, ShieldCheck } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'
import type { GuardrailVerdict } from '@/lib/types'
import { cn } from '@/lib/utils'

/** Where each rail sits in the pipeline. Names come from voxrag.guardrails; an unknown name
 *  is shown without a stage rather than guessed at. */
const RAIL_STAGE: Record<string, 'input' | 'retrieval' | 'output'> = {
  empty_query: 'input',
  language: 'input',
  safety: 'input',
  prompt_injection: 'input',
  personal_data: 'input',
  // These three run AFTER retrieval, not before it: they need the candidates to judge.
  // 'off_topic' was labelled 'input' while the live trace shows it post-retrieval, and the
  // other two were absent entirely, so they rendered with no stage badge at all.
  off_topic: 'retrieval',
  entity_grounding: 'retrieval',
  cross_language: 'retrieval',
  groundedness: 'output',
  citations: 'output',
  refusal_consistency: 'output',
}

export interface GuardrailPanelProps {
  /** `result.guardrails` from the last PipelineResult. Empty before the first request. */
  verdicts: GuardrailVerdict[]
  /** True while a request is in flight, so the panel says "running" instead of "none yet". */
  running?: boolean
  className?: string
}

export function GuardrailPanel({ verdicts, running = false, className }: GuardrailPanelProps) {
  const blocking = verdicts.filter((verdict) => !verdict.allowed)
  const passed = verdicts.filter((verdict) => verdict.allowed)
  const ordered = [...blocking, ...passed]

  // An input rail that blocked short-circuits the pipeline, so the output rails never ran.
  // Saying so is more honest than listing rails the deployment may not even have.
  const shortCircuit = blocking.find((verdict) => RAIL_STAGE[verdict.name] === 'input')

  return (
    <Card className={cn('gap-3 overflow-hidden', className)}>
      <CardHeader className="grid-cols-[1fr_auto] items-center gap-2">
        <CardTitle className="font-mono text-[11px] tracking-widest text-muted-foreground uppercase">
          guardrails
        </CardTitle>
        <span className="font-mono text-[11px] text-muted-foreground">
          {verdicts.length === 0
            ? running
              ? 'running'
              : 'idle'
            : blocking.length > 0
              ? `${blocking.length} blocked · ${passed.length} passed`
              : `${passed.length} passed`}
        </span>
      </CardHeader>

      <CardContent className="min-h-0 flex-1 px-0">
        {verdicts.length === 0 ? (
          <p className="px-(--card-spacing) text-xs leading-relaxed text-muted-foreground">
            {running
              ? 'Input rails run before retrieval; output rails run on the generated answer.'
              : 'No verdicts yet. Input rails (empty query, language, safety, prompt injection, off-topic) run before retrieval; output rails (groundedness, citations, refusal consistency) run on the answer.'}
          </p>
        ) : (
          <ScrollArea className="h-full max-h-full">
            <ul className="flex flex-col gap-1.5 px-(--card-spacing)">
              {ordered.map((verdict, index) => {
                const stage = RAIL_STAGE[verdict.name]
                return (
                  <li
                    // Indexed: a rail can legitimately report twice in one run (an input
                    // rail that also runs on the rewritten query), and a name-only key
                    // would silently drop the second verdict.
                    key={`${verdict.name}-${index}`}
                    className={cn(
                      'flex flex-col gap-1 border-s-2 py-1.5 ps-2.5',
                      verdict.allowed ? 'border-border' : 'border-destructive',
                    )}
                  >
                    <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                      {verdict.allowed ? (
                        <ShieldCheck className="size-3.5 shrink-0 text-muted-foreground" aria-hidden="true" />
                      ) : (
                        <ShieldAlert className="size-3.5 shrink-0 text-destructive" aria-hidden="true" />
                      )}
                      <span
                        className={cn(
                          'font-mono text-xs',
                          verdict.allowed ? 'text-foreground' : 'text-destructive',
                        )}
                      >
                        {verdict.name}
                      </span>
                      {stage ? (
                        <span className="font-mono text-[10px] text-muted-foreground">{stage}</span>
                      ) : null}
                      <span className="ms-auto flex items-center gap-2">
                        {Number.isFinite(verdict.score) && verdict.score > 0 ? (
                          <span className="font-mono text-[10px] tabular-nums text-muted-foreground">
                            {verdict.score.toFixed(2)}
                          </span>
                        ) : null}
                        {verdict.allowed ? null : (
                          <Badge variant="destructive" className="font-mono text-[10px]">
                            {/* The refusal reason usually repeats the rail's own name; in a
                                column this narrow, saying it twice costs the detail line. */}
                            {verdict.reason && verdict.reason !== verdict.name
                              ? verdict.reason
                              : 'blocked'}
                          </Badge>
                        )}
                      </span>
                    </div>
                    {verdict.detail ? (
                      <p
                        className={cn(
                          'text-[11px] leading-snug',
                          verdict.allowed ? 'text-muted-foreground' : 'text-destructive/90',
                        )}
                      >
                        {verdict.detail}
                      </p>
                    ) : null}
                  </li>
                )
              })}
            </ul>
            {shortCircuit ? (
              <p className="mt-2 px-(--card-spacing) text-[11px] leading-snug text-muted-foreground">
                The <span className="font-mono">{shortCircuit.name}</span> rail blocked before
                retrieval, so the output rails never ran on this request.
              </p>
            ) : null}
          </ScrollArea>
        )}
      </CardContent>
    </Card>
  )
}

export default GuardrailPanel
