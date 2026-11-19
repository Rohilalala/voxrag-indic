/**
 * The input surface: push-to-talk mic, live meter, language, generation mode, and a text
 * box that reaches the identical pipeline without a microphone.
 *
 * The text box is not a fallback afterthought. A user whose mic is denied, missing, or
 * blocked by an insecure origin still has to be able to try the system in the ten seconds
 * they will give it, so both routes are equally prominent and always both visible.
 *
 * Voice goes over the websocket while it is open, because that is what streams the stage
 * timings and the transcript as they happen. The moment the socket is not open the same
 * clip goes to POST /api/voice instead -- one answer either way, never two.
 */

import { useEffect, useImperativeHandle, useRef, useState, type Ref } from 'react'
import { LoaderCircle, Mic, Send, TriangleAlert } from 'lucide-react'

import { LanguagePicker } from '@/components/LanguagePicker'
import { WaveformMeter } from '@/components/WaveformMeter'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { useRecorder, type RecordedClip } from '@/hooks/useRecorder'
import {
  VoiceSocket,
  query as postQuery,
  voice as postVoice,
  type SocketStatus,
} from '@/lib/api'
import {
  LANGS,
  type ApiError,
  type GenerationMode,
  type Lang,
  type PipelineResult,
  type WsStageFrame,
} from '@/lib/types'
import { cn, langAttrs, langLabel } from '@/lib/utils'

/** Matches the REST client's own deadline. A websocket that simply stops answering would
 *  otherwise leave every control disabled for the rest of the demo. */
const WS_WATCHDOG_MS = 90_000

const KNOWN_LANGS: ReadonlySet<string> = new Set<string>([...LANGS, 'en'])

/** The `transcript` frame carries a bare string, empty when the provider reported nothing. */
function asLang(value: string | null | undefined): Lang | null {
  return value && KNOWN_LANGS.has(value) ? (value as Lang) : null
}

interface TranscriptView {
  text: string
  lang: Lang | null
  source: 'typed' | 'voice'
}

/** Imperative entry point so the shell can fire an example query into this panel. */
export interface VoicePanelHandle {
  ask: (text: string, lang?: Lang | null) => void
  focusInput: () => void
}

export interface VoicePanelProps {
  ref?: Ref<VoicePanelHandle>
  className?: string
  /** Hard-disables submission, e.g. while the backend reports the index is not loaded. */
  disabled?: boolean
  onRunStart?: (source: 'voice' | 'text') => void
  /** Streams only on the websocket path, in pipeline order, as each stage finishes. */
  onStage?: (frame: WsStageFrame) => void
  onTranscript?: (text: string, lang: Lang | null) => void
  /** Quality mode only: one incrementally generated token. */
  onToken?: (text: string) => void
  onResult?: (result: PipelineResult) => void
  onError?: (error: ApiError) => void
  onBusyChange?: (busy: boolean) => void
  onSocketStatus?: (status: SocketStatus) => void
}

export function VoicePanel(props: VoicePanelProps) {
  const { ref, className, disabled = false } = props

  const [lang, setLang] = useState<Lang | null>(null)
  const [mode, setMode] = useState<GenerationMode>('fast')
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [transcript, setTranscript] = useState<TranscriptView | null>(null)
  const [apiError, setApiError] = useState<ApiError | null>(null)
  const [socketStatus, setSocketStatus] = useState<SocketStatus>('idle')

  const inputRef = useRef<HTMLInputElement>(null)
  /** Every run gets a number; a late answer from a superseded run is dropped. */
  const runRef = useRef(0)
  /** The run currently waiting on the socket, or null when nothing is. */
  const wsRunRef = useRef<number | null>(null)
  const watchdogRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const busyRef = useRef(false)
  const socketRef = useRef<VoiceSocket | null>(null)

  // The socket handlers are bound once for the life of the panel, so they read the
  // callbacks through a ref instead of closing over one render's props.
  const propsRef = useRef(props)
  propsRef.current = props

  const clearWatchdog = () => {
    if (watchdogRef.current !== null) {
      clearTimeout(watchdogRef.current)
      watchdogRef.current = null
    }
  }

  const setBusyState = (next: boolean) => {
    busyRef.current = next
    setBusy(next)
    propsRef.current.onBusyChange?.(next)
  }

  const begin = (source: 'voice' | 'text'): number => {
    const id = runRef.current + 1
    runRef.current = id
    setApiError(null)
    setBusyState(true)
    propsRef.current.onRunStart?.(source)
    return id
  }

  const succeed = (id: number, result: PipelineResult) => {
    if (runRef.current !== id) return
    wsRunRef.current = null
    clearWatchdog()
    setBusyState(false)
    // The POST path has no streaming transcript frame, so this is where voice text lands.
    if (result.transcript?.text) {
      setTranscript({
        text: result.transcript.text,
        lang: result.transcript.lang ?? result.detected_lang,
        source: 'voice',
      })
    }
    propsRef.current.onResult?.(result)
  }

  const fail = (id: number, error: ApiError) => {
    if (runRef.current !== id) return
    wsRunRef.current = null
    clearWatchdog()
    setBusyState(false)
    setApiError(error)
    propsRef.current.onError?.(error)
  }

  // -- the socket ------------------------------------------------------------------------

  useEffect(() => {
    const socket = new VoiceSocket({
      onStage: (frame) => propsRef.current.onStage?.(frame),
      onTranscript: (frame) => {
        if (wsRunRef.current === null) return
        const detected = asLang(frame.lang)
        setTranscript({ text: frame.text, lang: detected, source: 'voice' })
        propsRef.current.onTranscript?.(frame.text, detected)
      },
      onToken: (frame) => propsRef.current.onToken?.(frame.text),
      onResult: (result) => {
        const id = wsRunRef.current
        if (id === null) return
        succeed(id, result)
      },
      onError: (error) => {
        const id = wsRunRef.current
        // A drop while nothing is in flight is the socket's own business: it reconnects,
        // and voice still works over POST. Only a live run needs to be told.
        if (id === null) return
        fail(id, error)
      },
      onStatusChange: (status) => {
        setSocketStatus(status)
        propsRef.current.onSocketStatus?.(status)
      },
    })
    socketRef.current = socket
    socket.connect()
    return () => {
      socket.close()
      socketRef.current = null
      clearWatchdog()
    }
    // Bound once for the life of the panel; everything variable is read through a ref.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // -- submission ------------------------------------------------------------------------

  const locked = busy || disabled

  const sendClip = (clip: RecordedClip) => {
    if (busyRef.current || disabled) return
    const id = begin('voice')
    setTranscript(null) // never leave the previous question on screen under a new answer
    const socket = socketRef.current
    if (socket?.isOpen) {
      wsRunRef.current = id
      watchdogRef.current = setTimeout(() => {
        fail(id, {
          kind: 'timeout',
          message: 'the live connection went quiet; press again to retry over POST /api/voice',
        })
      }, WS_WATCHDOG_MS)
      // Not awaited: the answer arrives as a frame, not as a return value.
      void socket.send(clip.blob, { lang, mode, mime: clip.mime })
      return
    }
    void postVoice({ audio: clip.blob, lang, mode, mime: clip.mime }).then((result) => {
      if (result.ok) succeed(id, result.data)
      else fail(id, result.error)
    })
  }

  const submitText = (raw: string, forced?: Lang | null) => {
    const value = raw.trim()
    if (!value || busyRef.current || disabled) return
    const chosen = forced === undefined ? lang : forced
    const id = begin('text')
    setTranscript({ text: value, lang: chosen, source: 'typed' })
    void postQuery({ text: value, lang: chosen, mode }).then((result) => {
      if (result.ok) succeed(id, result.data)
      else fail(id, result.error)
    })
  }

  const recorder = useRecorder({ onClip: sendClip })

  useImperativeHandle(ref, () => ({
    ask: (value: string, forced?: Lang | null) => {
      setText(value)
      if (forced !== undefined) setLang(forced)
      submitText(value, forced)
    },
    focusInput: () => inputRef.current?.focus(),
  }))

  // -- push-to-talk ----------------------------------------------------------------------

  const lockedRef = useRef(locked)
  lockedRef.current = locked || !recorder.available
  const recordingRef = useRef(false)
  recordingRef.current = recorder.isRecording
  const startRef = useRef(recorder.start)
  startRef.current = recorder.start
  const stopRef = useRef(recorder.stop)
  stopRef.current = recorder.stop

  useEffect(() => {
    const typing = (target: EventTarget | null): boolean => {
      const element = target as HTMLElement | null
      if (!element || typeof element.closest !== 'function') return false
      const tag = element.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true
      if (element.isContentEditable) return true
      // Space is how Radix's Select opens and commits, so the picker keeps its own key.
      return element.closest('[role="combobox"],[data-slot="select-content"]') !== null
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && recordingRef.current) {
        stopRef.current(true) // discard: an escape is a retraction, not a question
        return
      }
      if (event.code !== 'Space' || event.repeat || typing(event.target)) return
      if (lockedRef.current) return
      // Also suppresses the click a focused button would otherwise fire on keyup.
      event.preventDefault()
      startRef.current()
    }
    const onKeyUp = (event: KeyboardEvent) => {
      if (event.code !== 'Space' || typing(event.target)) return
      event.preventDefault()
      stopRef.current(false)
    }
    // A keyup lost to alt-tab mid-hold would otherwise record until the duration ceiling.
    const onBlur = () => stopRef.current(false)

    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('keyup', onKeyUp)
    window.addEventListener('blur', onBlur)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('keyup', onKeyUp)
      window.removeEventListener('blur', onBlur)
    }
  }, [])

  // -- readouts --------------------------------------------------------------------------

  const micDisabled = locked || !recorder.available
  const seconds = (recorder.elapsedMs / 1000).toFixed(1)
  const codec = recorder.mimeType ? recorder.mimeType.split(';')[0] : ''

  const micStatus = recorder.isRecording
    ? `recording ${seconds}s`
    : recorder.isArming
      ? 'opening the microphone'
      : busy
        ? 'waiting for the answer'
        : !recorder.available
          ? 'microphone unavailable'
          : recorder.ready
            ? `mic live${codec ? ` · ${codec}` : ''}`
            : 'hold to talk'

  const streamLabel =
    socketStatus === 'open'
      ? 'ws live'
      : socketStatus === 'connecting' || socketStatus === 'reconnecting'
        ? 'ws connecting'
        : 'ws offline · POST fallback'

  return (
    <section
      className={cn('rounded-lg border border-border bg-card', className)}
      aria-label="Ask the pipeline"
    >
      {/* Controls that describe the next run, not the current one, so they stay live. */}
      <div className="flex flex-wrap items-center gap-2 border-b border-border px-4 py-2.5">
        <LanguagePicker value={lang} onChange={setLang} id="voxrag-lang" />
        <ToggleGroup
          type="single"
          value={mode}
          spacing={0}
          variant="outline"
          onValueChange={(next) => {
            // Radix hands back "" when the pressed item is toggled off; a mode is required.
            if (next === 'fast' || next === 'quality') setMode(next)
          }}
          aria-label="Generation mode"
          className="h-9"
        >
          <ToggleGroupItem value="fast" className="h-9 px-3 text-xs" aria-label="Fast mode">
            fast
          </ToggleGroupItem>
          <ToggleGroupItem value="quality" className="h-9 px-3 text-xs" aria-label="Quality mode">
            quality
          </ToggleGroupItem>
        </ToggleGroup>
        <div className="ms-auto flex items-center gap-2">
          <Badge
            variant="outline"
            className="font-mono text-[0.7rem] text-muted-foreground"
            title="Voice streams over the websocket when it is open, and falls back to POST /api/voice when it is not."
          >
            {streamLabel}
          </Badge>
        </div>
      </div>

      {/* Mic + meter. */}
      <div className="flex items-center gap-4 px-4 py-4">
        <button
          type="button"
          disabled={micDisabled}
          aria-label={recorder.isRecording ? 'Release to send' : 'Hold to record'}
          aria-pressed={recorder.isRecording}
          data-recording={recorder.isRecording ? 'true' : 'false'}
          onPointerEnter={() => recorder.warmUp()}
          onPointerDown={(event) => {
            if (event.pointerType === 'mouse' && event.button !== 0) return
            event.preventDefault()
            event.currentTarget.setPointerCapture?.(event.pointerId)
            recorder.start()
          }}
          onPointerUp={(event) => {
            event.preventDefault()
            recorder.stop(false)
          }}
          onPointerCancel={() => recorder.stop(true)}
          onContextMenu={(event) => event.preventDefault()}
          onKeyDown={(event) => {
            // Enter toggles, for anyone who cannot hold a pointer down.
            if (event.key !== 'Enter') return
            event.preventDefault()
            if (recorder.isRecording) recorder.stop(false)
            else recorder.start()
          }}
          className={cn(
            'relative flex size-[76px] shrink-0 touch-none items-center justify-center rounded-full border text-sm transition-colors outline-none select-none',
            'focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50',
            'disabled:opacity-40',
            recorder.isRecording
              ? 'border-primary bg-primary/15 text-primary'
              : 'border-border bg-secondary text-foreground hover:bg-secondary/70',
          )}
        >
          {/* Ring scaled by the live RMS. Real signal only -- it is motionless in silence. */}
          <span
            aria-hidden="true"
            className="pointer-events-none absolute inset-0 rounded-full border border-primary"
            style={{
              transform: `scale(${1 + recorder.level * 0.28})`,
              opacity: recorder.isRecording ? 0.15 + recorder.level * 0.65 : 0,
            }}
          />
          {busy ? (
            <LoaderCircle className="size-6 animate-spin text-muted-foreground" />
          ) : (
            <Mic className="size-6" />
          )}
        </button>

        <div className="flex min-w-0 flex-1 flex-col gap-2">
          <div className="h-12 rounded-md border border-border/60 bg-background px-2 py-1">
            <WaveformMeter analyser={recorder.analyser} active={recorder.isRecording} />
          </div>
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 font-mono text-xs">
            <span
              className={cn(
                'tabular-nums',
                recorder.isRecording ? 'text-primary' : 'text-muted-foreground',
              )}
            >
              {micStatus}
            </span>
            <span className="text-muted-foreground/70">
              hold the button or the space bar · esc discards
            </span>
            {recorder.notice ? <span className="text-warning">{recorder.notice}</span> : null}
          </div>
        </div>
      </div>

      {/* The route that needs no microphone, deliberately not hidden behind the mic. */}
      <form
        className="flex items-center gap-2 border-t border-border px-4 py-3"
        onSubmit={(event) => {
          event.preventDefault()
          submitText(text)
        }}
      >
        <input
          ref={inputRef}
          type="text"
          value={text}
          onChange={(event) => setText(event.target.value)}
          disabled={locked}
          // `auto` so a typed Urdu question renders right-to-left as it is entered.
          dir="auto"
          autoComplete="off"
          spellCheck={false}
          aria-label="Type a question"
          placeholder="Type a question in any of the 14 languages"
          className="h-9 w-full min-w-0 rounded-lg border border-input bg-input/30 px-3 text-sm outline-none placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 disabled:opacity-50"
        />
        <Button type="submit" size="lg" disabled={locked || text.trim().length === 0}>
          {busy ? (
            <LoaderCircle className="animate-spin" data-icon="inline-start" />
          ) : (
            <Send data-icon="inline-start" />
          )}
          {busy ? 'running' : 'ask'}
        </Button>
      </form>

      {/* Transcript. Fixed height: this fills in mid-run and must not shove the answer. */}
      <div className="min-h-[3.5rem] border-t border-border px-4 py-2.5">
        <div className="flex items-baseline gap-2">
          <span className="font-mono text-[0.7rem] tracking-widest text-muted-foreground uppercase">
            {transcript?.source === 'typed' ? 'query' : 'transcript'}
          </span>
          {transcript?.lang ? (
            <span className="font-mono text-[0.7rem] text-muted-foreground">
              {langLabel(transcript.lang)}
            </span>
          ) : null}
          {busy ? (
            <span className="font-mono text-[0.7rem] text-primary">
              {mode} pipeline running
            </span>
          ) : null}
        </div>
        {transcript ? (
          <p className="indic-text text-sm" {...langAttrs(transcript.lang)}>
            {transcript.text}
          </p>
        ) : (
          <p className="text-sm text-muted-foreground">
            Nothing asked yet. Hold the mic, or type a question.
          </p>
        )}
      </div>

      {(recorder.error || apiError) && (
        <div className="space-y-2 border-t border-border px-4 py-3">
          {recorder.error ? (
            <Alert>
              <TriangleAlert className="text-warning" />
              <AlertTitle>{recorder.error.message}</AlertTitle>
              <AlertDescription>{recorder.error.hint}</AlertDescription>
            </Alert>
          ) : null}
          {apiError ? (
            <Alert variant="destructive">
              <TriangleAlert />
              <AlertTitle>The request failed</AlertTitle>
              <AlertDescription>
                {apiError.message}
                {apiError.requestId ? (
                  <span className="ms-1 font-mono text-xs">({apiError.requestId})</span>
                ) : null}
              </AlertDescription>
            </Alert>
          ) : null}
        </div>
      )}
    </section>
  )
}
