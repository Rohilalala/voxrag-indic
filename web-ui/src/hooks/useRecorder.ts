/**
 * Push-to-talk microphone capture.
 *
 * Hold to record, release to send: `start()` on pointer-down, `stop()` on pointer-up, and
 * the finished clip arrives through `onClip` with the MIME type the browser actually
 * produced -- never an asserted one, because Chrome gives WebM/Opus and Safari gives MP4/AAC
 * and the backend sniffs what it is handed.
 *
 * Everything that can fail live has its own message. The one that matters most is
 * `insecure-context`: off HTTPS and off localhost the browser does not expose
 * `getUserMedia` at all, so nothing appears broken -- the mic button simply never does
 * anything. That is reported before the user presses it, not after.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

export type RecorderStatus = 'idle' | 'arming' | 'recording'

export type RecorderErrorKind =
  | 'insecure-context'
  | 'unsupported'
  | 'permission-denied'
  | 'no-microphone'
  | 'device-busy'
  | 'device-error'
  | 'recorder-failed'

export interface RecorderError {
  kind: RecorderErrorKind
  /** One line, safe to render as a heading. */
  message: string
  /** What to do about it. Always ends by pointing at the text input, which needs no mic. */
  hint: string
}

export interface RecordedClip {
  blob: Blob
  /** The container the browser really wrote, e.g. `audio/webm;codecs=opus`. */
  mime: string
  durationMs: number
}

export interface UseRecorderOptions {
  /** Fires once per completed hold, after the recorder has flushed its last chunk. */
  onClip?: (clip: RecordedClip) => void
  /** Presses shorter than this are treated as mis-clicks and discarded. */
  minMs?: number
  /** Hard ceiling, so a stuck key cannot record until the tab runs out of memory. */
  maxMs?: number
}

export interface Recorder {
  status: RecorderStatus
  isRecording: boolean
  /** getUserMedia is in flight -- the very first press pays 100-300ms here. */
  isArming: boolean
  /** Smoothed RMS of the live input, 0-1. Published at ~15Hz; the canvas meter reads the
   *  analyser directly at frame rate instead of re-rendering React. */
  level: number
  elapsedMs: number
  /** Live while the mic is held open; null before the first successful acquisition. */
  analyser: AnalyserNode | null
  /** The negotiated container, for display. Empty until the mic is acquired. */
  mimeType: string
  /** The most recent accepted recording. */
  clip: RecordedClip | null
  error: RecorderError | null
  /** Transient, non-fatal feedback (a press too short to be speech). */
  notice: string | null
  /** False when this browser/origin can never record: the button should say why, not sulk. */
  available: boolean
  /** The mic stream is open and warm. */
  ready: boolean
  start: () => void
  /** `cancel` discards the audio instead of emitting it. */
  stop: (cancel?: boolean) => void
  /** Acquire the mic ahead of the first press (call on hover) so no syllable is clipped. */
  warmUp: () => void
}

/** Preference order. WebM/Opus first because it is the smallest thing every Chromium and
 *  Firefox produces; MP4/AAC is Safari's only option. Whatever wins, the recorder's own
 *  `mimeType` is what gets sent. */
const CANDIDATE_MIMES = [
  'audio/webm;codecs=opus',
  'audio/webm',
  'audio/ogg;codecs=opus',
  'audio/mp4;codecs=mp4a.40.2',
  'audio/mp4',
  'audio/aac',
  'audio/wav',
] as const

/** Below this a "recording" is keyboard noise, not a question. */
const MIN_BYTES = 512

function pickMime(): string {
  if (typeof MediaRecorder === 'undefined' || typeof MediaRecorder.isTypeSupported !== 'function') {
    return ''
  }
  for (const mime of CANDIDATE_MIMES) {
    if (MediaRecorder.isTypeSupported(mime)) return mime
  }
  // Empty string means "let the browser choose"; `recorder.mimeType` still reports the truth.
  return ''
}

/** Can this page record at all? Runs before any user gesture so the UI can say why not. */
function probeCapability(): RecorderError | null {
  if (typeof window === 'undefined') return null
  if (!window.isSecureContext) {
    return {
      kind: 'insecure-context',
      message: 'Microphone blocked: this page is not on HTTPS or localhost',
      hint:
        'Browsers only expose getUserMedia in a secure context, so the mic API is missing ' +
        'entirely here -- nothing looks broken, it just never fires. Open the app over ' +
        'https:// or at http://localhost. The text box runs the identical pipeline minus STT.',
    }
  }
  if (typeof MediaRecorder === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
    return {
      kind: 'unsupported',
      message: 'This browser cannot record audio',
      hint: 'MediaRecorder is unavailable. Use the text box -- it exercises the same pipeline without the STT stage.',
    }
  }
  return null
}

/** Map a getUserMedia rejection onto the thing the person in front of the laptop must do. */
function classifyMicError(cause: unknown): RecorderError {
  const name = cause instanceof DOMException ? cause.name : ''
  const detail = cause instanceof Error && cause.message ? cause.message : 'unknown device error'
  switch (name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return {
        kind: 'permission-denied',
        message: 'Microphone permission denied',
        hint: 'Allow mic access from the icon in the address bar and press again, or type the question instead.',
      }
    case 'NotFoundError':
    case 'DevicesNotFoundError':
    case 'OverconstrainedError':
      return {
        kind: 'no-microphone',
        message: 'No microphone found',
        hint: 'This machine reports no audio input device. Plug one in, or use the text box.',
      }
    case 'NotReadableError':
    case 'TrackStartError':
      return {
        kind: 'device-busy',
        message: 'The microphone is held by another application',
        hint: `The OS refused the device (${detail}). Quit whatever else is recording, then press again.`,
      }
    default:
      return {
        kind: 'device-error',
        message: 'The microphone could not be opened',
        hint: `${detail}. Use the text box -- it runs the same pipeline without STT.`,
      }
  }
}

export function useRecorder(options: UseRecorderOptions = {}): Recorder {
  const { minMs = 300, maxMs = 15_000 } = options

  const [status, setStatus] = useState<RecorderStatus>('idle')
  const [level, setLevel] = useState(0)
  const [elapsedMs, setElapsedMs] = useState(0)
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null)
  const [mimeType, setMimeType] = useState('')
  const [clip, setClip] = useState<RecordedClip | null>(null)
  const [error, setError] = useState<RecorderError | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [ready, setReady] = useState(false)
  // Probed once per mount: the answer cannot change without a reload.
  const [capability] = useState<RecorderError | null>(() => probeCapability())

  const onClipRef = useRef(options.onClip)
  onClipRef.current = options.onClip

  const streamRef = useRef<MediaStream | null>(null)
  const contextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  // Explicit buffer parameter: the DOM types reject the SharedArrayBuffer-backed default.
  const bufferRef = useRef<Uint8Array<ArrayBuffer> | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const acquireRef = useRef<Promise<boolean> | null>(null)
  const rafRef = useRef(0)
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const startedAtRef = useRef(0)
  const cancelRef = useRef(false)
  /** True between pointer-down and pointer-up. A press can be released before
   *  getUserMedia resolves; without this the recorder would start after the button was
   *  already up and run to `maxMs` with the mic apparently stuck on. */
  const wantRef = useRef(false)
  const statusRef = useRef<RecorderStatus>('idle')
  /** Latched only for failures a retry cannot fix. A busy device can be freed by quitting
   *  the other app, so that one stays retryable. */
  const deniedRef = useRef(false)
  const mountedRef = useRef(true)
  /** `start` schedules the max-duration guard, which has to call `stop` -- defined below it. */
  const stopRef = useRef<(cancel?: boolean) => void>(() => undefined)

  const setPhase = useCallback((next: RecorderStatus) => {
    statusRef.current = next
    setStatus(next)
  }, [])

  // Surface an unusable origin/browser immediately, before anything is pressed.
  useEffect(() => {
    if (capability) setError(capability)
  }, [capability])

  const stopMeter = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
    rafRef.current = 0
    setLevel(0)
  }, [])

  /** One rAF loop while recording. RMS lands in React state only when it moves and at most
   *  ~15 times a second: the mic ring has to visibly react, but a 60Hz setState would
   *  re-render the panel on every frame of the demo. */
  const startMeter = useCallback(() => {
    const node = analyserRef.current
    const buffer = bufferRef.current
    if (!node || !buffer) return
    let published = -1
    let publishedAt = 0
    const frame = () => {
      node.getByteTimeDomainData(buffer)
      let sumSquares = 0
      for (let i = 0; i < buffer.length; i += 2) {
        const v = ((buffer[i] ?? 128) - 128) / 128
        sumSquares += v * v
      }
      const rms = Math.sqrt(sumSquares / (buffer.length / 2))
      // sqrt-compressed: conversational speech sits around 0.05 RMS, which would be
      // invisible on a linear scale.
      const next = Math.min(1, Math.sqrt(rms) * 2.2)
      const now = performance.now()
      if (Math.abs(next - published) > 0.02 && now - publishedAt > 66) {
        published = next
        publishedAt = now
        setLevel(next)
      }
      rafRef.current = requestAnimationFrame(frame)
    }
    rafRef.current = requestAnimationFrame(frame)
  }, [])

  const acquire = useCallback(async (): Promise<boolean> => {
    if (streamRef.current) return true
    if (capability || deniedRef.current) return false

    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })
    } catch (cause) {
      const failure = classifyMicError(cause)
      deniedRef.current =
        failure.kind === 'permission-denied' || failure.kind === 'no-microphone'
      if (mountedRef.current) setError(failure)
      return false
    }

    if (!mountedRef.current) {
      // Unmounted while the permission prompt was open: do not leave the tab's mic light on.
      for (const track of stream.getTracks()) track.stop()
      return false
    }

    streamRef.current = stream
    const Ctor =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    if (Ctor) {
      const context = new Ctor()
      const node = context.createAnalyser()
      node.fftSize = 1024
      node.smoothingTimeConstant = 0.55
      context.createMediaStreamSource(stream).connect(node)
      contextRef.current = context
      analyserRef.current = node
      // Allocated once: reading this every frame would otherwise show up as GC jitter in
      // the waveform, which is exactly the thing that has to look smooth.
      bufferRef.current = new Uint8Array(node.fftSize)
      setAnalyser(node)
    }

    const mime = pickMime()
    setMimeType(mime)
    setError(null)
    setReady(true)
    return true
  }, [capability])

  /** Memoised so the hover warm-up and the first press cannot open two streams. */
  const ensure = useCallback((): Promise<boolean> => {
    if (streamRef.current) return Promise.resolve(true)
    if (!acquireRef.current) {
      acquireRef.current = acquire().finally(() => {
        acquireRef.current = null
      })
    }
    return acquireRef.current
  }, [acquire])

  const warmUp = useCallback(() => {
    void ensure()
  }, [ensure])

  const start = useCallback(() => {
    if (statusRef.current !== 'idle') return
    if (capability) return
    wantRef.current = true
    setNotice(null)
    setPhase('arming')

    void (async () => {
      const ok = await ensure()
      // Released (or unmounted) before the mic opened: stay idle rather than recording
      // into a button nobody is holding.
      if (!ok || !wantRef.current || !mountedRef.current) {
        if (mountedRef.current && statusRef.current === 'arming') setPhase('idle')
        return
      }
      const context = contextRef.current
      // Autoplay policy suspends a context created outside a gesture; without this the
      // analyser reports pure silence and the meter never moves.
      if (context?.state === 'suspended') {
        try {
          await context.resume()
        } catch {
          /* the meter degrades, the recording does not */
        }
      }
      if (!wantRef.current || !mountedRef.current) {
        if (statusRef.current === 'arming') setPhase('idle')
        return
      }

      const stream = streamRef.current
      if (!stream) {
        setPhase('idle')
        return
      }

      chunksRef.current = []
      cancelRef.current = false
      let recorder: MediaRecorder
      try {
        const mime = pickMime()
        recorder = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream)
      } catch (cause) {
        setPhase('idle')
        setError({
          kind: 'recorder-failed',
          message: 'The recorder refused to start',
          hint: `${cause instanceof Error ? cause.message : String(cause)}. Use the text box instead.`,
        })
        return
      }

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) chunksRef.current.push(event.data)
      }
      recorder.onstop = () => {
        // The recorder is the authority on the container it wrote; the preference list is
        // only a request, and Safari in particular hands back something else.
        const mime = recorder.mimeType || pickMime() || 'audio/webm'
        const blob = new Blob(chunksRef.current, { type: mime })
        chunksRef.current = []
        const durationMs = performance.now() - startedAtRef.current
        if (!mountedRef.current) return
        if (cancelRef.current) {
          setNotice('Recording discarded.')
          return
        }
        if (durationMs < minMs || blob.size < MIN_BYTES) {
          setNotice('Too short - hold the button while you speak.')
          return
        }
        const recorded: RecordedClip = { blob, mime, durationMs }
        setClip(recorded)
        onClipRef.current?.(recorded)
      }
      // No timeslice: one flush at stop. Chunked MP4 from Safari is fragmented, and the
      // single-blob path is the one the backend has been exercised against.
      recorder.start()

      recorderRef.current = recorder
      startedAtRef.current = performance.now()
      setElapsedMs(0)
      setPhase('recording')
      setMimeType(recorder.mimeType || pickMime())
      startMeter()

      if (tickRef.current) clearInterval(tickRef.current)
      tickRef.current = setInterval(() => {
        const held = performance.now() - startedAtRef.current
        setElapsedMs(held)
        if (held >= maxMs) stopRef.current(false)
      }, 100)
    })()
  }, [capability, ensure, maxMs, minMs, setPhase, startMeter])

  const stop = useCallback(
    (cancel = false) => {
      // Also aborts a start still waiting on getUserMedia.
      wantRef.current = false
      if (statusRef.current !== 'recording') {
        if (statusRef.current === 'arming') setPhase('idle')
        return
      }
      cancelRef.current = cancel
      setPhase('idle')
      setElapsedMs(0)
      if (tickRef.current) {
        clearInterval(tickRef.current)
        tickRef.current = null
      }
      stopMeter()
      try {
        recorderRef.current?.stop()
      } catch {
        /* already stopped by the max-duration guard */
      }
    },
    [setPhase, stopMeter],
  )

  stopRef.current = stop

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      wantRef.current = false
      if (tickRef.current) clearInterval(tickRef.current)
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      try {
        if (recorderRef.current && recorderRef.current.state !== 'inactive') {
          recorderRef.current.stop()
        }
      } catch {
        /* nothing left to stop */
      }
      // Release the device so the browser's recording indicator goes out with the page.
      for (const track of streamRef.current?.getTracks() ?? []) track.stop()
      streamRef.current = null
      void contextRef.current?.close().catch(() => undefined)
      contextRef.current = null
      analyserRef.current = null
      bufferRef.current = null
    }
  }, [])

  return {
    status,
    isRecording: status === 'recording',
    isArming: status === 'arming',
    level,
    elapsedMs,
    analyser,
    mimeType,
    clip,
    error,
    notice,
    available: capability === null,
    ready,
    start,
    stop,
    warmUp,
  }
}
