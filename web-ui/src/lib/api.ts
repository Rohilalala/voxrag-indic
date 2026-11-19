/**
 * The only place in the UI that talks to the backend.
 *
 * Two rules hold everywhere below:
 *   1. Nothing throws. Every call returns `ApiResult<T>`; a failure is a value a component
 *      renders, never an exception that unmounts the tree into a blank screen.
 *   2. Every request has a deadline. On bad conference wifi a hung fetch is
 *      indistinguishable from a broken app, so the UI is told about it instead.
 */

import type {
  AnalyticsReport,
  ApiError,
  ApiResult,
  GenerationMode,
  HealthReport,
  Lang,
  PipelineResult,
  QueryRequest,
  StrategyInfo,
  VoiceRequest,
  WsAudioMessage,
  WsResultFrame,
  WsServerFrame,
  WsStageFrame,
  WsTokenFrame,
  WsTranscriptFrame,
} from './types'

/** Empty in the normal deployment: FastAPI serves the bundle and the API from one origin,
 *  and `npm run dev` proxies /api and /ws to 127.0.0.1:8000. Set VITE_API_BASE only to
 *  point a local UI at a remote backend. */
const API_BASE: string = String(import.meta.env.VITE_API_BASE ?? '').replace(/\/+$/, '')

/** Quality mode runs a local LLM; fast mode is budgeted at 200ms. One generous ceiling
 *  covers both without turning a slow answer into a false failure. */
const PIPELINE_TIMEOUT_MS = 90_000
/** Panels poll these; they must fail fast enough to show a stale-data state. */
const PANEL_TIMEOUT_MS = 10_000

// ---------------------------------------------------------------------------------------
// Fetch plumbing
// ---------------------------------------------------------------------------------------

function url(path: string): string {
  return `${API_BASE}${path}`
}

function err(kind: ApiError['kind'], message: string, extra: Partial<ApiError> = {}): ApiError {
  return { kind, message, ...extra }
}

/** Merge a caller's abort signal with our own deadline. Hand-rolled rather than
 *  `AbortSignal.any`, which is recent enough that a conference laptop's browser may not
 *  have it -- and a missing method here would break every request, not one. */
function withDeadline(ms: number, signal?: AbortSignal): { signal: AbortSignal; done(): void } {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(new DOMException('timeout', 'TimeoutError')), ms)
  const relay = () => controller.abort(signal?.reason)
  if (signal) {
    if (signal.aborted) relay()
    else signal.addEventListener('abort', relay, { once: true })
  }
  return {
    signal: controller.signal,
    done() {
      clearTimeout(timer)
      signal?.removeEventListener('abort', relay)
    },
  }
}

/** Pull the backend's structured error envelope out of a non-2xx body.
 *  Shape: {request_id, ok:false, error, message, stage?}. */
function errorFromBody(status: number, body: unknown, fallback: string): ApiError {
  if (body && typeof body === 'object') {
    const b = body as Record<string, unknown>
    // FastAPI's own 422s use `detail`, which is a list of validation objects.
    const detail = typeof b.detail === 'string' ? b.detail : undefined
    const message =
      (typeof b.message === 'string' && b.message) || detail || `${fallback} (HTTP ${status})`
    return err('http', message, {
      status,
      requestId: typeof b.request_id === 'string' ? b.request_id : undefined,
      error: typeof b.error === 'string' ? b.error : undefined,
      stage: typeof b.stage === 'string' ? b.stage : undefined,
    })
  }
  return err('http', `${fallback} (HTTP ${status})`, { status })
}

async function request<T>(
  path: string,
  init: RequestInit,
  timeoutMs: number,
  signal?: AbortSignal,
): Promise<ApiResult<T>> {
  const deadline = withDeadline(timeoutMs, signal)
  try {
    const response = await fetch(url(path), { ...init, signal: deadline.signal })
    const raw = await response.text()
    let body: unknown = null
    if (raw) {
      try {
        body = JSON.parse(raw)
      } catch {
        body = null
      }
    }
    if (!response.ok) {
      return { ok: false, error: errorFromBody(response.status, body, `${path} failed`) }
    }
    if (body === null) {
      return { ok: false, error: err('parse', `${path} returned a body that was not JSON`) }
    }
    return { ok: true, data: body as T }
  } catch (cause) {
    return { ok: false, error: classify(cause, path) }
  } finally {
    deadline.done()
  }
}

function classify(cause: unknown, path: string): ApiError {
  if (cause instanceof DOMException && cause.name === 'TimeoutError') {
    return err('timeout', `${path} timed out`)
  }
  if (cause instanceof DOMException && cause.name === 'AbortError') {
    return err('aborted', `${path} was cancelled`)
  }
  const message = cause instanceof Error ? cause.message : String(cause)
  // A fetch rejection with no HTTP status means the request never landed: server down,
  // wifi gone, or the dev proxy is not pointed at a running backend.
  return err('network', `cannot reach the backend (${message})`)
}

const JSON_POST = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

// ---------------------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------------------

/** POST /api/query -- text in, full PipelineResult out. */
export function query(req: QueryRequest, signal?: AbortSignal): Promise<ApiResult<PipelineResult>> {
  return request<PipelineResult>(
    '/api/query',
    JSON_POST({ text: req.text, lang: req.lang ?? null, mode: req.mode ?? 'fast' }),
    PIPELINE_TIMEOUT_MS,
    signal,
  )
}

/** Filename extension per container, so the upload is self-describing even if a proxy
 *  strips the blob's content type. */
const EXT: Record<string, string> = {
  'audio/webm': 'webm',
  'audio/ogg': 'ogg',
  'audio/wav': 'wav',
  'audio/x-wav': 'wav',
  'audio/mpeg': 'mp3',
  'audio/mp4': 'm4a',
  'audio/flac': 'flac',
}

/** POST /api/voice -- multipart audio in, full PipelineResult out. */
export function voice(req: VoiceRequest, signal?: AbortSignal): Promise<ApiResult<PipelineResult>> {
  const mime = req.mime ?? req.audio.type ?? 'audio/webm'
  const base = mime.split(';', 1)[0] ?? ''
  const form = new FormData()
  form.append('file', req.audio, `recording.${EXT[base.trim().toLowerCase()] ?? 'webm'}`)
  // The backend coerces "" and "auto" to auto-detect; sending nothing at all is the same.
  if (req.lang) form.append('lang', req.lang)
  form.append('mode', req.mode ?? 'fast')
  // No Content-Type header: the browser has to set the multipart boundary itself.
  return request<PipelineResult>('/api/voice', { method: 'POST', body: form }, PIPELINE_TIMEOUT_MS, signal)
}

/** GET /api/analytics -- live per-stage percentiles over this process's traces. */
export function analytics(signal?: AbortSignal): Promise<ApiResult<AnalyticsReport>> {
  return request<AnalyticsReport>('/api/analytics', { method: 'GET' }, PANEL_TIMEOUT_MS, signal)
}

/** GET /api/health -- always HTTP 200; the payload carries the verdict. */
export function health(signal?: AbortSignal): Promise<ApiResult<HealthReport>> {
  return request<HealthReport>('/api/health', { method: 'GET' }, PANEL_TIMEOUT_MS, signal)
}

/** GET /api/strategies -- every chunking strategy and what it contributed to the index. */
export function strategies(signal?: AbortSignal): Promise<ApiResult<StrategyInfo[]>> {
  return request<StrategyInfo[]>('/api/strategies', { method: 'GET' }, PANEL_TIMEOUT_MS, signal)
}

// ---------------------------------------------------------------------------------------
// Audio helpers
// ---------------------------------------------------------------------------------------

/** Blob -> bare base64 (no `data:` prefix). FileReader rather than a manual btoa loop:
 *  it does the work off the main thread, which matters for a multi-second recording. */
export function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(reader.error ?? new Error('could not read the recording'))
    reader.onload = () => {
      const text = String(reader.result ?? '')
      const comma = text.indexOf(',')
      resolve(comma === -1 ? text : text.slice(comma + 1))
    }
    reader.readAsDataURL(blob)
  })
}

// ---------------------------------------------------------------------------------------
// WS /ws/voice
// ---------------------------------------------------------------------------------------

export type SocketStatus = 'idle' | 'connecting' | 'open' | 'reconnecting' | 'closed'

export interface VoiceSocketHandlers {
  /** A stage finished, with its measured duration. Arrives as the pipeline runs. */
  onStage?: (frame: WsStageFrame) => void
  /** Speech-to-text landed, before retrieval starts. */
  onTranscript?: (frame: WsTranscriptFrame) => void
  /** Quality mode only: one incrementally generated token. */
  onToken?: (frame: WsTokenFrame) => void
  onResult?: (result: PipelineResult) => void
  onError?: (error: ApiError) => void
  onStatusChange?: (status: SocketStatus) => void
}

/** Backoff schedule in ms, then it stays at the last value. Short first steps because the
 *  usual cause is a backend restart, which is over in about a second. */
const BACKOFF_MS = [250, 500, 1000, 2000, 4000, 8000] as const

/** Keepalive interval. Proxies and captive-portal wifi drop idle sockets silently; the
 *  server ignores `ping` frames by contract, so this is free. */
const HEARTBEAT_MS = 20_000

/**
 * A managed connection to /ws/voice.
 *
 * Reconnects with backoff, keeps one pending turn across a reconnect, and reports every
 * failure through `onError` as a typed `ApiError` rather than throwing. `close()` is final:
 * a socket the user closed never reconnects on its own.
 */
export class VoiceSocket {
  private handlers: VoiceSocketHandlers
  private ws: WebSocket | null = null
  private state: SocketStatus = 'idle'
  private attempt = 0
  private retryTimer: ReturnType<typeof setTimeout> | null = null
  private heartbeat: ReturnType<typeof setInterval> | null = null
  private closedByUser = false
  /** At most one queued turn: an older recording the user has already moved past is
   *  noise, so a second send replaces the first rather than queueing behind it. */
  private pending: string | null = null

  constructor(handlers: VoiceSocketHandlers = {}) {
    this.handlers = handlers
  }

  get status(): SocketStatus {
    return this.state
  }

  get isOpen(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
  }

  /** Absolute ws:// or wss:// URL for the socket, derived from the page origin so it
   *  works behind the dev proxy, behind TLS, and on a plain container alike. */
  static endpoint(): string {
    if (API_BASE) return API_BASE.replace(/^http/, 'ws') + '/ws/voice'
    const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    return `${scheme}//${window.location.host}/ws/voice`
  }

  /** Open the socket. Safe to call repeatedly; a live or in-flight connection is reused. */
  connect(): void {
    this.closedByUser = false
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return
    }
    this.setStatus(this.attempt === 0 ? 'connecting' : 'reconnecting')

    let socket: WebSocket
    try {
      socket = new WebSocket(VoiceSocket.endpoint())
    } catch (cause) {
      // Thrown synchronously for a malformed URL or a blocked mixed-content upgrade.
      this.fail(err('socket', cause instanceof Error ? cause.message : String(cause)))
      this.scheduleRetry()
      return
    }
    this.ws = socket

    socket.onopen = () => {
      this.attempt = 0
      this.setStatus('open')
      this.startHeartbeat()
      if (this.pending) {
        const frame = this.pending
        this.pending = null
        this.rawSend(frame)
      }
    }

    socket.onmessage = (event) => this.receive(event.data)

    socket.onerror = () => {
      // The event carries no detail by design (browsers hide cross-origin failure causes).
      // `onclose` follows immediately and drives the retry, so this only reports.
      if (!this.closedByUser) {
        this.handlers.onError?.(err('socket', 'the live connection dropped'))
      }
    }

    socket.onclose = (event) => {
      this.stopHeartbeat()
      if (socket !== this.ws) return // a stale socket we already replaced
      this.ws = null
      if (this.closedByUser) {
        this.setStatus('closed')
        return
      }
      // 1000 = the server closed cleanly; anything else is worth retrying.
      if (event.code !== 1000 || this.pending) this.scheduleRetry()
      else this.setStatus('closed')
    }
  }

  /**
   * Send one recorded turn. Connects first if needed and queues the frame until the socket
   * is open, so a click during a reconnect is not silently dropped.
   */
  async send(
    audio: Blob,
    options: { lang?: Lang | null; mode?: GenerationMode; mime?: string } = {},
  ): Promise<void> {
    let data: string
    try {
      data = await blobToBase64(audio)
    } catch (cause) {
      this.handlers.onError?.(
        err('socket', cause instanceof Error ? cause.message : 'could not read the recording'),
      )
      return
    }
    const message: WsAudioMessage = {
      type: 'audio',
      mime: options.mime ?? audio.type ?? 'audio/webm',
      data,
      lang: options.lang ?? null,
      mode: options.mode ?? 'fast',
    }
    const frame = JSON.stringify(message)
    if (this.isOpen) this.rawSend(frame)
    else {
      this.pending = frame
      this.connect()
    }
  }

  /** Close for good. Cancels any pending retry and drops the queued turn. */
  close(): void {
    this.closedByUser = true
    this.pending = null
    this.attempt = 0
    if (this.retryTimer !== null) {
      clearTimeout(this.retryTimer)
      this.retryTimer = null
    }
    this.stopHeartbeat()
    const socket = this.ws
    this.ws = null
    if (socket) {
      socket.onopen = socket.onmessage = socket.onerror = socket.onclose = null
      if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
        socket.close(1000, 'client closed')
      }
    }
    this.setStatus('closed')
  }

  // -- internals ------------------------------------------------------------------------

  private rawSend(frame: string): void {
    try {
      this.ws?.send(frame)
    } catch (cause) {
      this.pending = frame
      this.fail(err('socket', cause instanceof Error ? cause.message : 'send failed'))
      this.scheduleRetry()
    }
  }

  private receive(raw: unknown): void {
    if (typeof raw !== 'string') return // the server only ever sends text frames
    let frame: WsServerFrame
    try {
      frame = JSON.parse(raw) as WsServerFrame
    } catch {
      this.handlers.onError?.(err('parse', 'the server sent a frame that was not JSON'))
      return
    }
    switch (frame.type) {
      case 'stage':
        this.handlers.onStage?.(frame)
        return
      case 'transcript':
        this.handlers.onTranscript?.(frame)
        return
      case 'token':
        this.handlers.onToken?.(frame)
        return
      case 'result':
        this.handlers.onResult?.((frame as WsResultFrame).result)
        return
      case 'error':
        this.handlers.onError?.(
          err('socket', frame.message || 'the pipeline failed', {
            requestId: frame.request_id,
            error: frame.error,
            stage: frame.stage,
          }),
        )
        return
      default:
        // Forward-compatible: an unknown frame type is ignored, not an error.
        return
    }
  }

  private scheduleRetry(): void {
    if (this.closedByUser || this.retryTimer !== null) return
    const step = BACKOFF_MS[Math.min(this.attempt, BACKOFF_MS.length - 1)] ?? 8000
    this.attempt += 1
    // Jitter so a backend restart does not bring every open tab back in lockstep.
    const delay = step * (0.85 + Math.random() * 0.3)
    this.setStatus('reconnecting')
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null
      this.connect()
    }, delay)
  }

  private startHeartbeat(): void {
    this.stopHeartbeat()
    this.heartbeat = setInterval(() => {
      if (this.isOpen) this.ws?.send('{"type":"ping"}')
    }, HEARTBEAT_MS)
  }

  private stopHeartbeat(): void {
    if (this.heartbeat !== null) {
      clearInterval(this.heartbeat)
      this.heartbeat = null
    }
  }

  private fail(error: ApiError): void {
    this.handlers.onError?.(error)
  }

  private setStatus(next: SocketStatus): void {
    if (this.state === next) return
    this.state = next
    this.handlers.onStatusChange?.(next)
  }
}
