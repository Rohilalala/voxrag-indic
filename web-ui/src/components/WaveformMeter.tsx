/**
 * Live input meter, drawn from the recorder's AnalyserNode.
 *
 * A mic button that looks identical whether or not audio is arriving is the classic way a
 * voice demo fails on stage: the presenter talks, nothing is being captured, and no one
 * finds out until the answer comes back empty. These bars are the proof of life -- they
 * move only because samples are actually arriving, and they are flat and still when they
 * are not. Nothing here animates on its own.
 */

import { useEffect, useRef } from 'react'

import { cn } from '@/lib/utils'

export interface WaveformMeterProps {
  /** Null until the microphone has been acquired. */
  analyser: AnalyserNode | null
  /** True while recording. False draws the resting state and stops the frame loop. */
  active: boolean
  className?: string
}

const BAR_WIDTH = 3
const BAR_GAP = 3
/** Resting height of a bar, in CSS pixels. */
const FLOOR = 2

/** Canvas cannot read CSS custom properties, so the theme is sampled from the element
 *  itself: the accent arrives as `color` (the element carries `text-primary`) and the
 *  rule line as `--border`. Both are oklch(); the literal fallbacks are assigned first so
 *  an engine that cannot parse oklch in canvas still draws something visible rather than
 *  silently keeping black-on-black. */
function readPalette(element: HTMLElement): { accent: string; rule: string } {
  const style = getComputedStyle(element)
  return {
    accent: style.color || '#5fd9e0',
    rule: style.getPropertyValue('--border').trim() || 'rgba(255,255,255,0.12)',
  }
}

export function WaveformMeter({ analyser, active, className }: WaveformMeterProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const fallback = { accent: '#5fd9e0', rule: 'rgba(255,255,255,0.12)' }
    let palette = fallback
    let width = 0
    let height = 0
    let raf = 0
    let lastPaint = 0

    // The meter is information, not decoration, so it keeps running under reduced motion --
    // but at a calmer rate than the display refresh.
    const calm = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false
    const minFrameMs = calm ? 80 : 0

    const buffer = analyser ? new Uint8Array(analyser.fftSize) : null

    const resize = () => {
      const dpr = window.devicePixelRatio || 1
      const cssWidth = Math.max(1, Math.round(canvas.clientWidth))
      const cssHeight = Math.max(1, Math.round(canvas.clientHeight))
      const pixelWidth = Math.round(cssWidth * dpr)
      const pixelHeight = Math.round(cssHeight * dpr)
      if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
        canvas.width = pixelWidth
        canvas.height = pixelHeight
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      width = cssWidth
      height = cssHeight
      palette = readPalette(canvas)
    }

    const draw = () => {
      if (width === 0 || height === 0) return
      const mid = Math.round(height / 2) + 0.5
      const span = height / 2 - 2

      ctx.clearRect(0, 0, width, height)

      // Resting rule, always present, so the strip reads as an instrument and not as an
      // empty box while idle.
      ctx.strokeStyle = fallback.rule
      ctx.strokeStyle = palette.rule
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(0, mid)
      ctx.lineTo(width, mid)
      ctx.stroke()

      const step = BAR_WIDTH + BAR_GAP
      const count = Math.max(1, Math.floor((width + BAR_GAP) / step))
      ctx.fillStyle = fallback.accent
      ctx.fillStyle = palette.accent

      if (!active || !analyser || !buffer) {
        // Still. No decorative idle animation: a bar that moves without audio would be a
        // lie about the microphone.
        ctx.globalAlpha = 0.22
        for (let i = 0; i < count; i += 1) {
          ctx.fillRect(i * step, mid - FLOOR / 2, BAR_WIDTH, FLOOR)
        }
        ctx.globalAlpha = 1
        return
      }

      analyser.getByteTimeDomainData(buffer)
      const samples = buffer.length
      for (let i = 0; i < count; i += 1) {
        const from = Math.floor((i * samples) / count)
        const to = Math.max(from + 1, Math.floor(((i + 1) * samples) / count))
        let peak = 0
        for (let s = from; s < to; s += 1) {
          const v = Math.abs(((buffer[s] ?? 128) - 128) / 128)
          if (v > peak) peak = v
        }
        // Same sqrt compression as the recorder's level: raw speech amplitude is small
        // enough that a linear mapping barely leaves the floor.
        const scaled = Math.min(1, Math.sqrt(peak) * 1.35)
        const barHeight = Math.max(FLOOR, scaled * span * 2)
        ctx.globalAlpha = 0.35 + 0.65 * scaled
        ctx.fillRect(i * step, mid - barHeight / 2, BAR_WIDTH, barHeight)
      }
      ctx.globalAlpha = 1
    }

    const frame = (now: number) => {
      if (now - lastPaint >= minFrameMs) {
        lastPaint = now
        draw()
      }
      raf = requestAnimationFrame(frame)
    }

    resize()
    draw()

    const observer = new ResizeObserver(() => {
      resize()
      // While active the loop repaints anyway; while idle this is the only repaint there is.
      if (!active) draw()
    })
    observer.observe(canvas)

    if (active && analyser) raf = requestAnimationFrame(frame)

    return () => {
      if (raf) cancelAnimationFrame(raf)
      observer.disconnect()
    }
  }, [analyser, active])

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className={cn('h-full w-full text-primary', className)}
    />
  )
}
