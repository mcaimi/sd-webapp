import { useEffect, useId, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'

import { formatPercent } from '@/lib/format'

export interface Point {
  t: number
  value: number
}

interface LineChartProps {
  title: string
  points: Point[]
  /** Number of slots on the x-axis; new samples enter on the right. */
  capacity: number
  /** Human span of the full axis, e.g. "3 min". */
  span: string
}

const HEIGHT = 128
const MARGIN = { top: 10, right: 10, bottom: 20, left: 34 }
const GRID = [0, 50, 100]

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [width, setWidth] = useState(0)
  useEffect(() => {
    if (!ref.current) return
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width))
    observer.observe(ref.current)
    return () => observer.disconnect()
  }, [])
  return [ref, width] as const
}

function secondsAgo(t: number, now: number) {
  const s = Math.max(0, Math.round((now - t) / 1000))
  return s < 2 ? 'now' : s < 60 ? `${s}s ago` : `${Math.floor(s / 60)}m ${s % 60}s ago`
}

/**
 * Single-series percentage chart on a fixed 0–100% scale. Hover or focus it
 * and use the arrow keys to read individual samples.
 */
export function LineChart({ title, points, capacity, span }: LineChartProps) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const [active, setActive] = useState<number | null>(null)
  const readoutId = useId()

  const innerW = Math.max(0, width - MARGIN.left - MARGIN.right)
  const innerH = HEIGHT - MARGIN.top - MARGIN.bottom
  const firstSlot = capacity - points.length
  const x = (i: number) => MARGIN.left + ((firstSlot + i) / (capacity - 1)) * innerW
  const y = (v: number) => MARGIN.top + (1 - Math.min(100, Math.max(0, v)) / 100) * innerH

  const current = points.at(-1)
  const values = points.map((p) => p.value)
  const summary = current
    ? `${title}: currently ${formatPercent(current.value)}, min ${formatPercent(Math.min(...values))}, max ${formatPercent(Math.max(...values))} over the last ${points.length} samples`
    : `${title}: collecting data`

  const line = points.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p.value).toFixed(1)}`).join('')
  const area = points.length > 1 ? `${line}L${x(points.length - 1)},${y(0)}L${x(0)},${y(0)}Z` : ''
  const shown = active !== null && points[active] ? active : null
  const now = current?.t ?? 0

  const onPointerMove = (e: PointerEvent<SVGRectElement>) => {
    if (!points.length) return
    const rect = e.currentTarget.getBoundingClientRect()
    const slot = Math.round(((e.clientX - rect.left) / rect.width) * (capacity - 1))
    setActive(Math.min(points.length - 1, Math.max(0, slot - firstSlot)))
  }
  const onKeyDown = (e: KeyboardEvent) => {
    if (!points.length) return
    const last = points.length - 1
    const index = shown ?? last
    const next =
      e.key === 'ArrowLeft'
        ? Math.max(0, index - 1)
        : e.key === 'ArrowRight'
          ? Math.min(last, index + 1)
          : e.key === 'Home'
            ? 0
            : e.key === 'End'
              ? last
              : undefined
    if (e.key === 'Escape') setActive(null)
    if (next === undefined) return
    e.preventDefault()
    setActive(next)
  }

  return (
    <figure className="space-y-2">
      <figcaption className="flex items-baseline justify-between gap-2">
        <span className="text-sm font-medium">{title}</span>
        <span className="text-lg font-semibold">{current ? formatPercent(current.value) : '–'}</span>
      </figcaption>
      <div
        ref={ref}
        role="group"
        tabIndex={0}
        aria-label={summary}
        aria-describedby={readoutId}
        onKeyDown={onKeyDown}
        onFocus={() => points.length && setActive((a) => a ?? points.length - 1)}
        onBlur={() => setActive(null)}
        className="relative rounded-md outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
      >
        {width > 0 && (
          <svg width={width} height={HEIGHT} aria-hidden className="block overflow-visible">
            {GRID.map((v) => (
              <g key={v}>
                <line x1={MARGIN.left} x2={width - MARGIN.right} y1={y(v)} y2={y(v)} className="stroke-border" />
                <text
                  x={MARGIN.left - 6}
                  y={y(v)}
                  textAnchor="end"
                  dominantBaseline="middle"
                  className="fill-muted-foreground text-[10px]"
                >
                  {v}%
                </text>
              </g>
            ))}
            <text x={MARGIN.left} y={HEIGHT - 4} className="fill-muted-foreground text-[10px]">
              {span} ago
            </text>
            <text
              x={width - MARGIN.right}
              y={HEIGHT - 4}
              textAnchor="end"
              className="fill-muted-foreground text-[10px]"
            >
              now
            </text>
            {points.length > 1 && (
              <>
                <path d={area} className="fill-meter-ok/10" />
                <path
                  d={line}
                  fill="none"
                  strokeWidth={2}
                  strokeLinejoin="round"
                  strokeLinecap="round"
                  className="stroke-meter-ok"
                />
              </>
            )}
            {current && shown === null && (
              <circle
                cx={x(points.length - 1)}
                cy={y(current.value)}
                r={4}
                strokeWidth={2}
                className="fill-meter-ok stroke-card"
              />
            )}
            {shown !== null && (
              <>
                <line
                  x1={x(shown)}
                  x2={x(shown)}
                  y1={MARGIN.top}
                  y2={y(0)}
                  strokeDasharray="3 3"
                  className="stroke-muted-foreground"
                />
                <circle
                  cx={x(shown)}
                  cy={y(points[shown].value)}
                  r={4}
                  strokeWidth={2}
                  className="fill-meter-ok stroke-card"
                />
              </>
            )}
            {/* Hit area larger than the line, covering the whole plot. */}
            <rect
              x={MARGIN.left}
              y={0}
              width={innerW}
              height={HEIGHT}
              fill="transparent"
              onPointerMove={onPointerMove}
              onPointerLeave={() => setActive(null)}
            />
          </svg>
        )}
        {shown !== null && (
          <div
            className="pointer-events-none absolute top-0 z-10 -translate-x-1/2 -translate-y-full rounded-md border bg-popover px-2 py-1 text-xs whitespace-nowrap text-popover-foreground shadow-md"
            style={{ left: Math.min(Math.max(x(shown), 48), width - 48) }}
          >
            <span className="font-medium">{formatPercent(points[shown].value)}</span>
            <span className="text-muted-foreground"> · {secondsAgo(points[shown].t, now)}</span>
          </div>
        )}
        {points.length < 2 && (
          <p className="absolute inset-0 flex items-center justify-center text-xs text-muted-foreground">
            Collecting data…
          </p>
        )}
      </div>
      <p id={readoutId} className="sr-only" aria-live="polite">
        {shown !== null ? `${formatPercent(points[shown].value)}, ${secondsAgo(points[shown].t, now)}` : ''}
      </p>
      <table className="sr-only">
        <caption>{title} history</caption>
        <thead>
          <tr>
            <th scope="col">Time</th>
            <th scope="col">Value</th>
          </tr>
        </thead>
        <tbody>
          {points.map((p) => (
            <tr key={p.t}>
              <td>{secondsAgo(p.t, now)}</td>
              <td>{formatPercent(p.value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  )
}
