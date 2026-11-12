import { useRef, useState } from 'react'

interface BeforeAfterProps {
  before: string
  after: string
  alt: string
}

/** Drag (or use arrow keys) to reveal the original over the inpainted result. */
export function BeforeAfter({ before, after, alt }: BeforeAfterProps) {
  const [position, setPosition] = useState(50)
  const containerRef = useRef<HTMLDivElement>(null)
  const dragging = useRef(false)

  const setFromPointer = (clientX: number) => {
    const rect = containerRef.current!.getBoundingClientRect()
    setPosition(Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100)))
  }

  return (
    <figure className="mb-5 space-y-2">
      <div
        ref={containerRef}
        className="relative cursor-ew-resize touch-none overflow-hidden rounded-lg border select-none"
        onPointerDown={(e) => {
          dragging.current = true
          e.currentTarget.setPointerCapture(e.pointerId)
          setFromPointer(e.clientX)
        }}
        onPointerMove={(e) => dragging.current && setFromPointer(e.clientX)}
        onPointerUp={() => (dragging.current = false)}
        onPointerCancel={() => (dragging.current = false)}
      >
        <img src={after} alt={`${alt} (inpainted)`} draggable={false} className="block w-full" />
        <img
          src={before}
          alt={`${alt} (original)`}
          draggable={false}
          className="absolute inset-0 size-full object-cover"
          style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
        />
        <div
          role="slider"
          tabIndex={0}
          aria-label="Compare original and inpainted image"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(position)}
          aria-valuetext={`${Math.round(position)}% original`}
          onKeyDown={(e) => {
            const step = e.shiftKey ? 10 : 2
            if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') setPosition((p) => Math.max(0, p - step))
            else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') setPosition((p) => Math.min(100, p + step))
            else if (e.key === 'Home') setPosition(0)
            else if (e.key === 'End') setPosition(100)
            else return
            e.preventDefault()
          }}
          className="absolute inset-y-0 w-0.5 -translate-x-1/2 bg-white shadow-[0_0_0_1px_rgb(0_0_0/0.3)] outline-none focus-visible:ring-3 focus-visible:ring-ring"
          style={{ left: `${position}%` }}
        >
          <span className="absolute top-1/2 left-1/2 flex size-7 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full bg-white text-[10px] font-semibold text-black shadow">
            ⇆
          </span>
        </div>
        <span className="pointer-events-none absolute top-2 left-2 rounded bg-black/60 px-1.5 py-0.5 text-[10px] text-white">
          Before
        </span>
        <span className="pointer-events-none absolute top-2 right-2 rounded bg-black/60 px-1.5 py-0.5 text-[10px] text-white">
          After
        </span>
      </div>
      <figcaption className="text-xs text-muted-foreground">Drag or use the arrow keys to compare.</figcaption>
    </figure>
  )
}
