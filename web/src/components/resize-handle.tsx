import type { KeyboardEvent, PointerEvent } from 'react'

import { cn } from '@/lib/utils'

type ResizeHandleProps = {
  label: string
  /** Current size as a percentage, announced to assistive tech. */
  value: number
  min: number
  max: number
  /** Called while dragging with the pointer position and the handle element. */
  onDrag: (clientX: number, handle: HTMLElement) => void
  /** Keyboard resizing, in percentage points. */
  onStep: (delta: number) => void
  /** Double-click restores the default size. */
  onReset: () => void
  /** Hide the divider line until hovered or focused. */
  subtle?: boolean
  className?: string
}

/** A vertical drag handle between two columns. */
export function ResizeHandle({
  label,
  value,
  min,
  max,
  onDrag,
  onStep,
  onReset,
  subtle,
  className,
}: ResizeHandleProps) {
  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return
    event.preventDefault()
    event.currentTarget.setPointerCapture(event.pointerId)
    // Disables text selection and sidebar width transitions while dragging.
    document.body.dataset.resizing = ''
  }
  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) onDrag(event.clientX, event.currentTarget)
  }
  const endDrag = (event: PointerEvent<HTMLDivElement>) => {
    event.currentTarget.releasePointerCapture(event.pointerId)
    delete document.body.dataset.resizing
  }
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? 5 : 1
    if (event.key === 'ArrowLeft') onStep(-step)
    else if (event.key === 'ArrowRight') onStep(step)
    else return
    event.preventDefault()
  }

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label={label}
      aria-valuenow={Math.round(value)}
      aria-valuemin={min}
      aria-valuemax={max}
      tabIndex={0}
      title="Drag to resize, double-click to reset"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      onDoubleClick={onReset}
      onKeyDown={onKeyDown}
      className={cn(
        'flex w-4 cursor-col-resize touch-none justify-center outline-none select-none',
        'after:w-px after:transition-colors hover:after:w-0.5 hover:after:bg-primary/60',
        'focus-visible:after:w-0.5 focus-visible:after:bg-ring active:after:w-0.5 active:after:bg-primary',
        subtle ? 'after:bg-transparent' : 'after:bg-border',
        className,
      )}
    />
  )
}
