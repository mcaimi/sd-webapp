import { Brush, Eraser, FlipHorizontal2, ImageUp, Redo2, Trash2, Undo2 } from 'lucide-react'
import { useCallback, useEffect, useImperativeHandle, useRef, useState, type Ref } from 'react'

import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { Slider } from '@/components/ui/slider'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { MOD_KEY } from '@/lib/platform'

import { hasPaint, invertMask, MASK_RGB, pushBounded } from './mask-utils'
import type { SourceImage } from './source-images'

const HISTORY_LIMIT = 20
const MIN_BRUSH = 4
const MAX_BRUSH = 200

type Tool = 'brush' | 'eraser'

export interface MaskEditorHandle {
  /** Mask as PNG base64 at image resolution: white = repaint, black = keep. */
  exportMask(): string
}

interface MaskEditorProps {
  image: SourceImage
  /** Mask to restore (e.g. when returning to the page). */
  initialMask?: ImageData
  onMaskChange: (hasMask: boolean) => void
  /** Called on unmount with the painted mask, so it can be restored later. */
  onPersist?: (mask: ImageData, image: SourceImage) => void
  onReplaceImage: () => void
  ref?: Ref<MaskEditorHandle>
}

/**
 * Paint the area to regenerate. The mask canvas works at the image's own
 * resolution and is scaled with CSS to fit, so strokes stay sharp at any size.
 */
export function MaskEditor({ image, initialMask, onMaskChange, onPersist, onReplaceImage, ref }: MaskEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const cursorRef = useRef<HTMLDivElement>(null)
  const [tool, setTool] = useState<Tool>('brush')
  const [size, setSize] = useState(() =>
    Math.min(MAX_BRUSH, Math.max(MIN_BRUSH, Math.round(Math.max(image.width, image.height) / 16))),
  )
  const undoStack = useRef<ImageData[]>([])
  const redoStack = useRef<ImageData[]>([])
  const [history, setHistory] = useState({ undo: 0, redo: 0 })
  const stroke = useRef<{ x: number; y: number } | null>(null)

  const context = () => canvasRef.current!.getContext('2d', { willReadFrequently: true })!
  const snapshot = () => context().getImageData(0, 0, image.width, image.height)
  const syncHistory = () => setHistory({ undo: undoStack.current.length, redo: redoStack.current.length })
  const report = useCallback(() => {
    const ctx = canvasRef.current?.getContext('2d', { willReadFrequently: true })
    if (ctx) onMaskChange(hasPaint(ctx.getImageData(0, 0, image.width, image.height).data))
  }, [image.width, image.height, onMaskChange])

  // Size the canvas to the image, restore a saved mask, and hand the mask back on unmount.
  useEffect(() => {
    const canvas = canvasRef.current!
    canvas.width = image.width
    canvas.height = image.height
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!
    if (initialMask && initialMask.width === image.width && initialMask.height === image.height) {
      ctx.putImageData(initialMask, 0, 0)
    }
    undoStack.current = []
    redoStack.current = []
    setHistory({ undo: 0, redo: 0 })
    report()
    return () => onPersist?.(ctx.getImageData(0, 0, image.width, image.height), image)
    // Only when the image changes; initialMask/onPersist are read once per image.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [image])

  const commitChange = (apply: () => void) => {
    pushBounded(undoStack.current, snapshot(), HISTORY_LIMIT)
    redoStack.current = []
    apply()
    syncHistory()
    report()
  }

  const undo = () => {
    const previous = undoStack.current.pop()
    if (!previous) return
    redoStack.current.push(snapshot())
    context().putImageData(previous, 0, 0)
    syncHistory()
    report()
  }
  const redo = () => {
    const next = redoStack.current.pop()
    if (!next) return
    pushBounded(undoStack.current, snapshot(), HISTORY_LIMIT)
    context().putImageData(next, 0, 0)
    syncHistory()
    report()
  }
  const clear = () => commitChange(() => context().clearRect(0, 0, image.width, image.height))
  const invert = () =>
    commitChange(() => {
      const data = snapshot()
      invertMask(data.data)
      context().putImageData(data, 0, 0)
    })

  useImperativeHandle(ref, () => ({
    exportMask() {
      const out = document.createElement('canvas')
      out.width = image.width
      out.height = image.height
      const ctx = out.getContext('2d')!
      ctx.drawImage(canvasRef.current!, 0, 0)
      ctx.globalCompositeOperation = 'source-in'
      ctx.fillStyle = '#fff'
      ctx.fillRect(0, 0, out.width, out.height)
      ctx.globalCompositeOperation = 'destination-over'
      ctx.fillStyle = '#000'
      ctx.fillRect(0, 0, out.width, out.height)
      const url = out.toDataURL('image/png')
      return url.slice(url.indexOf(',') + 1)
    },
  }))

  // Keyboard shortcuts, ignored while typing.
  const actions = useRef({ undo, redo })
  useEffect(() => {
    actions.current = { undo, redo }
  })
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      if (target?.closest('input, textarea, select, [contenteditable="true"], [role="combobox"]')) return
      const mod = e.metaKey || e.ctrlKey
      if (mod && e.key.toLowerCase() === 'z') {
        e.preventDefault()
        if (e.shiftKey) actions.current.redo()
        else actions.current.undo()
      } else if (!mod && !e.altKey) {
        if (e.key === 'b' || e.key === 'B') setTool('brush')
        else if (e.key === 'e' || e.key === 'E') setTool('eraser')
        else if (e.key === '[') setSize((s) => Math.max(MIN_BRUSH, Math.round(s / 1.25)))
        else if (e.key === ']') setSize((s) => Math.min(MAX_BRUSH, Math.round(s * 1.25)))
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const toCanvasPoint = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    return {
      x: ((e.clientX - rect.left) * image.width) / rect.width,
      y: ((e.clientY - rect.top) * image.height) / rect.height,
    }
  }

  const moveCursor = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const cursor = cursorRef.current
    if (!cursor) return
    const rect = e.currentTarget.getBoundingClientRect()
    const diameter = (size * rect.width) / image.width
    cursor.style.width = cursor.style.height = `${diameter}px`
    cursor.style.transform = `translate(${e.clientX - rect.left - diameter / 2}px, ${e.clientY - rect.top - diameter / 2}px)`
    cursor.style.opacity = e.pointerType === 'touch' ? '0' : '1'
  }

  const paintTo = (point: { x: number; y: number }) => {
    const ctx = context()
    const from = stroke.current ?? point
    ctx.globalCompositeOperation = tool === 'eraser' ? 'destination-out' : 'source-over'
    ctx.strokeStyle = ctx.fillStyle = `rgb(${MASK_RGB.join(' ')})`
    ctx.lineWidth = size
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.beginPath()
    if (stroke.current) {
      ctx.moveTo(from.x, from.y)
      ctx.lineTo(point.x, point.y)
      ctx.stroke()
    } else {
      // Canvas doesn't render zero-length lines, so a click/tap stamps a dot.
      ctx.arc(point.x, point.y, size / 2, 0, Math.PI * 2)
      ctx.fill()
    }
    stroke.current = point
  }

  return (
    <div className="space-y-3">
      <div
        role="toolbar"
        aria-label="Mask tools"
        className="flex flex-wrap items-center gap-x-3 gap-y-2 rounded-lg border bg-card p-1.5"
      >
        <ToggleGroup
          type="single"
          size="sm"
          value={tool}
          onValueChange={(v) => v && setTool(v as Tool)}
          aria-label="Tool"
        >
          <ToggleGroupItem value="brush" aria-label="Brush (B)" title="Brush (B)">
            <Brush />
            <span className="hidden sm:inline">Brush</span>
          </ToggleGroupItem>
          <ToggleGroupItem value="eraser" aria-label="Eraser (E)" title="Eraser (E)">
            <Eraser />
            <span className="hidden sm:inline">Eraser</span>
          </ToggleGroupItem>
        </ToggleGroup>
        <div className="flex min-w-40 flex-1 items-center gap-2 sm:max-w-56">
          <span className="text-xs text-muted-foreground">Size</span>
          <Slider
            value={[size]}
            min={MIN_BRUSH}
            max={MAX_BRUSH}
            step={1}
            onValueChange={([v]) => setSize(v)}
            aria-label="Brush size"
          />
          <span className="w-8 text-right text-xs text-muted-foreground tabular-nums">{size}</span>
        </div>
        <Separator orientation="vertical" className="hidden data-vertical:h-5 data-vertical:self-center sm:block" />
        <div className="flex items-center gap-0.5">
          <ToolButton label={`Undo (${MOD_KEY}+Z)`} onClick={undo} disabled={history.undo === 0}>
            <Undo2 />
          </ToolButton>
          <ToolButton label={`Redo (Shift+${MOD_KEY}+Z)`} onClick={redo} disabled={history.redo === 0}>
            <Redo2 />
          </ToolButton>
          <ToolButton label="Invert mask" onClick={invert}>
            <FlipHorizontal2 />
          </ToolButton>
          <ToolButton label="Clear mask" onClick={clear}>
            <Trash2 />
          </ToolButton>
          <ToolButton label="Use a different image" onClick={onReplaceImage}>
            <ImageUp />
          </ToolButton>
        </div>
      </div>

      <div className="flex justify-center rounded-xl border bg-[repeating-conic-gradient(var(--muted)_0_25%,transparent_0_50%)] bg-[length:24px_24px] p-2">
        <div className="relative w-fit max-w-full overflow-hidden rounded-md">
          <img
            src={image.url}
            alt={`Image to inpaint: ${image.name}`}
            draggable={false}
            className="block max-h-[min(65svh,48rem)] max-w-full select-none"
          />
          <canvas
            ref={canvasRef}
            aria-label="Mask canvas: paint over the area to regenerate"
            className="absolute inset-0 size-full cursor-none touch-none opacity-55"
            onPointerDown={(e) => {
              if (e.button !== 0) return
              e.currentTarget.setPointerCapture(e.pointerId)
              pushBounded(undoStack.current, snapshot(), HISTORY_LIMIT)
              redoStack.current = []
              stroke.current = null
              paintTo(toCanvasPoint(e))
            }}
            onPointerMove={(e) => {
              moveCursor(e)
              if (stroke.current) paintTo(toCanvasPoint(e))
            }}
            onPointerUp={() => {
              if (!stroke.current) return
              stroke.current = null
              syncHistory()
              report()
            }}
            onPointerCancel={() => {
              stroke.current = null
              syncHistory()
              report()
            }}
            onPointerLeave={() => cursorRef.current && (cursorRef.current.style.opacity = '0')}
          />
          <div
            ref={cursorRef}
            aria-hidden
            className="pointer-events-none absolute top-0 left-0 rounded-full border-2 border-white opacity-0 mix-blend-difference"
          />
        </div>
      </div>
      <p className="text-xs text-muted-foreground [@media(hover:none)]:hidden">
        Paint over what should change. <kbd>B</kbd> brush · <kbd>E</kbd> eraser · <kbd>[</kbd> <kbd>]</kbd> size.
      </p>
    </div>
  )
}

function ToolButton({
  label,
  onClick,
  disabled,
  children,
}: {
  label: string
  onClick: () => void
  disabled?: boolean
  children: React.ReactNode
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button type="button" variant="ghost" size="icon-sm" aria-label={label} onClick={onClick} disabled={disabled}>
          {children}
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}
