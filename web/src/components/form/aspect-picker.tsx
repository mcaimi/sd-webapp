import { ArrowLeftRight } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import type { Resolution } from '@/lib/families'

interface AspectPickerProps {
  resolutions: Resolution[]
  width: number
  height: number
  onChange: (size: { width: number; height: number }) => void
}

const MIN_SIZE = 256
const MAX_SIZE = 2048

/** Aspect-ratio presets near the model's native size, plus exact width/height. */
export function AspectPicker({ resolutions, width, height, onChange }: AspectPickerProps) {
  const preset = resolutions.find((r) => r.width === width && r.height === height)
  const setDimension = (key: 'width' | 'height', raw: string) => {
    const value = Math.round(Number(raw) / 8) * 8
    if (Number.isFinite(value)) onChange({ width, height, [key]: Math.min(MAX_SIZE, Math.max(MIN_SIZE, value)) })
  }

  return (
    <div className="space-y-2">
      <Label>Size</Label>
      <ToggleGroup
        type="single"
        variant="outline"
        size="sm"
        value={preset?.label ?? ''}
        onValueChange={(label) => {
          const r = resolutions.find((x) => x.label === label)
          if (r) onChange({ width: r.width, height: r.height })
        }}
        className="flex w-full flex-wrap"
      >
        {resolutions.map((r) => (
          <ToggleGroupItem
            key={r.label}
            value={r.label}
            aria-label={`${r.label} (${r.width}×${r.height})`}
            className="flex-1 gap-1.5"
          >
            <AspectIcon width={r.width} height={r.height} />
            <span className="text-xs">{r.label}</span>
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
      <div className="flex items-center gap-2">
        <Input
          key={`w${width}`}
          aria-label="Width"
          inputMode="numeric"
          defaultValue={width}
          className="h-8 text-center tabular-nums"
          onBlur={(e) => setDimension('width', e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && e.currentTarget.blur()}
        />
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          aria-label="Swap width and height"
          onClick={() => onChange({ width: height, height: width })}
        >
          <ArrowLeftRight />
        </Button>
        <Input
          key={`h${height}`}
          aria-label="Height"
          inputMode="numeric"
          defaultValue={height}
          className="h-8 text-center tabular-nums"
          onBlur={(e) => setDimension('height', e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && e.currentTarget.blur()}
        />
      </div>
      <p className="text-xs text-muted-foreground">
        {preset ? 'Native-resolution preset.' : 'Custom size, rounded to multiples of 8.'}
      </p>
    </div>
  )
}

function AspectIcon({ width, height }: { width: number; height: number }) {
  const scale = 12 / Math.max(width, height)
  return (
    <span className="flex size-3.5 items-center justify-center" aria-hidden>
      <span
        className="rounded-[2px] border-[1.5px] border-current"
        style={{ width: Math.max(5, width * scale), height: Math.max(5, height * scale) }}
      />
    </span>
  )
}
