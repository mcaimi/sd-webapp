import { useId, useState } from 'react'

import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'

interface SliderFieldProps {
  label: string
  value: number
  onChange: (value: number) => void
  min: number
  max: number
  step?: number
  hint?: string
  disabled?: boolean
}

/** Slider with an editable number for exact values. */
export function SliderField({ label, value, onChange, min, max, step = 1, hint, disabled }: SliderFieldProps) {
  const id = useId()
  // Local text so the user can clear the box and type freely; committed on blur/Enter.
  const [text, setText] = useState<string | null>(null)
  const commit = () => {
    if (text === null) return
    const parsed = Number(text)
    if (text.trim() !== '' && Number.isFinite(parsed)) {
      onChange(Math.min(max, Math.max(min, Math.round(parsed / step) * step)))
    }
    setText(null)
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <Label htmlFor={id}>{label}</Label>
        <Input
          id={id}
          inputMode="decimal"
          className="h-7 w-16 text-right tabular-nums"
          value={text ?? String(value)}
          disabled={disabled}
          onChange={(e) => setText(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => e.key === 'Enter' && commit()}
        />
      </div>
      <Slider
        value={[value]}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onValueChange={([v]) => onChange(v)}
        aria-label={label}
      />
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  )
}
