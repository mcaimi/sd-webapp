import { Loader2, TriangleAlert } from 'lucide-react'
import { useId, type ReactNode } from 'react'

import { ModelCombobox } from '@/components/pickers/model-combobox'
import { ModelMetadataButton } from '@/components/pickers/model-metadata'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import type { FamilyId } from '@/lib/families'
import { cn } from '@/lib/utils'

import { useWouldOverwrite } from './merge-hooks'
import { ensureSafetensors, isBlend, MERGE_METHODS, type MergeMethod } from './merge-utils'

/** Radio cards for the merge algorithm (native radios, so arrow keys work). */
export function MethodPicker({
  value,
  onChange,
  compact,
}: {
  value: MergeMethod
  onChange: (method: MergeMethod) => void
  compact?: boolean
}) {
  const name = useId()
  return (
    <fieldset className="space-y-2">
      <legend className="mb-2 text-sm font-medium">Method</legend>
      <div className={cn('grid gap-2', compact ? 'grid-cols-2 sm:grid-cols-4' : 'grid-cols-2')}>
        {MERGE_METHODS.map((m) => (
          <label
            key={m.id}
            className={cn(
              'relative cursor-pointer rounded-lg border p-3 transition-colors hover:bg-accent/50',
              'has-[:focus-visible]:ring-3 has-[:focus-visible]:ring-ring/50',
              value === m.id && 'border-primary bg-primary/5',
            )}
          >
            <input
              type="radio"
              name={name}
              value={m.id}
              checked={value === m.id}
              onChange={() => onChange(m.id)}
              className="sr-only"
            />
            <span className="block text-sm font-medium">{m.label}</span>
            {!compact && <span className="mt-1 block text-xs text-muted-foreground">{m.description}</span>}
          </label>
        ))}
      </div>
      {compact && (
        <p className="text-xs text-muted-foreground">{MERGE_METHODS.find((m) => m.id === value)?.description}</p>
      )}
    </fieldset>
  )
}

/** Alpha slider. For blends it reads as a position between A and B; otherwise as B's strength. */
export function AlphaField({
  method,
  value,
  onChange,
  label,
}: {
  method: MergeMethod
  value: number
  onChange: (alpha: number) => void
  label?: string
}) {
  const blend = isBlend(method)
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span className="font-medium">{label ?? (blend ? 'Mix' : 'Strength')}</span>
        <span className="text-muted-foreground tabular-nums">{value.toFixed(2)}</span>
      </div>
      <Slider
        value={[value]}
        min={0}
        max={1}
        step={0.05}
        onValueChange={([v]) => onChange(v)}
        aria-label={blend ? 'Mix between base and target' : 'Strength of the target'}
      />
      <div className="flex justify-between text-xs text-muted-foreground">
        {blend ? (
          <>
            <span>← All base (A)</span>
            <span>All target (B) →</span>
          </>
        ) : (
          <>
            <span>Subtle</span>
            <span>Strong</span>
          </>
        )}
      </div>
    </div>
  )
}

/** Labelled checkpoint picker with a metadata button. */
export function ModelField({
  family,
  label,
  value,
  onChange,
  exclude,
}: {
  family: FamilyId
  label: string
  value: string
  onChange: (name: string) => void
  exclude?: string[]
}) {
  const id = useId()
  return (
    <div className="space-y-2">
      <Label htmlFor={id}>{label}</Label>
      <div className="flex gap-1">
        {/* min-w-0 lets the picker shrink so the metadata button stays visible. */}
        <div className="min-w-0 flex-1">
          <ModelCombobox
            id={id}
            family={family}
            resource="checkpoints"
            value={value}
            onChange={onChange}
            exclude={exclude}
            placeholder="Select a checkpoint"
          />
        </div>
        {value && <ModelMetadataButton family={family} resource="checkpoints" name={value} />}
      </div>
    </div>
  )
}

export function OutputNameField({
  family,
  value,
  onChange,
  label = 'Output filename',
}: {
  family: FamilyId
  value: string
  onChange: (name: string) => void
  label?: string
}) {
  const id = useId()
  const overwrite = useWouldOverwrite(family, value)
  return (
    <div className="space-y-2">
      <Label htmlFor={id}>{label}</Label>
      <Input
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onBlur={() => value.trim() && onChange(ensureSafetensors(value))}
        placeholder="my_merge.safetensors"
        aria-invalid={overwrite || undefined}
        aria-describedby={overwrite ? `${id}-warn` : undefined}
      />
      {overwrite && (
        <p id={`${id}-warn`} className="flex items-center gap-1.5 text-xs text-meter-warn">
          <TriangleAlert className="size-3.5 shrink-0" aria-hidden />A checkpoint with this name exists and will be
          overwritten.
        </p>
      )}
    </div>
  )
}

/** Submit row: a disabled button explains why. */
export function SubmitRow({
  problem,
  pending,
  onSubmit,
  label,
  icon,
}: {
  problem: string | null
  pending: boolean
  onSubmit: () => void
  label: string
  icon: ReactNode
}) {
  return (
    <div className="space-y-2 pt-1">
      <Button size="lg" className="w-full" onClick={onSubmit} disabled={!!problem || pending}>
        {pending ? <Loader2 className="animate-spin" /> : icon}
        {label}
      </Button>
      <p className="text-center text-xs text-muted-foreground" aria-live="polite">
        {problem ?? 'Merges run on the backend and can take several minutes.'}
      </p>
    </div>
  )
}
