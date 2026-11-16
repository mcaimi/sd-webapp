import { ChevronDown } from 'lucide-react'
import { useState, type ReactNode } from 'react'

import { useSchedulers } from '@/api/queries'
import { AspectPicker } from '@/components/form/aspect-picker'
import { SeedField } from '@/components/form/seed-field'
import { SliderField } from '@/components/form/slider-field'
import { LoraList } from '@/components/pickers/lora-list'
import { ModelCombobox } from '@/components/pickers/model-combobox'
import { ModelMetadataButton } from '@/components/pickers/model-metadata'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import { FAMILIES, type FamilyId } from '@/lib/families'
import { appendToPrompt, MAX_BATCH, type GenerationParams } from '@/lib/generation'
import { cn } from '@/lib/utils'

interface GenerationSettingsProps {
  family: FamilyId
  params: GenerationParams
  onChange: (patch: Partial<GenerationParams>) => void
  /** Hide the checkpoint picker (comparison mode picks several). */
  showModel?: boolean
  /** LoRAs and custom VAEs aren't supported by every endpoint (e.g. compare). */
  showAdapters?: boolean
  /** Inpainting uses the input image's size. */
  showSize?: boolean
  showBatch?: boolean
  /** Extra controls rendered in the sampling section (e.g. inpaint strength). */
  extraSampling?: ReactNode
}

export function GenerationSettings({
  family,
  params,
  onChange,
  showModel = true,
  showAdapters = true,
  showSize = true,
  showBatch = true,
  extraSampling,
}: GenerationSettingsProps) {
  const info = FAMILIES[family]
  const { data: schedulers = [info.defaults.scheduler] } = useSchedulers(family)
  const isFlux = family === 'flux'

  return (
    <div className="space-y-5">
      {showModel && (
        <div className="space-y-2">
          <Label htmlFor="checkpoint">Checkpoint</Label>
          <div className="flex gap-1">
            <div className="min-w-0 flex-1">
              <ModelCombobox
                id="checkpoint"
                family={family}
                resource="checkpoints"
                value={params.model}
                onChange={(model) => onChange({ model })}
                placeholder="No checkpoints installed"
              />
            </div>
            {params.model && <ModelMetadataButton family={family} resource="checkpoints" name={params.model} />}
          </div>
        </div>
      )}

      {showAdapters && (
        <LoraList
          family={family}
          value={params.loras}
          onChange={(loras) => onChange({ loras })}
          onInsertTriggerWords={(words) => onChange({ prompt: appendToPrompt(params.prompt, words) })}
        />
      )}

      {(showModel || showAdapters) && <Separator />}

      {showSize && (
        <AspectPicker
          resolutions={info.resolutions}
          width={params.width}
          height={params.height}
          onChange={(size) => onChange(size)}
        />
      )}

      <SliderField
        label="Steps"
        value={params.steps}
        onChange={(steps) => onChange({ steps })}
        min={1}
        max={isFlux ? 50 : 100}
      />
      <SliderField
        label={isFlux ? 'Guidance' : 'CFG scale'}
        value={params.cfgScale}
        onChange={(cfgScale) => onChange({ cfgScale })}
        min={isFlux ? 1 : 0}
        max={isFlux ? 10 : 20}
        step={0.5}
        hint={isFlux ? 'Distilled guidance; 3–4 is typical.' : undefined}
      />
      {info.trueCfgScale && (
        <SliderField
          label="True CFG"
          value={params.trueCfgScale}
          onChange={(trueCfgScale) => onChange({ trueCfgScale })}
          min={info.trueCfgScale.min}
          max={info.trueCfgScale.max}
          step={0.5}
          hint="Above 1 enables the negative prompt, at roughly twice the generation time."
        />
      )}
      {extraSampling}

      <div className="space-y-2">
        <Label htmlFor="scheduler">Scheduler</Label>
        <Select value={params.scheduler} onValueChange={(scheduler) => onChange({ scheduler })}>
          <SelectTrigger id="scheduler" className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {schedulers.map((s) => (
              <SelectItem key={s} value={s}>
                {s}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <SeedField seed={params.seed} locked={params.seedLocked} onChange={onChange} />

      {showBatch && (
        <SliderField
          label="Images"
          value={params.count}
          onChange={(count) => onChange({ count })}
          min={1}
          max={MAX_BATCH}
          hint={params.count > 1 && params.seedLocked ? 'Locked seed: images use consecutive seeds.' : undefined}
        />
      )}

      {showAdapters && (
        <Advanced>
          <div className="space-y-2">
            <Label htmlFor="vae">VAE</Label>
            <ModelCombobox
              id="vae"
              family={family}
              resource="vaes"
              value={params.vae}
              onChange={(vae) => onChange({ vae })}
              noneLabel="Checkpoint default"
            />
          </div>
        </Advanced>
      )}
    </div>
  )
}

function Advanced({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false)
  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger className="flex w-full items-center justify-between text-sm font-medium text-muted-foreground hover:text-foreground">
        Advanced
        <ChevronDown className={cn('size-4 transition-transform', open && 'rotate-180')} aria-hidden />
      </CollapsibleTrigger>
      <CollapsibleContent className="pt-3">{children}</CollapsibleContent>
    </Collapsible>
  )
}
