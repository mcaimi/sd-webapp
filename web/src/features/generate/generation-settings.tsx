import { ChevronDown } from 'lucide-react'
import { useEffect, useState, type ReactNode } from 'react'

import { useModels, useSchedulers } from '@/api/queries'
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
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { FAMILIES, type FamilyId, type VariantId } from '@/lib/families'
import { appendToPrompt, MAX_BATCH, modelPatch, variantPatch, type GenerationParams } from '@/lib/generation'
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
  const { data: apiSchedulers } = useSchedulers(family)
  const schedulers = apiSchedulers ?? [params.scheduler]
  const { data: checkpoints } = useModels(family, 'checkpoints')

  // Recover from a saved scheduler the backend no longer offers.
  useEffect(() => {
    if (apiSchedulers?.length && !apiSchedulers.includes(params.scheduler)) onChange({ scheduler: apiSchedulers[0] })
  }, [apiSchedulers, params.scheduler, onChange])

  const selectModel = (model: string) => {
    const entry = checkpoints?.find((m) => m.name === model)
    onChange(entry ? modelPatch(family, entry, params) : { model })
  }

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
                onChange={selectModel}
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

      {info.variants && (
        <div className="space-y-2">
          <Label id="variant-label">Variant</Label>
          <ToggleGroup
            type="single"
            variant="outline"
            aria-labelledby="variant-label"
            value={params.variant}
            onValueChange={(variant) => variant && onChange(variantPatch(family, variant as VariantId))}
            className="w-full"
          >
            {info.variants.map((v) => (
              <ToggleGroupItem key={v.id} value={v.id} className="flex-1">
                {v.label}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
          <p className="text-xs text-muted-foreground">Must match the checkpoint. Switching resets steps and CFG.</p>
        </div>
      )}

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
        max={info.sampling.maxSteps}
      />
      <SliderField
        label={info.sampling.cfg.label}
        value={params.cfgScale}
        onChange={(cfgScale) => onChange({ cfgScale })}
        min={info.sampling.cfg.min}
        max={info.sampling.cfg.max}
        step={0.5}
        hint={info.sampling.cfg.hint}
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
