import { Plus, X } from 'lucide-react'
import { AnimatePresence, motion } from 'motion/react'
import { useState } from 'react'

import { useModels } from '@/api/queries'
import { Button } from '@/components/ui/button'
import { Command, CommandEmpty, CommandInput, CommandItem, CommandList } from '@/components/ui/command'
import { Label } from '@/components/ui/label'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Slider } from '@/components/ui/slider'
import type { FamilyId } from '@/lib/families'
import { MAX_LORAS } from '@/lib/generation'
import { civitaiLabel, modelDisplayName } from '@/lib/models'

import { ModelMetadataButton } from './model-metadata'
import { ModelThumb } from './model-thumb'
import { NsfwBadge } from './nsfw-badge'

type Lora = { name: string; strength: number }

interface LoraListProps {
  family: FamilyId
  value: Lora[]
  onChange: (loras: Lora[]) => void
  /** Add a LoRA's trigger words (known for LoRAs matched on Civitai) to the prompt. */
  onInsertTriggerWords?: (words: string[]) => void
  disabled?: boolean
}

export function LoraList({ family, value, onChange, onInsertTriggerWords, disabled }: LoraListProps) {
  const [open, setOpen] = useState(false)
  const { data = [] } = useModels(family, 'loras')
  const byName = new Map(data.map((m) => [m.name, m]))
  const available = data.filter((m) => !value.some((l) => l.name === m.name))
  const withThumbs = available.some((m) => m.has_preview)
  const setStrength = (name: string, strength: number) =>
    onChange(value.map((l) => (l.name === name ? { ...l, strength } : l)))

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label>
          LoRAs
          <span className="font-normal text-muted-foreground">
            {value.length}/{MAX_LORAS}
          </span>
        </Label>
        <Popover open={open} onOpenChange={setOpen}>
          <PopoverTrigger asChild>
            <Button
              type="button"
              variant="ghost"
              size="xs"
              disabled={disabled || value.length >= MAX_LORAS || available.length === 0}
            >
              <Plus />
              Add
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-72 p-0" align="end">
            <Command>
              <CommandInput placeholder="Search LoRAs…" />
              <CommandList>
                <CommandEmpty>No LoRAs found.</CommandEmpty>
                {available.map((m) => {
                  const civitai = civitaiLabel(m)
                  return (
                    <CommandItem
                      key={m.name}
                      value={m.name}
                      keywords={civitai ? [civitai] : undefined}
                      onSelect={() => {
                        onChange([...value, { name: m.name, strength: 0.7 }])
                        setOpen(false)
                      }}
                    >
                      {withThumbs && <ModelThumb family={family} resource="loras" model={m} className="size-8" />}
                      <span className="min-w-0 flex-1">
                        <span className="block truncate">{modelDisplayName(m.name)}</span>
                        {civitai && <span className="block truncate text-xs text-muted-foreground">{civitai}</span>}
                      </span>
                      {m.civitai?.nsfw && <NsfwBadge />}
                    </CommandItem>
                  )
                })}
              </CommandList>
            </Command>
          </PopoverContent>
        </Popover>
      </div>
      {data.length === 0 && <p className="text-xs text-muted-foreground">No LoRAs installed for this family.</p>}
      <ul className="space-y-2">
        <AnimatePresence initial={false}>
          {value.map((lora) => {
            const model = byName.get(lora.name)
            const civitai = model && civitaiLabel(model)
            const triggerWords = model?.civitai?.trained_words ?? []
            return (
              <motion.li
                key={lora.name}
                layout
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="overflow-hidden"
              >
                <div className="space-y-2 rounded-lg border p-2.5">
                  <div className="flex items-center gap-1">
                    {model?.has_preview && (
                      <ModelThumb family={family} resource="loras" model={model} className="size-7" />
                    )}
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-sm">{modelDisplayName(lora.name)}</span>
                      {civitai && <span className="block truncate text-xs text-muted-foreground">{civitai}</span>}
                    </span>
                    <span className="w-10 text-right text-xs text-muted-foreground tabular-nums">
                      {lora.strength.toFixed(2)}
                    </span>
                    <ModelMetadataButton family={family} resource="loras" name={lora.name} />
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon-sm"
                      aria-label={`Remove ${lora.name}`}
                      onClick={() => onChange(value.filter((l) => l.name !== lora.name))}
                      disabled={disabled}
                    >
                      <X />
                    </Button>
                  </div>
                  <Slider
                    value={[lora.strength]}
                    min={0}
                    max={2}
                    step={0.05}
                    disabled={disabled}
                    onValueChange={([v]) => setStrength(lora.name, v)}
                    aria-label={`${lora.name} strength`}
                  />
                  {onInsertTriggerWords && triggerWords.length > 0 && (
                    <TriggerWords words={triggerWords} onInsert={onInsertTriggerWords} disabled={disabled} />
                  )}
                </div>
              </motion.li>
            )
          })}
        </AnimatePresence>
      </ul>
    </div>
  )
}

const MAX_SHOWN_WORDS = 6

/** A LoRA's trigger words: click one to add it to the prompt, or add them all. */
function TriggerWords({
  words,
  onInsert,
  disabled,
}: {
  words: string[]
  onInsert: (words: string[]) => void
  disabled?: boolean
}) {
  const [showAll, setShowAll] = useState(false)
  const shown = showAll ? words : words.slice(0, MAX_SHOWN_WORDS)
  return (
    <div className="flex flex-wrap items-center gap-1">
      <span className="text-xs text-muted-foreground">Trigger words</span>
      {shown.map((word) => (
        <button
          key={word}
          type="button"
          disabled={disabled}
          onClick={() => onInsert([word])}
          title={`Add "${word}" to the prompt`}
          className="max-w-full truncate rounded-sm border px-1.5 py-0.5 text-xs hover:bg-muted disabled:opacity-50"
        >
          {word}
        </button>
      ))}
      {words.length > MAX_SHOWN_WORDS && (
        <Button type="button" variant="link" size="xs" className="h-auto px-1" onClick={() => setShowAll((s) => !s)}>
          {showAll ? 'Less' : `+${words.length - MAX_SHOWN_WORDS}`}
        </Button>
      )}
      {words.length > 1 && (
        <Button
          type="button"
          variant="ghost"
          size="xs"
          className="h-auto px-1.5 py-0.5"
          disabled={disabled}
          onClick={() => onInsert(words)}
        >
          <Plus />
          All
        </Button>
      )}
    </div>
  )
}
