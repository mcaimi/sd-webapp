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
import { modelDisplayName } from '@/lib/models'

import { ModelMetadataButton } from './model-metadata'

type Lora = { name: string; strength: number }

interface LoraListProps {
  family: FamilyId
  value: Lora[]
  onChange: (loras: Lora[]) => void
  disabled?: boolean
}

export function LoraList({ family, value, onChange, disabled }: LoraListProps) {
  const [open, setOpen] = useState(false)
  const { data = [] } = useModels(family, 'loras')
  const available = data.filter((m) => !value.some((l) => l.name === m.name))
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
                {available.map((m) => (
                  <CommandItem
                    key={m.name}
                    value={m.name}
                    onSelect={() => {
                      onChange([...value, { name: m.name, strength: 0.7 }])
                      setOpen(false)
                    }}
                  >
                    {modelDisplayName(m.name)}
                  </CommandItem>
                ))}
              </CommandList>
            </Command>
          </PopoverContent>
        </Popover>
      </div>
      {data.length === 0 && <p className="text-xs text-muted-foreground">No LoRAs installed for this family.</p>}
      <ul className="space-y-2">
        <AnimatePresence initial={false}>
          {value.map((lora) => (
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
                  <span className="min-w-0 flex-1 truncate text-sm">{modelDisplayName(lora.name)}</span>
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
              </div>
            </motion.li>
          ))}
        </AnimatePresence>
      </ul>
    </div>
  )
}
