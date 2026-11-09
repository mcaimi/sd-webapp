import { Check, ChevronsUpDown } from 'lucide-react'
import { useState } from 'react'

import { useModels } from '@/api/queries'
import { Button } from '@/components/ui/button'
import { Command, CommandEmpty, CommandInput, CommandItem, CommandList } from '@/components/ui/command'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import type { FamilyId, ResourceType } from '@/lib/families'
import { modelDisplayName as displayName } from '@/lib/models'
import { cn } from '@/lib/utils'

interface ModelComboboxProps {
  family: FamilyId
  resource: ResourceType
  value: string
  onChange: (name: string) => void
  placeholder?: string
  /** Names to hide (e.g. the base model when picking a merge target). */
  exclude?: string[]
  /** Offer an empty choice with this label (e.g. "Checkpoint default" for VAEs). */
  noneLabel?: string
  id?: string
  disabled?: boolean
}

/** Searchable picker over the backend's models for a family. */
export function ModelCombobox({
  family,
  resource,
  value,
  onChange,
  placeholder = 'Select a model',
  exclude = [],
  noneLabel,
  id,
  disabled,
}: ModelComboboxProps) {
  const [open, setOpen] = useState(false)
  const { data = [], isPending, isError } = useModels(family, resource)
  const options = data.filter((m) => !exclude.includes(m.name))
  const select = (name: string) => {
    onChange(name)
    setOpen(false)
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          id={id}
          type="button"
          variant="outline"
          role="combobox"
          aria-expanded={open}
          disabled={disabled}
          className="w-full min-w-0 justify-between font-normal"
        >
          <span className={cn('truncate', !value && 'text-muted-foreground')}>
            {value ? displayName(value) : isPending ? 'Loading…' : (noneLabel ?? placeholder)}
          </span>
          <ChevronsUpDown className="opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-(--radix-popover-trigger-width) min-w-64 p-0" align="start">
        <Command>
          <CommandInput placeholder="Search…" />
          <CommandList>
            <CommandEmpty>{isError ? 'Could not load models.' : 'No models found.'}</CommandEmpty>
            {noneLabel && (
              <CommandItem value={`__none__ ${noneLabel}`} onSelect={() => select('')} data-checked={!value}>
                {noneLabel}
              </CommandItem>
            )}
            {options.map((m) => (
              <CommandItem key={m.name} value={m.name} onSelect={() => select(m.name)}>
                <span className="truncate">{displayName(m.name)}</span>
                <Check className={cn('ml-auto', value === m.name ? 'opacity-100' : 'opacity-0')} />
              </CommandItem>
            ))}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}
