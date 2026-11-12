import { Monitor, Moon, Sun } from 'lucide-react'
import { useEffect } from 'react'
import { useNavigate } from 'react-router'

import { useTheme } from '@/components/theme-provider'
import {
  Command,
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
  CommandShortcut,
} from '@/components/ui/command'
import { FAMILY_LIST } from '@/lib/families'
import { PAGES, TOOLS, toolPath } from '@/lib/navigation'
import { useActiveFamily } from '@/lib/use-active-family'

interface CommandMenuProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

/** ⌘K / Ctrl+K palette: navigation, family switching and theme. */
export function CommandMenu({ open, onOpenChange }: CommandMenuProps) {
  const navigate = useNavigate()
  const [family, setFamily] = useActiveFamily()
  const { theme, setTheme } = useTheme()

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === 'k' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        onOpenChange(!open)
      }
    }
    document.addEventListener('keydown', onKeyDown)
    return () => document.removeEventListener('keydown', onKeyDown)
  }, [open, onOpenChange])

  const run = (action: () => void) => {
    onOpenChange(false)
    action()
  }

  return (
    <CommandDialog open={open} onOpenChange={onOpenChange}>
      {/* CommandDialog doesn't provide the cmdk root; items need it for filtering and keyboard nav. */}
      <Command>
        <CommandInput placeholder="Search pages, families, actions…" />
        <CommandList>
          <CommandEmpty>No results found.</CommandEmpty>
          <CommandGroup heading="Go to">
            {TOOLS.map((tool) => (
              <CommandItem
                key={tool.id}
                value={`${tool.label} ${tool.description}`}
                onSelect={() => run(() => navigate(toolPath(tool.id, family)))}
              >
                <tool.icon />
                <span>{tool.label}</span>
                <CommandShortcut className="tracking-normal">{tool.description}</CommandShortcut>
              </CommandItem>
            ))}
            {PAGES.map((page) => (
              <CommandItem
                key={page.path}
                value={`${page.label} ${page.description}`}
                onSelect={() => run(() => navigate(page.path))}
              >
                <page.icon />
                <span>{page.label}</span>
                <CommandShortcut className="tracking-normal">{page.description}</CommandShortcut>
              </CommandItem>
            ))}
          </CommandGroup>
          <CommandSeparator />
          <CommandGroup heading="Model family">
            {FAMILY_LIST.map((f) => (
              <CommandItem
                key={f.id}
                value={`family ${f.label}`}
                data-checked={f.id === family}
                onSelect={() => run(() => setFamily(f.id))}
              >
                <span>{f.label}</span>
              </CommandItem>
            ))}
          </CommandGroup>
          <CommandSeparator />
          <CommandGroup heading="Theme">
            {(
              [
                ['light', 'Light', Sun],
                ['dark', 'Dark', Moon],
                ['system', 'System', Monitor],
              ] as const
            ).map(([value, label, Icon]) => (
              <CommandItem
                key={value}
                value={`theme ${label}`}
                data-checked={theme === value}
                onSelect={() => run(() => setTheme(value))}
              >
                <Icon />
                <span>{label} theme</span>
              </CommandItem>
            ))}
          </CommandGroup>
        </CommandList>
      </Command>
    </CommandDialog>
  )
}
