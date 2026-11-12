import { Check, ChevronsUpDown, Sparkles } from 'lucide-react'

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { SidebarMenu, SidebarMenuButton, SidebarMenuItem, useSidebar } from '@/components/ui/sidebar'
import { FAMILIES, FAMILY_LIST } from '@/lib/families'
import { useActiveFamily } from '@/lib/use-active-family'

export function FamilySwitcher() {
  const [family, setFamily] = useActiveFamily()
  const { isMobile } = useSidebar()
  const active = FAMILIES[family]

  return (
    <SidebarMenu>
      <SidebarMenuItem>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <SidebarMenuButton
              size="lg"
              className="data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground"
            >
              <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-sidebar-primary text-sidebar-primary-foreground">
                <Sparkles className="size-4" aria-hidden />
              </div>
              <div className="grid flex-1 text-left text-sm leading-tight">
                <span className="truncate font-semibold">SD Studio</span>
                <span className="truncate text-xs text-muted-foreground">{active.label}</span>
              </div>
              <ChevronsUpDown className="ml-auto" />
            </SidebarMenuButton>
          </DropdownMenuTrigger>
          <DropdownMenuContent
            className="w-(--radix-dropdown-menu-trigger-width) min-w-64 rounded-lg"
            align="start"
            side={isMobile ? 'bottom' : 'right'}
            sideOffset={4}
          >
            <DropdownMenuLabel className="text-xs text-muted-foreground">Model family</DropdownMenuLabel>
            {FAMILY_LIST.map((f) => (
              <DropdownMenuItem key={f.id} onSelect={() => setFamily(f.id)} className="items-start gap-2 p-2">
                <div className="grid flex-1 gap-0.5">
                  <span className="font-medium">{f.label}</span>
                  <span className="text-xs text-muted-foreground">{f.description}</span>
                </div>
                {f.id === family && <Check className="mt-0.5 size-4" aria-label="Active" />}
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
      </SidebarMenuItem>
    </SidebarMenu>
  )
}
