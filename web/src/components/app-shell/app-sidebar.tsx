import { NavLink, useLocation } from 'react-router'

import { ResizeHandle } from '@/components/resize-handle'

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
  useSidebar,
} from '@/components/ui/sidebar'
import { FAMILIES } from '@/lib/families'
import { DEFAULT_LAYOUT, MENU_RANGE, useLayout } from '@/lib/layout'
import { PAGES, TOOLS, toolPath } from '@/lib/navigation'
import { useActiveFamily } from '@/lib/use-active-family'

import { FamilySwitcher } from './family-switcher'
import { HealthIndicator } from './health-indicator'

export function AppSidebar() {
  const [family] = useActiveFamily()
  const { pathname } = useLocation()
  const { isMobile, setOpenMobile, state } = useSidebar()
  const { layout, setMenu } = useLayout()
  // Close the off-canvas drawer after navigating on small screens.
  const onNavigate = () => isMobile && setOpenMobile(false)

  return (
    <Sidebar collapsible="icon" variant="inset">
      <SidebarHeader>
        <FamilySwitcher />
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>{FAMILIES[family].shortLabel} tools</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {TOOLS.map((tool) => (
                <SidebarMenuItem key={tool.id}>
                  <SidebarMenuButton asChild isActive={pathname === toolPath(tool.id, family)} tooltip={tool.label}>
                    <NavLink to={toolPath(tool.id, family)} onClick={onNavigate}>
                      <tool.icon />
                      <span>{tool.label}</span>
                    </NavLink>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
        <SidebarGroup>
          <SidebarGroupLabel>Library</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {PAGES.map((page) => (
                <SidebarMenuItem key={page.path}>
                  <SidebarMenuButton asChild isActive={pathname === page.path} tooltip={page.label}>
                    <NavLink to={page.path} onClick={onNavigate}>
                      <page.icon />
                      <span>{page.label}</span>
                    </NavLink>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter>
        <HealthIndicator />
      </SidebarFooter>
      {/* Expanded: drag to resize. Collapsed: the rail expands it again. */}
      {state === 'expanded' ? (
        <ResizeHandle
          label="Resize navigation"
          value={layout.menu}
          min={MENU_RANGE[0]}
          max={MENU_RANGE[1]}
          onDrag={(x) => setMenu((x / window.innerWidth) * 100)}
          onStep={(delta) => setMenu(layout.menu + delta)}
          onReset={() => setMenu(DEFAULT_LAYOUT.menu)}
          subtle
          className="absolute inset-y-0 right-0 z-20 translate-x-1/2"
        />
      ) : (
        <SidebarRail />
      )}
    </Sidebar>
  )
}
