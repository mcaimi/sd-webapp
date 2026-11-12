import { NavLink, useLocation } from 'react-router'

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
import { PAGES, TOOLS, toolPath } from '@/lib/navigation'
import { useActiveFamily } from '@/lib/use-active-family'

import { FamilySwitcher } from './family-switcher'
import { HealthIndicator } from './health-indicator'

export function AppSidebar() {
  const [family] = useActiveFamily()
  const { pathname } = useLocation()
  const { isMobile, setOpenMobile } = useSidebar()
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
      <SidebarRail />
    </Sidebar>
  )
}
