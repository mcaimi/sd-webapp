import { NavLink } from 'react-router'

import { useHealth, useSystemInfo } from '@/api/queries'
import type { QueueStats } from '@/api/types'
import { SidebarMenu, SidebarMenuBadge, SidebarMenuButton, SidebarMenuItem } from '@/components/ui/sidebar'
import { cn } from '@/lib/utils'

/** API status dot + label; status is never conveyed by color alone. */
export function HealthIndicator() {
  const health = useHealth()
  const systemInfo = useSystemInfo()
  const queue = systemInfo.data?.queue_stats as unknown as QueueStats | undefined
  // Generation and merges only; downloads queue separately on the I/O lane.
  const queued = queue?.lanes?.gpu?.queue_size ?? queue?.queue_size

  const state = health.isPending ? 'checking' : health.isError ? 'offline' : 'online'
  const label = {
    checking: 'Connecting…',
    offline: 'API unreachable',
    online: 'API online',
  }[state]
  const detail = state === 'online' && systemInfo.data ? systemInfo.data.device.toUpperCase() : undefined

  return (
    <SidebarMenu>
      <SidebarMenuItem>
        <SidebarMenuButton asChild tooltip={detail ? `${label} · ${detail}` : label}>
          <NavLink to="/system">
            <span className="relative flex size-4 shrink-0 items-center justify-center" aria-hidden>
              {state === 'online' && (
                <span className="absolute size-2.5 animate-ping rounded-full bg-emerald-500/60 motion-reduce:hidden" />
              )}
              <span
                className={cn(
                  'relative size-2.5 rounded-full',
                  state === 'online' && 'bg-emerald-500',
                  state === 'offline' && 'bg-destructive',
                  state === 'checking' && 'bg-muted-foreground',
                )}
              />
            </span>
            <span className="truncate">{label}</span>
            {detail && <span className="ml-auto text-xs text-muted-foreground">{detail}</span>}
          </NavLink>
        </SidebarMenuButton>
        {!!queued && <SidebarMenuBadge title={`${queued} queued jobs`}>{queued}</SidebarMenuBadge>}
      </SidebarMenuItem>
    </SidebarMenu>
  )
}
