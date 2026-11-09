import { AnimatePresence, motion } from 'motion/react'
import { Search } from 'lucide-react'
import { useState } from 'react'
import { useLocation, useMatch, useOutlet } from 'react-router'

import { JobDock } from '@/components/jobs/job-dock'
import { Button } from '@/components/ui/button'
import { Kbd } from '@/components/ui/kbd'
import { Separator } from '@/components/ui/separator'
import { SidebarInset, SidebarProvider, SidebarTrigger } from '@/components/ui/sidebar'
import { useJobSync } from '@/jobs/use-job-sync'
import { FAMILIES, isFamilyId } from '@/lib/families'
import { PAGES, TOOLS } from '@/lib/navigation'
import { MOD_KEY } from '@/lib/platform'

import { AppSidebar } from './app-sidebar'
import { CommandMenu } from './command-menu'
import { ThemeToggle } from './theme-toggle'

function useBreadcrumb(): string[] {
  const { pathname } = useLocation()
  const toolMatch = useMatch('/:tool/:family')
  const tool = TOOLS.find((t) => t.id === toolMatch?.params.tool)
  const family = toolMatch?.params.family
  if (tool && isFamilyId(family)) return [FAMILIES[family].shortLabel, tool.label]
  const page = PAGES.find((p) => p.path === pathname)
  return page ? [page.label] : []
}

/**
 * Route content cross-fades on navigation. `useOutlet()` captures the element
 * for this render, so the exiting page keeps rendering its own content while
 * AnimatePresence plays its exit animation.
 */
function AnimatedOutlet() {
  const { pathname } = useLocation()
  const outlet = useOutlet()

  return (
    <AnimatePresence mode="wait" initial={false}>
      <motion.div
        key={pathname}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -4 }}
        transition={{ duration: 0.18, ease: 'easeOut' }}
        className="flex flex-1 flex-col"
      >
        {outlet}
      </motion.div>
    </AnimatePresence>
  )
}

export function AppShell() {
  const [commandOpen, setCommandOpen] = useState(false)
  const breadcrumb = useBreadcrumb()
  useJobSync()

  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <header className="sticky top-0 z-10 flex h-14 shrink-0 items-center gap-2 rounded-t-xl border-b bg-background/80 px-3 backdrop-blur-md md:px-4">
          <SidebarTrigger className="-ml-1" />
          <Separator orientation="vertical" className="mr-1 data-vertical:h-4 data-vertical:self-center" />
          <nav aria-label="Breadcrumb" className="min-w-0 flex-1">
            <ol className="flex items-center gap-1.5 truncate text-sm">
              {breadcrumb.map((crumb, i) => (
                <li key={crumb} className="flex items-center gap-1.5">
                  {i > 0 && <span className="text-muted-foreground">/</span>}
                  <span className={i === breadcrumb.length - 1 ? 'font-medium' : 'text-muted-foreground'}>{crumb}</span>
                </li>
              ))}
            </ol>
          </nav>
          <Button
            variant="outline"
            className="hidden w-56 justify-start text-muted-foreground sm:flex"
            onClick={() => setCommandOpen(true)}
          >
            <Search />
            <span className="flex-1 text-left">Search…</span>
            <Kbd>{MOD_KEY} K</Kbd>
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="sm:hidden"
            onClick={() => setCommandOpen(true)}
            aria-label="Search"
          >
            <Search />
          </Button>
          <JobDock />
          <ThemeToggle />
        </header>
        <div className="flex flex-1 flex-col p-4 md:p-6">
          <AnimatedOutlet />
        </div>
      </SidebarInset>
      <CommandMenu open={commandOpen} onOpenChange={setCommandOpen} />
    </SidebarProvider>
  )
}
