import { Bell, GitMerge, ListTodo, Trash2, X } from 'lucide-react'
import { AnimatePresence, motion } from 'motion/react'
import { useState } from 'react'
import { useNavigate } from 'react-router'
import { toast } from 'sonner'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from '@/components/ui/sheet'
import { Switch } from '@/components/ui/switch'
import { useJobResult } from '@/jobs/job-results'
import { isActive, JOB_TOOL, jobStore, useTrackedJobs, type TrackedJob } from '@/jobs/job-store'
import { cancelJob } from '@/jobs/submit'
import { notifyPreference, useNotifyPreference } from '@/jobs/use-job-sync'
import { FAMILIES } from '@/lib/families'
import { formatRelative } from '@/lib/format'
import { toolPath } from '@/lib/navigation'

import { IndeterminateBar, JobStatus } from './job-status'

/** Header button + side panel listing every job submitted from this browser. */
export function JobDock() {
  const jobs = useTrackedJobs()
  const [open, setOpen] = useState(false)
  // Running first, then in queue order.
  const active = jobs
    .filter(isActive)
    .sort(
      (a, b) =>
        (a.status === 'running' ? 0 : (a.queuePosition ?? 99)) - (b.status === 'running' ? 0 : (b.queuePosition ?? 99)),
    )
  const finished = jobs.filter((j) => !isActive(j)).slice(0, 50)

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button variant="ghost" size="icon" className="relative" aria-label={`Jobs (${active.length} active)`}>
          <ListTodo />
          <AnimatePresence>
            {active.length > 0 && (
              <motion.span
                key="badge"
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                exit={{ scale: 0 }}
                className="absolute -top-0.5 -right-0.5 flex size-4 items-center justify-center rounded-full bg-primary text-[10px] font-semibold text-primary-foreground"
              >
                {active.length}
              </motion.span>
            )}
          </AnimatePresence>
        </Button>
      </SheetTrigger>
      <SheetContent className="flex w-full flex-col gap-0 p-0 sm:max-w-md">
        <SheetHeader className="border-b">
          <SheetTitle>Jobs</SheetTitle>
          <SheetDescription>Everything submitted from this browser. Keep working while jobs run.</SheetDescription>
          <NotifyToggle />
        </SheetHeader>
        <div className="min-h-0 flex-1 overflow-y-auto">
          <div className="space-y-6 p-4">
            <section>
              <h3 className="mb-2 text-xs font-medium text-muted-foreground uppercase">Active ({active.length})</h3>
              {active.length === 0 ? (
                <p className="text-sm text-muted-foreground">Nothing running.</p>
              ) : (
                <ul className="space-y-2">
                  <AnimatePresence initial={false}>
                    {active.map((job) => (
                      <JobRow key={job.id} job={job} onNavigate={() => setOpen(false)} />
                    ))}
                  </AnimatePresence>
                </ul>
              )}
            </section>
            {finished.length > 0 && (
              <section>
                <div className="mb-2 flex items-center justify-between">
                  <h3 className="text-xs font-medium text-muted-foreground uppercase">Recent</h3>
                  <Button variant="ghost" size="xs" onClick={() => jobStore.clearFinished()}>
                    <Trash2 />
                    Clear finished
                  </Button>
                </div>
                <ul className="space-y-2">
                  <AnimatePresence initial={false}>
                    {finished.map((job) => (
                      <JobRow key={job.id} job={job} onNavigate={() => setOpen(false)} />
                    ))}
                  </AnimatePresence>
                </ul>
              </section>
            )}
          </div>
        </div>
      </SheetContent>
    </Sheet>
  )
}

function NotifyToggle() {
  const enabled = useNotifyPreference()
  if (!('Notification' in window)) return null
  return (
    <div className="flex items-center gap-2 pt-2">
      <Switch
        id="notify-jobs"
        checked={enabled}
        onCheckedChange={async (on) => {
          const granted = await notifyPreference.set(on)
          if (on && !granted) toast.error('Notifications are blocked by the browser')
        }}
      />
      <Label htmlFor="notify-jobs" className="text-sm font-normal">
        <Bell className="size-3.5" aria-hidden />
        Notify me when jobs finish in the background
      </Label>
    </div>
  )
}

function JobRow({ job, onNavigate }: { job: TrackedJob; onNavigate: () => void }) {
  const navigate = useNavigate()
  const goToTool = () => {
    navigate(toolPath(JOB_TOOL[job.kind], job.family))
    onNavigate()
  }

  return (
    <motion.li
      layout
      initial={{ opacity: 0, y: -6 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, height: 0 }}
      className="relative overflow-hidden rounded-lg border bg-card"
    >
      <div className="flex items-center gap-3 p-2">
        <button type="button" onClick={goToTool} className="flex min-w-0 flex-1 items-center gap-3 text-left">
          <JobThumb job={job} />
          <div className="min-w-0 flex-1 space-y-1">
            <p className="truncate text-sm font-medium">{job.label || 'Untitled'}</p>
            <div className="flex items-center gap-2 text-muted-foreground">
              <Badge variant="outline" className="h-4 px-1.5 text-[10px]">
                {FAMILIES[job.family].shortLabel}
              </Badge>
              <JobStatus job={job} />
              {!isActive(job) && <span className="text-xs">· {formatRelative(job.submittedAt)}</span>}
            </div>
          </div>
        </button>
        {job.status === 'queued' ? (
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label="Cancel job"
            onClick={() =>
              cancelJob(job.id).catch((e: Error) => toast.error('Could not cancel', { description: e.message }))
            }
          >
            <X />
          </Button>
        ) : (
          !isActive(job) && (
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label="Remove from list"
              onClick={() => jobStore.remove([job.id])}
            >
              <Trash2 />
            </Button>
          )
        )}
      </div>
      {job.status === 'running' && <IndeterminateBar className="absolute inset-x-0 bottom-0" />}
    </motion.li>
  )
}

function JobThumb({ job }: { job: TrackedJob }) {
  const result = useJobResult(job)
  const isMerge = job.kind.endsWith('merge')
  return (
    <div className="flex size-10 shrink-0 items-center justify-center overflow-hidden rounded-md bg-muted">
      {result.data?.type === 'image' ? (
        <img src={result.data.imageUrl} alt="" className="size-full object-cover" />
      ) : isMerge ? (
        <GitMerge className="size-4 text-muted-foreground" aria-hidden />
      ) : (
        isActive(job) && <div className="shimmer size-full" />
      )}
    </div>
  )
}
