import { Ban, CircleAlert, Clock, Hourglass, Loader2 } from 'lucide-react'

import { useNow } from '@/hooks/use-now'
import type { TrackedJob } from '@/jobs/job-store'
import { formatElapsed } from '@/lib/format'
import { cn } from '@/lib/utils'

/**
 * Status line for a job. The backend reports no intermediate progress, so a
 * running job shows elapsed time rather than a made-up percentage.
 */
export function JobStatus({ job, className }: { job: TrackedJob; className?: string }) {
  const now = useNow(job.status === 'running' || job.status === 'queued')

  const content = (() => {
    switch (job.status) {
      case 'queued':
        return (
          <>
            <Clock aria-hidden />
            {job.queuePosition ? `Queued · #${job.queuePosition}` : 'Queued'}
          </>
        )
      case 'running': {
        const started = Date.parse(job.startedAt ?? job.submittedAt)
        return (
          <>
            <Loader2 className="animate-spin motion-reduce:animate-none" aria-hidden />
            Running · {formatElapsed(now - started)}
          </>
        )
      }
      case 'failed':
        return (
          <>
            <CircleAlert className="text-destructive" aria-hidden />
            Failed
          </>
        )
      case 'cancelled':
        return (
          <>
            <Ban aria-hidden />
            Cancelled
          </>
        )
      case 'expired':
        return (
          <>
            <Hourglass aria-hidden />
            Expired
          </>
        )
      case 'completed':
        return job.startedAt && job.completedAt ? (
          <>Took {formatElapsed(Date.parse(job.completedAt) - Date.parse(job.startedAt))}</>
        ) : (
          <>Done</>
        )
    }
  })()

  return <span className={cn('inline-flex items-center gap-1.5 text-xs [&_svg]:size-3.5', className)}>{content}</span>
}

/** Thin animated bar for "working, duration unknown". */
export function IndeterminateBar({ className }: { className?: string }) {
  return (
    <div className={cn('h-0.5 w-full overflow-hidden bg-primary/15', className)} aria-hidden>
      <div className="indeterminate-bar h-full w-1/3 bg-primary" />
    </div>
  )
}
