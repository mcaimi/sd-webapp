import { Ban, CircleAlert, Clock, Hourglass, Loader2 } from 'lucide-react'

import { useNow } from '@/hooks/use-now'
import { reportedProgress, type TrackedJob } from '@/jobs/job-store'
import { formatElapsed, formatPercent } from '@/lib/format'
import { cn } from '@/lib/utils'

/**
 * Status line for a job. Only some jobs (downloads) report intermediate
 * progress; the others show elapsed time rather than a made-up percentage.
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
        const progress = reportedProgress(job)
        return (
          <>
            <Loader2 className="animate-spin motion-reduce:animate-none" aria-hidden />
            Running · {progress !== undefined && `${formatPercent(progress * 100)} · `}
            {formatElapsed(now - started)}
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

/** Bottom bar for a running job: determinate when it reports progress, indeterminate otherwise. */
export function JobProgressBar({ job, className }: { job: TrackedJob; className?: string }) {
  const progress = reportedProgress(job)
  if (progress === undefined) return <IndeterminateBar className={className} />
  return (
    <div className={cn('h-0.5 w-full bg-primary/15', className)} aria-hidden>
      <div className="h-full bg-primary transition-[width] duration-500" style={{ width: `${progress * 100}%` }} />
    </div>
  )
}
