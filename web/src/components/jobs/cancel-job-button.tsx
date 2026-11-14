import { X } from 'lucide-react'
import { useState } from 'react'
import { toast } from 'sonner'

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import type { TrackedJob } from '@/jobs/job-store'
import { cancelJob } from '@/jobs/submit'

/** True when cancelling only throws the result away: the backend still runs the job to completion. */
function onlyDiscards(job: Pick<TrackedJob, 'status' | 'cancellable'>): boolean {
  return job.status === 'running' && !job.cancellable
}

/**
 * Cancel action for an active job. Queued and interruptible jobs are
 * cancelled right away; for a running job the backend can't interrupt, it
 * asks first and says plainly that only the result is discarded.
 */
export function CancelJobButton({
  job,
  iconOnly,
  className,
}: {
  job: TrackedJob
  /** Compact icon button (job lists) instead of a labelled one (cards). */
  iconOnly?: boolean
  className?: string
}) {
  const [confirming, setConfirming] = useState(false)
  const discard = onlyDiscards(job)
  const label = discard ? 'Discard result' : 'Cancel job'
  const hint = discard ? 'The backend keeps running this job until it finishes; only the result is discarded.' : label

  const cancel = () =>
    cancelJob(job.id).catch((e: Error) => toast.error('Could not cancel', { description: e.message }))

  const button = (
    <Button
      variant="ghost"
      size={iconOnly ? 'icon-sm' : 'xs'}
      aria-label={iconOnly ? label : undefined}
      className={className}
      onClick={() => (discard ? setConfirming(true) : cancel())}
    >
      <X />
      {!iconOnly && (discard ? 'Discard' : 'Cancel')}
    </Button>
  )

  return (
    <>
      <Tooltip>
        <TooltipTrigger asChild>{button}</TooltipTrigger>
        <TooltipContent className="max-w-60">{hint}</TooltipContent>
      </Tooltip>
      <AlertDialog open={confirming} onOpenChange={setConfirming}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Discard this job's result?</AlertDialogTitle>
            <AlertDialogDescription>
              The backend can't stop a job once it has started. It keeps running to completion and keeps the GPU busy
              until then, so queued jobs won't start any sooner. Only its result is thrown away.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Keep running</AlertDialogCancel>
            <AlertDialogAction variant="destructive" onClick={cancel}>
              Discard result
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  )
}
