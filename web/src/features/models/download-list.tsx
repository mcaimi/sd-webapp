import { CircleCheck, Trash2 } from 'lucide-react'
import { AnimatePresence, motion } from 'motion/react'

import { CancelJobButton } from '@/components/jobs/cancel-job-button'
import { JobProgressBar, JobStatus } from '@/components/jobs/job-status'
import { Button } from '@/components/ui/button'
import { isActive, jobStore, useTrackedJobs, type TrackedJob } from '@/jobs/job-store'
import { downloadProgressText } from '@/lib/civitai'
import { isResourceType, RESOURCE_LABELS, type FamilyId } from '@/lib/families'

const MAX_FINISHED = 5

const isFamilyDownload = (family: FamilyId) => (j: TrackedJob) => j.kind === 'download' && j.family === family

/** Civitai downloads of a family started from this browser: running ones and the latest finished. */
export function DownloadList({ family }: { family: FamilyId }) {
  const downloads = useTrackedJobs().filter(isFamilyDownload(family))
  const active = downloads.filter(isActive)
  const finished = downloads.filter((j) => !isActive(j)).slice(0, MAX_FINISHED)
  if (!active.length && !finished.length) return null

  return (
    <section aria-label="Downloads" className="space-y-2">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium text-muted-foreground">Downloads</h2>
        {finished.length > 0 && (
          <Button variant="ghost" size="xs" onClick={() => jobStore.clearFinished(isFamilyDownload(family))}>
            <Trash2 />
            Clear finished
          </Button>
        )}
      </div>
      <ul className="space-y-2">
        <AnimatePresence initial={false}>
          {[...active, ...finished].map((job) => (
            <DownloadRow key={job.id} job={job} />
          ))}
        </AnimatePresence>
      </ul>
    </section>
  )
}

function DownloadRow({ job }: { job: TrackedJob }) {
  const resource = job.request.resource_type as string
  return (
    <motion.li
      layout
      initial={{ opacity: 0, y: -6 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, height: 0 }}
      className="relative flex items-center gap-3 overflow-hidden rounded-lg border bg-card p-3"
    >
      <div className="min-w-0 flex-1 space-y-1">
        <p className="truncate text-sm font-medium">{job.label}</p>
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
          {job.status === 'completed' ? (
            <span className="inline-flex items-center gap-1">
              <CircleCheck className="size-3.5 text-emerald-600 dark:text-emerald-400" aria-hidden />
              Downloaded
            </span>
          ) : (
            <JobStatus job={job} />
          )}
          {job.model && <span className="truncate">· {job.model}</span>}
          {isResourceType(resource) && <span>· {RESOURCE_LABELS[resource]}</span>}
        </div>
        {job.status === 'running' && (
          <p className="text-xs text-muted-foreground tabular-nums">
            {downloadProgressText(job.detail) ?? 'Starting…'}
          </p>
        )}
        {job.status === 'failed' && job.error && <p className="text-xs break-words">{job.error}</p>}
      </div>
      {isActive(job) ? (
        <CancelJobButton job={job} />
      ) : (
        <Button variant="ghost" size="icon-xs" aria-label="Remove from list" onClick={() => jobStore.remove([job.id])}>
          <Trash2 />
        </Button>
      )}
      {job.status === 'running' && <JobProgressBar job={job} className="absolute inset-x-0 bottom-0" />}
    </motion.li>
  )
}
