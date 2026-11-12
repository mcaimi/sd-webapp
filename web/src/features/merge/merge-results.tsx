import { Download, FlaskConical, GitMerge, Layers, Sparkles, Trash2, X } from 'lucide-react'
import { AnimatePresence, motion } from 'motion/react'
import { useNavigate } from 'react-router'
import { toast } from 'sonner'

import { IndeterminateBar, JobStatus } from '@/components/jobs/job-status'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { useJobResult } from '@/jobs/job-results'
import { isActive, jobStore, type JobKind, type TrackedJob } from '@/jobs/job-store'
import { cancelJob } from '@/jobs/submit'
import { download } from '@/lib/download'
import type { FamilyId } from '@/lib/families'
import { formatRelative } from '@/lib/format'
import { patchGenerationDraft } from '@/lib/generation-draft'
import { modelDisplayName } from '@/lib/models'

import { basename, checkpointDownloadUrl } from './merge-utils'

const KIND_INFO: Partial<Record<JobKind, { label: string; icon: typeof GitMerge }>> = {
  merge: { label: 'Merge', icon: GitMerge },
  'batch-merge': { label: 'Batch', icon: Layers },
  'recipe-merge': { label: 'Recipe', icon: FlaskConical },
}

export function MergeResults({ jobs, family }: { jobs: TrackedJob[]; family: FamilyId }) {
  if (jobs.length === 0) {
    return (
      <div className="flex min-h-64 flex-col items-center justify-center gap-3 rounded-xl border border-dashed p-8 text-center">
        <div className="flex size-12 items-center justify-center rounded-full bg-primary/10 text-primary">
          <GitMerge className="size-5" aria-hidden />
        </div>
        <h2 className="font-medium">No merges yet</h2>
        <p className="max-w-sm text-sm text-muted-foreground">
          Merges run on the backend and can take several minutes for large checkpoints. They keep going while you use
          the rest of the app; new checkpoints appear in every model picker when done.
        </p>
      </div>
    )
  }
  return (
    <ul className="space-y-3">
      <AnimatePresence initial={false}>
        {jobs.map((job) => (
          <MergeJobItem key={job.id} job={job} family={family} />
        ))}
      </AnimatePresence>
    </ul>
  )
}

function MergeJobItem({ job, family }: { job: TrackedJob; family: FamilyId }) {
  const navigate = useNavigate()
  const result = useJobResult(job)
  const info = KIND_INFO[job.kind] ?? KIND_INFO.merge!
  const merge = result.data?.type === 'merge' ? result.data.result : undefined
  const outputs = merge ? ('output_paths' in merge ? merge.output_paths : [merge.output_path]) : []

  const openInGenerate = (path: string) => {
    patchGenerationDraft('generate', family, { model: basename(path) })
    navigate(`/generate/${family}`)
  }

  return (
    <motion.li
      layout
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, height: 0 }}
      className="relative overflow-hidden rounded-xl border bg-card"
    >
      <div className="space-y-3 p-4">
        <div className="flex items-start gap-3">
          <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-muted">
            <info.icon className="size-4 text-muted-foreground" aria-hidden />
          </div>
          <div className="min-w-0 flex-1 space-y-1">
            <p className="truncate text-sm font-medium" title={job.label}>
              {job.label}
            </p>
            <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-muted-foreground">
              <Badge variant="outline" className="h-4 px-1.5 text-[10px]">
                {info.label}
              </Badge>
              <JobStatus job={job} />
              <span className="text-xs">
                · from {job.model ? modelDisplayName(job.model) : '?'} · {formatRelative(job.submittedAt)}
              </span>
            </div>
          </div>
          {job.status === 'queued' ? (
            <Button
              variant="ghost"
              size="xs"
              onClick={() =>
                cancelJob(job.id).catch((e: Error) => toast.error('Could not cancel', { description: e.message }))
              }
            >
              <X />
              Cancel
            </Button>
          ) : (
            !isActive(job) && (
              <Button
                variant="ghost"
                size="icon-xs"
                aria-label="Remove from list"
                onClick={() => jobStore.remove([job.id])}
              >
                <Trash2 />
              </Button>
            )
          )}
        </div>

        {job.status === 'running' && (
          <p className="text-xs text-muted-foreground">
            Merging on the backend. Large checkpoints can take several minutes.
          </p>
        )}
        {(job.status === 'failed' || job.status === 'expired' || result.isError) && (
          <p className="text-sm text-destructive">
            {job.status === 'expired'
              ? 'The backend no longer has this job. Check the checkpoint list for the output.'
              : (job.error ?? result.error?.message)}
          </p>
        )}

        {outputs.length > 0 && (
          <ul className="space-y-2">
            {outputs.map((path) => (
              <li key={path} className="flex flex-wrap items-center gap-2 rounded-lg bg-muted/50 p-2">
                <span className="min-w-0 flex-1 truncate font-mono text-xs" title={path}>
                  {basename(path)}
                </span>
                <div className="flex gap-1">
                  <Button
                    variant="outline"
                    size="xs"
                    onClick={() => download(checkpointDownloadUrl(family, path), basename(path))}
                  >
                    <Download />
                    Download
                  </Button>
                  <Button variant="outline" size="xs" onClick={() => openInGenerate(path)}>
                    <Sparkles />
                    Use in Generate
                  </Button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
      {job.status === 'running' && <IndeterminateBar className="absolute inset-x-0 bottom-0" />}
    </motion.li>
  )
}
