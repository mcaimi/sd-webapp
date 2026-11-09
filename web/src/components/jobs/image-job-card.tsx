import { Copy, Download, Expand, RotateCw, SlidersHorizontal, Trash2, X } from 'lucide-react'
import { motion } from 'motion/react'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useJobResult } from '@/jobs/job-results'
import { isActive, jobStore, type TrackedJob } from '@/jobs/job-store'
import { cancelJob, retryJob } from '@/jobs/submit'
import { download } from '@/lib/download'
import { cn } from '@/lib/utils'

import { IndeterminateBar, JobStatus } from './job-status'
import { RevealImage } from './reveal-image'

interface ImageJobCardProps {
  job: TrackedJob
  layoutId: string
  onOpen?: () => void
  /** Load this job's settings back into the form. */
  onUseSettings?: () => void
  /** Show the checkpoint name (useful in comparisons). */
  showModel?: boolean
}

export function ImageJobCard({ job, layoutId, onOpen, onUseSettings, showModel }: ImageJobCardProps) {
  const result = useJobResult(job)
  const image = result.data?.type === 'image' ? result.data : undefined
  const aspectRatio = job.width && job.height ? `${job.width} / ${job.height}` : '1 / 1'
  const filename = `${job.family}_${image?.seed ?? job.id.slice(0, 8)}.png`

  const run = (action: () => Promise<unknown>, failure: string) =>
    action().catch((e: Error) => toast.error(failure, { description: e.message }))

  return (
    <motion.div
      layout
      initial={{ opacity: 0, scale: 0.96 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.96 }}
      transition={{ duration: 0.2 }}
      className="group/card relative overflow-hidden rounded-xl border bg-muted"
      style={{ aspectRatio }}
    >
      {image ? (
        <>
          <button
            type="button"
            className="block size-full cursor-zoom-in"
            onClick={onOpen}
            aria-label={`Open image: ${job.label}`}
          >
            <RevealImage layoutId={layoutId} src={image.imageUrl} alt={job.label} className="size-full object-cover" />
          </button>
          <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-end justify-between gap-2 bg-gradient-to-t from-black/70 to-transparent p-2 pt-8 opacity-0 transition-opacity group-focus-within/card:opacity-100 group-hover/card:opacity-100 [@media(hover:none)]:opacity-100">
            <span className="truncate text-xs text-white/90">
              {showModel && job.model ? `${job.model.replace(/\.safetensors$/, '')} · ` : ''}seed {image.seed}
            </span>
            <div className="pointer-events-auto flex shrink-0 gap-1">
              <CardAction label="Open" onClick={onOpen}>
                <Expand />
              </CardAction>
              {onUseSettings && (
                <CardAction label="Use these settings" onClick={onUseSettings}>
                  <SlidersHorizontal />
                </CardAction>
              )}
              <CardAction
                label="Copy seed"
                onClick={() =>
                  navigator.clipboard.writeText(String(image.seed)).then(() => toast.success('Seed copied'))
                }
              >
                <Copy />
              </CardAction>
              <CardAction label="Download" onClick={() => download(image.imageUrl, filename)}>
                <Download />
              </CardAction>
            </div>
          </div>
        </>
      ) : isActive(job) || (job.status === 'completed' && !result.isError) ? (
        <div className="shimmer flex size-full flex-col items-center justify-center gap-2 p-3 text-center text-muted-foreground">
          <JobStatus job={job} />
          {showModel && job.model && (
            <span className="max-w-full truncate text-xs">{job.model.replace(/\.safetensors$/, '')}</span>
          )}
          {job.status === 'queued' && (
            <Button
              variant="ghost"
              size="xs"
              onClick={() => run(() => cancelJob(job.id), 'Could not cancel job')}
              className="mt-1"
            >
              <X />
              Cancel
            </Button>
          )}
          {job.status === 'running' && <IndeterminateBar className="absolute inset-x-0 bottom-0" />}
        </div>
      ) : (
        <div className="flex size-full flex-col items-center justify-center gap-2 p-4 text-center">
          <JobStatus job={job} className="text-muted-foreground" />
          <p className={cn('line-clamp-3 text-xs text-muted-foreground', job.status === 'failed' && 'text-foreground')}>
            {job.status === 'expired'
              ? 'The result is no longer on the backend. It may still be in the gallery.'
              : result.isError
                ? result.error.message
                : job.error}
          </p>
          <div className="flex gap-1">
            {(job.kind === 'generate' || job.kind === 'compare') && job.status !== 'expired' && (
              <Button variant="outline" size="xs" onClick={() => run(() => retryJob(job), 'Could not retry')}>
                <RotateCw />
                Retry
              </Button>
            )}
            <Button variant="ghost" size="xs" onClick={() => jobStore.remove([job.id])}>
              <Trash2 />
              Remove
            </Button>
          </div>
        </div>
      )}
    </motion.div>
  )
}

function CardAction({ label, onClick, children }: { label: string; onClick?: () => void; children: React.ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant="secondary"
          size="icon-sm"
          className="bg-black/50 text-white backdrop-blur hover:bg-black/70"
          onClick={onClick}
          aria-label={label}
        >
          {children}
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}
