import { useQueries } from '@tanstack/react-query'
import { Copy, Download, SlidersHorizontal, Trash2 } from 'lucide-react'
import { AnimatePresence, motion } from 'motion/react'
import { useMemo, useState, type ReactNode } from 'react'
import { toast } from 'sonner'

import { DetailList, Lightbox, type LightboxItem } from '@/components/lightbox'
import { Button } from '@/components/ui/button'
import { jobResultQuery, type ImageJobResult } from '@/jobs/job-results'
import { isActive, jobStore, type TrackedJob } from '@/jobs/job-store'
import { download } from '@/lib/download'
import { formatRelative } from '@/lib/format'
import { modelDisplayName } from '@/lib/models'

import { ImageJobCard } from './image-job-card'

interface ImageResultsProps {
  jobs: TrackedJob[]
  emptyState: ReactNode
  /** Load a job's settings back into the form. */
  onUseSettings?: (job: TrackedJob) => void
  /** Extra content shown above the details in the lightbox (e.g. a before/after view). */
  renderDetails?: (job: TrackedJob, result: ImageJobResult) => ReactNode
}

interface Group {
  id: string
  jobs: TrackedJob[]
}

/** Feed of image jobs, newest submission first, with a lightbox over finished images. */
export function ImageResults({ jobs, emptyState, onUseSettings, renderDetails }: ImageResultsProps) {
  const groups = useMemo(() => {
    const byId = new Map<string, Group>()
    for (const job of jobs) {
      if (!byId.has(job.groupId)) byId.set(job.groupId, { id: job.groupId, jobs: [] })
      byId.get(job.groupId)!.jobs.push(job)
    }
    // Jobs are stored newest first; show a group's images in submission order.
    return [...byId.values()].map((g) => ({ ...g, jobs: [...g.jobs].reverse() }))
  }, [jobs])

  const ordered = useMemo(() => groups.flatMap((g) => g.jobs), [groups])
  const results = useQueries({ queries: ordered.map(jobResultQuery) })
  const [openIndex, setOpenIndex] = useState<number | null>(null)

  const lightboxItems: (LightboxItem & { jobId: string })[] = ordered.flatMap((job, i) => {
    const result = results[i]?.data
    if (result?.type !== 'image') return []
    return [toLightboxItem(job, result, onUseSettings, renderDetails)]
  })
  const openJob = (jobId: string) => setOpenIndex(lightboxItems.findIndex((item) => item.jobId === jobId))

  if (groups.length === 0) return <>{emptyState}</>

  return (
    <>
      <div className="space-y-8">
        <AnimatePresence initial={false}>
          {groups.map((group) => {
            const first = group.jobs[0]
            const models = new Set(group.jobs.map((j) => j.model))
            return (
              <motion.section
                key={group.id}
                layout="position"
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, height: 0, marginTop: 0 }}
                className="space-y-3"
              >
                <header className="flex items-start gap-3">
                  <div className="min-w-0 flex-1">
                    <p className="line-clamp-2 text-sm font-medium" title={first.label}>
                      {first.label || 'Untitled'}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {group.jobs.length > 1 ? `${group.jobs.length} images · ` : ''}
                      {models.size === 1 && first.model ? `${modelDisplayName(first.model)} · ` : ''}
                      {first.width}×{first.height} · {formatRelative(first.submittedAt)}
                    </p>
                  </div>
                  <div className="flex shrink-0 gap-1">
                    {onUseSettings && (
                      <Button variant="ghost" size="xs" onClick={() => onUseSettings(first)}>
                        <SlidersHorizontal />
                        <span className="hidden sm:inline">Reuse settings</span>
                      </Button>
                    )}
                    {!group.jobs.some(isActive) && (
                      <Button
                        variant="ghost"
                        size="icon-xs"
                        aria-label="Remove from results"
                        onClick={() => jobStore.remove(group.jobs.map((j) => j.id))}
                      >
                        <Trash2 />
                      </Button>
                    )}
                  </div>
                </header>
                <div className="grid grid-cols-[repeat(auto-fill,minmax(min(100%,11rem),1fr))] items-start gap-3">
                  <AnimatePresence initial={false}>
                    {group.jobs.map((job) => (
                      <ImageJobCard
                        key={job.id}
                        job={job}
                        layoutId={`job-image-${job.id}`}
                        onOpen={() => openJob(job.id)}
                        onUseSettings={onUseSettings && (() => onUseSettings(job))}
                        showModel={models.size > 1}
                      />
                    ))}
                  </AnimatePresence>
                </div>
              </motion.section>
            )
          })}
        </AnimatePresence>
      </div>
      <Lightbox items={lightboxItems} index={openIndex} onIndexChange={setOpenIndex} />
    </>
  )
}

function toLightboxItem(
  job: TrackedJob,
  result: ImageJobResult,
  onUseSettings?: (job: TrackedJob) => void,
  renderDetails?: ImageResultsProps['renderDetails'],
): LightboxItem & { jobId: string } {
  const instance = result.instance
  const request = job.request as { loras?: { name: string; strength: number }[] | null; custom_vae?: string | null }
  return {
    key: job.id,
    jobId: job.id,
    src: result.imageUrl,
    alt: job.label || 'Generated image',
    layoutId: `job-image-${job.id}`,
    title: <span className="line-clamp-4">{job.label}</span>,
    actions: (
      <>
        <Button size="sm" onClick={() => download(result.imageUrl, `${job.family}_${result.seed}.png`)}>
          <Download />
          Download
        </Button>
        {onUseSettings && (
          <Button size="sm" variant="outline" onClick={() => onUseSettings(job)}>
            <SlidersHorizontal />
            Reuse settings
          </Button>
        )}
        <Button
          size="sm"
          variant="outline"
          onClick={() => navigator.clipboard.writeText(String(result.seed)).then(() => toast.success('Seed copied'))}
        >
          <Copy />
          Seed
        </Button>
      </>
    ),
    details: (
      <>
        {renderDetails?.(job, result)}
        <DetailList
          entries={[
            ['Model', job.model && modelDisplayName(job.model)],
            ['Negative', instance?.negative_prompt],
            ['Size', instance && `${instance.width}×${instance.height}`],
            ['Steps', instance?.num_inference_steps],
            ['CFG', instance?.guidance_scale],
            ['Scheduler', instance?.scheduler],
            ['Seed', result.seed],
            ['LoRAs', request.loras?.map((l) => `${modelDisplayName(l.name)} (${l.strength})`).join(', ')],
            ['VAE', request.custom_vae && modelDisplayName(request.custom_vae)],
          ]}
        />
      </>
    ),
  }
}
