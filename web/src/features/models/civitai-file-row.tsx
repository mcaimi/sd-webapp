import { useQueryClient } from '@tanstack/react-query'
import { CircleCheck, Download, Layers, Loader2, TriangleAlert } from 'lucide-react'
import { useState } from 'react'
import { toast } from 'sonner'

import { ApiError } from '@/api/client'
import type { CivitaiFile, CivitaiVersion } from '@/api/types'
import { CancelJobButton } from '@/components/jobs/cancel-job-button'
import { JobProgressBar } from '@/components/jobs/job-status'
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
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { isActive, useTrackedJobs } from '@/jobs/job-store'
import { submitDownload, trackDownload, type DownloadRequest } from '@/jobs/submit'
import { canCheckLayout, downloadConflict, downloadProgressText } from '@/lib/civitai'
import { isResourceType, RESOURCE_LABELS, type FamilyId } from '@/lib/families'
import { formatBytes } from '@/lib/format'
import { fluxComponents } from '@/lib/models'

import { useFileLayout } from './civitai-queries'

interface CivitaiFileRowProps {
  family: FamilyId
  version: CivitaiVersion
  file: CivitaiFile
  modelName: string
}

export function CivitaiFileRow({ family, version, file, modelName }: CivitaiFileRowProps) {
  const queryClient = useQueryClient()
  const [starting, setStarting] = useState(false)
  const [mismatch, setMismatch] = useState<string[]>()
  const job = useTrackedJobs().find((j) => j.kind === 'download' && isActive(j) && j.request.file_id === file.id)
  const resource = isResourceType(file.target_resource_type) ? file.target_resource_type : undefined

  const start = async (acceptMismatch = false) => {
    if (!resource) return
    const request: DownloadRequest = {
      family,
      resource,
      versionId: version.id,
      fileId: file.id,
      modelName,
      acceptMismatch,
    }
    setStarting(true)
    try {
      const started = await submitDownload(request)
      toast.success(`Downloading ${started.filename}`, {
        description: started.warnings?.length ? started.warnings.join(' ') : 'Follow it here or in the Jobs panel.',
      })
    } catch (e) {
      const conflict = downloadConflict(e)
      if (conflict?.kind === 'mismatch') {
        setMismatch(conflict.warnings)
      } else if (conflict?.kind === 'downloaded') {
        toast.info(conflict.message, { description: conflict.path })
        queryClient.invalidateQueries({ queryKey: ['civitai-model'] })
      } else if (conflict?.kind === 'in-progress') {
        trackDownload(conflict.jobId, file.name, request)
        toast.info(conflict.message)
      } else {
        toast.error(`Could not download ${file.name}`, { description: (e as Error).message })
      }
    } finally {
      setStarting(false)
    }
  }

  const details = [file.fp, file.size_variant, file.format].filter(Boolean)
  return (
    <li className="relative space-y-2 overflow-hidden rounded-lg border p-3">
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1 space-y-1">
          <p className="text-sm font-medium break-all">{file.name}</p>
          <div className="flex flex-wrap items-center gap-1 text-xs text-muted-foreground">
            <span className="tabular-nums">{formatBytes(file.size_bytes)}</span>
            {details.map((d) => (
              <Badge key={d} variant="outline" className="h-4 px-1.5 text-[10px]">
                {d}
              </Badge>
            ))}
            {file.primary && (
              <Badge variant="secondary" className="h-4 px-1.5 text-[10px]">
                Primary
              </Badge>
            )}
            {resource && <span>· saved to {RESOURCE_LABELS[resource]}</span>}
          </div>
        </div>
        <div className="shrink-0">
          {job ? (
            <CancelJobButton job={job} />
          ) : file.local_path && file.local_match === 'sha256' ? (
            <span className="inline-flex items-center gap-1 text-xs text-muted-foreground">
              <CircleCheck className="size-3.5 text-emerald-600 dark:text-emerald-400" aria-hidden />
              Downloaded
            </span>
          ) : (
            <Button
              size="sm"
              variant={file.compatible ? 'default' : 'outline'}
              disabled={!file.compatible || !resource || starting}
              onClick={() => start()}
              aria-label={`Download ${file.name}`}
            >
              {starting ? <Loader2 className="animate-spin" /> : <Download />}
              Download
            </Button>
          )}
        </div>
      </div>

      {job && (
        <p className="text-xs text-muted-foreground tabular-nums" aria-live="polite">
          {job.status === 'queued' ? 'Waiting for a download slot…' : (downloadProgressText(job.detail) ?? 'Starting…')}
        </p>
      )}
      {!file.compatible && file.reason && (
        <p className="text-xs text-muted-foreground">Can't download: {file.reason}</p>
      )}
      {file.local_path && file.local_match === 'name' && (
        <p className="text-xs text-muted-foreground">
          A different file with this name is already installed; this one will be saved under a new name.
        </p>
      )}
      {file.compatible && !!file.warnings.length && (
        <ul className="space-y-0.5 text-xs">
          {file.warnings.map((w) => (
            <li key={w} className="flex items-start gap-1.5">
              <TriangleAlert className="mt-0.5 size-3 shrink-0 text-amber-600 dark:text-amber-400" aria-hidden />
              {w}
            </li>
          ))}
        </ul>
      )}
      {canCheckLayout(version, file) && <LayoutCheck versionId={version.id} fileId={file.id} />}
      {job?.status === 'running' && <JobProgressBar job={job} className="absolute inset-x-0 bottom-0" />}

      <AlertDialog open={!!mismatch} onOpenChange={(open) => !open && setMismatch(undefined)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Different base model</AlertDialogTitle>
            <AlertDialogDescription>
              {mismatch?.join(' ')} It may not work, or not work well, with {family} models. Download it anyway?
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={() => start(true)}>Download anyway</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </li>
  )
}

/** Which Flux components a remote checkpoint bundles, read on request from its header. */
function LayoutCheck({ versionId, fileId }: { versionId: number; fileId: number }) {
  const [requested, setRequested] = useState(false)
  const { data, error, isFetching } = useFileLayout(versionId, fileId, requested)

  if (!requested || (isFetching && !data)) {
    return (
      <Button variant="outline" size="xs" onClick={() => setRequested(true)} disabled={isFetching}>
        {isFetching ? <Loader2 className="animate-spin" /> : <Layers />}
        Check components
      </Button>
    )
  }
  if (error) {
    const needsKey = error instanceof ApiError && error.status === 401
    return (
      <p className="text-xs text-muted-foreground">
        {needsKey ? 'Reading this file needs a Civitai API key on the backend.' : error.message}
      </p>
    )
  }
  if (!data.layout.transformer)
    return <p className="text-xs text-muted-foreground">No Flux transformer in this file.</p>
  const { bundled, shared } = fluxComponents(data.layout)
  return (
    <div className="space-y-0.5 text-xs">
      <p>Bundles {bundled.join(', ')}</p>
      {shared.length > 0 && (
        <p className="text-muted-foreground">Uses the shared Flux components for {shared.join(', ')}</p>
      )}
      {data.missing_components.length > 0 && (
        <p className="flex items-start gap-1.5">
          <TriangleAlert className="mt-0.5 size-3 shrink-0 text-amber-600 dark:text-amber-400" aria-hidden />
          Missing from the backend's shared components: {data.missing_components.join(', ')}
        </p>
      )}
    </div>
  )
}
