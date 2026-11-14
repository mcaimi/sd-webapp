import { queryOptions, useQuery } from '@tanstack/react-query'

import { api, ApiError, unwrap } from '@/api/client'
import { queryClient } from '@/api/query-client'
import type { BatchMergeResult, DownloadResult, GenerationInstance, GenerationResult, MergeResult } from '@/api/types'

import { jobStore, type TrackedJob } from './job-store'

export interface ImageJobResult {
  type: 'image'
  /** Object URL of the PNG (the base64 payload is dropped after decoding). */
  imageUrl: string
  seed: number
  instance?: GenerationInstance
}

export interface MergeJobResult {
  type: 'merge'
  result: MergeResult | BatchMergeResult
}

export interface DownloadJobResult {
  type: 'download'
  result: DownloadResult
}

export type JobResult = ImageJobResult | MergeJobResult | DownloadJobResult

const resultKey = (jobId: string) => ['job-result', jobId] as const

function base64ToObjectUrl(base64: string): string {
  const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0))
  return URL.createObjectURL(new Blob([bytes], { type: 'image/png' }))
}

async function fetchResult(job: TrackedJob): Promise<JobResult> {
  try {
    const status = await unwrap(api.GET('/api/v1/jobs/{job_id}', { params: { path: { job_id: job.id } } }))
    const result = status.result as unknown
    if (job.kind === 'merge' || job.kind === 'batch-merge' || job.kind === 'recipe-merge') {
      return { type: 'merge', result: result as MergeResult | BatchMergeResult }
    }
    if (job.kind === 'download') return { type: 'download', result: result as DownloadResult }
    const generation = result as GenerationResult
    return {
      type: 'image',
      imageUrl: base64ToObjectUrl(generation.image_base64),
      seed: generation.seed,
      instance: generation.metadata?.instances?.[0],
    }
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      jobStore.update(new Map([[job.id, { status: 'expired' }]]))
    }
    throw error
  }
}

/** Query options for a job's result; shared by single and list consumers. */
export function jobResultQuery(job: TrackedJob) {
  return queryOptions({
    queryKey: resultKey(job.id),
    queryFn: () => fetchResult(job),
    enabled: job.status === 'completed',
    staleTime: Infinity,
    gcTime: Infinity,
    retry: (count, error) => !(error instanceof ApiError && error.status === 404) && count < 2,
  })
}

/** The result of a completed job, fetched once and kept while the job is tracked. */
export function useJobResult(job: TrackedJob) {
  return useQuery(jobResultQuery(job))
}

function getCachedResult(jobId: string): JobResult | undefined {
  return queryClient.getQueryData(resultKey(jobId))
}

// Results live as long as their job is tracked; release image memory when it goes.
jobStore.onRemove((removed) => {
  for (const job of removed) {
    const cached = getCachedResult(job.id)
    if (cached?.type === 'image') URL.revokeObjectURL(cached.imageUrl)
    queryClient.removeQueries({ queryKey: resultKey(job.id) })
  }
})
