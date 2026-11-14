import { api, unwrap } from '@/api/client'
import { queryClient } from '@/api/query-client'
import { queryKeys } from '@/api/queries'
import type { BatchGenerateResponse, CompareResponse } from '@/api/types'
import type { FamilyId, ResourceType } from '@/lib/families'
import { seedsFor, toApiParams, type GenerationParams } from '@/lib/generation'

import { jobStore, type JobKind, type TrackedJob } from './job-store'

/**
 * Family-scoped POST. The four families expose identical routes under their
 * own prefix; the per-family request differences (Flux's extra fields) are
 * handled by `toApiParams`, so the body is passed through untyped here.
 */
async function postFamily<T>(family: FamilyId, route: string, body: unknown): Promise<T> {
  return unwrap(api.POST(`/api/v1/${family}/${route}` as '/api/v1/sdxl/generate', { body: body as never })) as T
}

function track(jobs: Omit<TrackedJob, 'status' | 'submittedAt'>[]) {
  const submittedAt = new Date().toISOString()
  jobStore.add(jobs.map((j) => ({ ...j, status: 'queued' as const, submittedAt })))
  // Start polling right away rather than on the next interval.
  queryClient.invalidateQueries({ queryKey: queryKeys.jobs })
}

const newGroupId = () => crypto.randomUUID()

export async function submitGenerate(family: FamilyId, params: GenerationParams): Promise<string[]> {
  const requests = seedsFor(params).map((seed) => ({
    ...toApiParams(family, params, seed),
    width: params.width,
    height: params.height,
  }))
  const jobIds =
    requests.length === 1
      ? [(await postFamily<{ job_id: string }>(family, 'generate', requests[0])).job_id]
      : (await postFamily<BatchGenerateResponse>(family, 'generate/batch', { requests })).job_ids

  const groupId = newGroupId()
  track(
    jobIds.map((id, i) => ({
      id,
      kind: 'generate' as JobKind,
      family,
      groupId,
      label: params.prompt.trim(),
      model: params.model,
      width: params.width,
      height: params.height,
      request: requests[i],
    })),
  )
  return jobIds
}

/** Same prompt and seed across several checkpoints. The API doesn't apply LoRAs or VAEs here. */
export async function submitCompare(family: FamilyId, params: GenerationParams, models: string[]): Promise<string[]> {
  const [seed] = seedsFor({ ...params, count: 1 })
  const base = toApiParams(family, params, seed)
  const request = {
    positive_prompt: base.positive_prompt,
    negative_prompt: base.negative_prompt,
    models,
    width: params.width,
    height: params.height,
    steps: base.steps,
    cfg_scale: base.cfg_scale,
    seed,
    scheduler: base.scheduler,
  }
  const { comparisons } = await postFamily<CompareResponse>(family, 'generate/compare', request)

  const groupId = newGroupId()
  track(
    comparisons.map((c) => ({
      id: c.job_id,
      kind: 'compare' as JobKind,
      family,
      groupId,
      label: params.prompt.trim(),
      model: c.model,
      width: params.width,
      height: params.height,
      request: { ...request, models: undefined, model_checkpoint: c.model },
    })),
  )
  return comparisons.map((c) => c.job_id)
}

export async function submitInpaint(
  family: FamilyId,
  params: GenerationParams,
  image: { base64: string; width: number; height: number },
  maskBase64: string,
  strength?: number,
): Promise<string> {
  const [seed] = seedsFor({ ...params, count: 1 })
  const request = {
    ...toApiParams(family, params, seed),
    image_base64: image.base64,
    mask_base64: maskBase64,
    ...(family === 'flux' && strength !== undefined ? { strength } : {}),
  }
  const { job_id } = await postFamily<{ job_id: string }>(family, 'inpaint', request)

  // Don't persist the images with the tracked job: they'd blow the localStorage quota.
  const { image_base64: _image, mask_base64: _mask, ...rest } = request
  track([
    {
      id: job_id,
      kind: 'inpaint',
      family,
      groupId: newGroupId(),
      label: params.prompt.trim(),
      model: params.model,
      width: image.width,
      height: image.height,
      request: rest,
    },
  ])
  return job_id
}

export interface MergeParams {
  base: string
  target: string
  method: string
  alpha: number
  outputName: string
  preserveMetadata: boolean
}

export async function submitMerge(family: FamilyId, p: MergeParams): Promise<string> {
  const request = {
    model_type: family,
    base_model: p.base,
    target_model: p.target,
    method: p.method,
    alpha: p.alpha,
    output_name: p.outputName,
    preserve_metadata: p.preserveMetadata,
  }
  const { job_id } = await unwrap(api.POST('/api/v1/merge/merge', { body: request }))
  track([{ id: job_id, kind: 'merge', family, groupId: newGroupId(), label: p.outputName, model: p.base, request }])
  return job_id
}

export interface BatchMergeParams {
  base: string
  targets: string[]
  method: string
  alpha: number
  outputSubdir: string
  preserveMetadata: boolean
}

export async function submitBatchMerge(family: FamilyId, p: BatchMergeParams): Promise<string> {
  const request = {
    model_type: family,
    base_model: p.base,
    target_models: p.targets,
    method: p.method,
    alpha: p.alpha,
    output_subdir: p.outputSubdir,
    preserve_metadata: p.preserveMetadata,
  }
  const { job_id } = await unwrap(api.POST('/api/v1/merge/batch', { body: request }))
  track([
    {
      id: job_id,
      kind: 'batch-merge',
      family,
      groupId: newGroupId(),
      label: `${p.base} × ${p.targets.length} models`,
      model: p.base,
      request,
    },
  ])
  return job_id
}

export interface RecipeStep {
  target: string
  method: string
  alpha: number
}

export async function submitRecipeMerge(
  family: FamilyId,
  base: string,
  steps: RecipeStep[],
  outputName: string,
): Promise<string> {
  const request = {
    model_type: family,
    base_model: base,
    steps: steps.map((s) => ({ target_model: s.target, method: s.method, alpha: s.alpha })),
    output_name: outputName,
  }
  const { job_id } = await unwrap(api.POST('/api/v1/merge/recipe', { body: request }))
  track([{ id: job_id, kind: 'recipe-merge', family, groupId: newGroupId(), label: outputName, model: base, request }])
  return job_id
}

export interface DownloadRequest {
  family: FamilyId
  resource: ResourceType
  versionId: number
  fileId: number
  /** Civitai model name, shown alongside the file name. */
  modelName: string
  acceptMismatch?: boolean
}

/**
 * Start downloading a Civitai file into the family's model directories.
 * Rejects with the API's 409 when the base model doesn't match (resend with
 * `acceptMismatch`), the file is already local, or it's already downloading.
 */
export async function submitDownload(download: DownloadRequest) {
  const request = {
    version_id: download.versionId,
    file_id: download.fileId,
    model_type: download.family,
    resource_type: download.resource,
    accept_mismatch: download.acceptMismatch ?? false,
  }
  const started = await unwrap(api.POST('/api/v1/civitai/downloads', { body: request }))
  trackDownload(started.job_id, started.filename, download)
  return started
}

/** Track a download job, e.g. one already running that this browser didn't start. */
export function trackDownload(jobId: string, filename: string, download: DownloadRequest) {
  if (jobStore.getAll().some((j) => j.id === jobId)) return
  track([
    {
      id: jobId,
      kind: 'download',
      family: download.family,
      groupId: newGroupId(),
      label: filename,
      model: download.modelName,
      request: {
        version_id: download.versionId,
        file_id: download.fileId,
        model_type: download.family,
        resource_type: download.resource,
      },
      lane: 'io',
      cancellable: true,
    },
  ])
}

/**
 * Cancel a job. Queued jobs never run. Running jobs are interrupted only when
 * `cancellable` (downloads); other running jobs (generation, merges) keep
 * going on the backend until they finish, and their result is discarded.
 */
export async function cancelJob(jobId: string) {
  await unwrap(api.DELETE('/api/v1/jobs/{job_id}', { params: { path: { job_id: jobId } } }))
  jobStore.update(new Map([[jobId, { status: 'cancelled', queuePosition: undefined }]]))
}

/** Re-run a generation job with the exact same request. Inpaint jobs can't be retried (images aren't kept). */
export async function retryJob(job: TrackedJob): Promise<string> {
  if (job.kind !== 'generate' && job.kind !== 'compare') throw new Error(`Cannot retry ${job.kind} jobs`)
  const { job_id } = await postFamily<{ job_id: string }>(job.family, 'generate', job.request)
  const { id: _id, status: _status, submittedAt: _at, startedAt: _s, completedAt: _c, error: _e, ...rest } = job
  track([{ ...rest, id: job_id, groupId: newGroupId(), queuePosition: undefined }])
  return job_id
}
