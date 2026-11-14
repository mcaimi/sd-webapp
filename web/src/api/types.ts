/**
 * Hand-written shapes for API payloads the OpenAPI spec leaves untyped
 * (`result: object`, `detail: object`, batch/compare responses, system stats).
 *
 * They mirror sd-api-backend's actual responses (api/routers/*.py,
 * api/services/generation.py, api/services/merge.py). Delete each one once
 * the backend declares a response model and `npm run api:sync` picks it up.
 */
/** Declared by the backend; re-exported so callers import API types from one place. */
export type {
  CivitaiFile,
  CivitaiModelDetail,
  CivitaiModelSummary,
  CivitaiPreview,
  CivitaiStatusResponse,
  CivitaiVersion,
  FluxLayout,
  ModelCivitaiInfo,
  ModelEntry,
} from './schema'

export type JobState = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'

/** Worker pool a job runs on: `gpu` for generation and merges, `io` for downloads. */
export type JobLane = 'gpu' | 'io'

/** `JobStatusResponse.detail`: free-form progress info, published by downloads. */
export interface JobDetail {
  phase?: string
  filename?: string
  bytes_done?: number
  bytes_total?: number
  speed_bps?: number
  eta_s?: number | null
}

/** One entry of `GenerationMetadata.instances`, as produced by metastable. */
export interface GenerationInstance {
  prompt: string
  negative_prompt: string
  num_inference_steps: number
  width: number
  height: number
  guidance_scale: number
  seed: number
  scheduler: string | null
}

/** `result` of a completed generate/inpaint job. */
export interface GenerationResult {
  image_base64: string
  metadata: { instances: GenerationInstance[] }
  seed: number
}

/** `result` of a completed merge job (single or recipe). */
export interface MergeResult {
  output_path: string
  base_model: string
  model_type: string
  target_model?: string
  method?: string
  alpha?: number
  steps?: unknown[]
  step_count?: number
}

/** `result` of a completed batch merge job. */
export interface BatchMergeResult {
  output_paths: string[]
  base_model: string
  target_models: string[]
  method: string
  alpha: number
  model_type: string
  count: number
}

/** `result` of a completed Civitai download job. */
export interface DownloadResult {
  path: string
  filename: string
  model_type: string
  resource_type: string
  sha256: string
  size_bytes: number
  civitai_model_id?: number | null
  civitai_version_id?: number | null
  flux_layout?: Record<string, unknown>
}

/** Response of POST /api/v1/{family}/generate/batch. */
export interface BatchGenerateResponse {
  job_ids: string[]
  message: string
}

/** Response of POST /api/v1/{family}/generate/compare. */
export interface CompareResponse {
  comparisons: { model: string; job_id: string }[]
  message: string
}

/** `SystemInfo.gpu_stats` (api/routers/system.py, metastable/telemetry.py). */
export interface GpuStats {
  available: boolean
  used_bytes?: number
  total_bytes?: number
  allocated_bytes?: number
  reserved_bytes?: number
  /** True when the backend can only estimate total memory (e.g. Apple unified memory). */
  total_is_approximate?: boolean
  compute?: { available: boolean; utilization_percent?: number; reason?: string }
}

/** `SystemInfo.host_stats`. */
export interface HostStats {
  cpu_percent: number
  cpu_count: number
  memory_used_bytes: number
  memory_total_bytes: number
  memory_percent: number
  load_average?: number[]
}

/** One entry of `SystemInfo.queue_stats.lanes`. */
export interface LaneStats {
  queue_size: number
  num_workers: number
  running_jobs: number
}

/** `SystemInfo.queue_stats`. `queue_size` sums every lane; `num_workers` counts GPU workers. */
export interface QueueStats {
  queue_size: number
  total_jobs: number
  num_workers: number
  workers_running: boolean
  lanes?: Partial<Record<JobLane, LaneStats>>
}

/** `SystemInfo.cache_stats`. */
export interface CacheStats {
  cached_models: number
  max_models: number
  /** Cached pipelines currently leased by running jobs. */
  in_use?: number
  cache_keys: string[]
}
