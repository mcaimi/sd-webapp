import { ApiError } from '@/api/client'
import type { CivitaiFile, CivitaiPreview, CivitaiVersion, JobDetail } from '@/api/types'

import { formatBytes, formatElapsed } from './format'

/** Civitai's "safe for work" ceiling; previews rated above it are blurred unless NSFW is shown by default. */
const SFW_NSFW_LEVEL = 1
const THUMB_WIDTH = 450

/** Civitai serves resized renditions: swap the original for a thumbnail-sized one. */
export function civitaiThumbUrl(url: string, width = THUMB_WIDTH): string {
  return url.replace('/original=true/', `/width=${width}/`)
}

export function isNsfwPreview(preview: Pick<CivitaiPreview, 'nsfw_level'> | null | undefined): boolean {
  return (preview?.nsfw_level ?? 0) > SFW_NSFW_LEVEL
}

export function civitaiModelUrl(modelId: number, versionId?: number): string {
  return `https://civitai.com/models/${modelId}${versionId ? `?modelVersionId=${versionId}` : ''}`
}

/**
 * Plain text of a Civitai model description (HTML written by model authors).
 * Parsed into an inert document and only its text is kept, so nothing from
 * it is ever rendered as markup.
 */
export function htmlToText(html: string | null | undefined): string {
  const doc = new DOMParser().parseFromString(html ?? '', 'text/html')
  doc.querySelectorAll('br').forEach((br) => br.replaceWith('\n'))
  doc.querySelectorAll('p, li, div, h1, h2, h3, h4, h5, h6').forEach((el) => el.append('\n'))
  return (doc.body.textContent ?? '').replace(/\n{3,}/g, '\n\n').trim()
}

/** Whether a file's Flux component layout can be read remotely (safetensors Flux checkpoints only). */
export function canCheckLayout(version: Pick<CivitaiVersion, 'model_type'>, file: CivitaiFile): boolean {
  return version.model_type === 'flux' && file.target_resource_type === 'checkpoints' && file.format === 'SafeTensor'
}

/** Why POST /civitai/downloads answered 409, from the error detail. */
export type DownloadConflict =
  | { kind: 'mismatch'; message: string; warnings: string[] }
  | { kind: 'downloaded'; message: string; path: string }
  | { kind: 'in-progress'; message: string; jobId: string }
  | { kind: 'other'; message: string }

export function downloadConflict(error: unknown): DownloadConflict | undefined {
  if (!(error instanceof ApiError) || error.status !== 409) return undefined
  const detail = (typeof error.detail === 'object' && error.detail ? error.detail : {}) as {
    warnings?: string[]
    path?: string
    job_id?: string
  }
  const message = error.message
  if (detail.warnings?.length) return { kind: 'mismatch', message, warnings: detail.warnings }
  if (detail.job_id) return { kind: 'in-progress', message, jobId: detail.job_id }
  if (detail.path) return { kind: 'downloaded', message, path: detail.path }
  return { kind: 'other', message }
}

/** "1.2 GB of 2.4 GB · 17 MB/s · 1:20 left", from a running download's job detail. */
export function downloadProgressText(detail: JobDetail | null | undefined): string | undefined {
  if (!detail?.bytes_total) return undefined
  if (detail.phase === 'verifying') return `Verifying ${formatBytes(detail.bytes_total)}…`
  return [
    `${formatBytes(detail.bytes_done ?? 0)} of ${formatBytes(detail.bytes_total)}`,
    detail.speed_bps && `${formatBytes(detail.speed_bps)}/s`,
    detail.eta_s != null && `${formatElapsed(detail.eta_s * 1000)} left`,
  ]
    .filter(Boolean)
    .join(' · ')
}
