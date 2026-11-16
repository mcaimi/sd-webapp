import { useSyncExternalStore } from 'react'

import type { JobDetail, JobLane, JobState } from '@/api/types'
import type { FamilyId } from '@/lib/families'
import type { ToolId } from '@/lib/navigation'

/**
 * Jobs submitted from this browser.
 *
 * The backend keeps job state (and results, for an hour) but has no notion of
 * who submitted what, so the client tracks its own jobs: what was asked for,
 * how to group them in the UI, and their last known status. Persisted to
 * localStorage so a reload keeps the queue and recent results visible.
 */

export type JobKind = 'generate' | 'compare' | 'inpaint' | 'merge' | 'batch-merge' | 'recipe-merge' | 'download'

/** `expired`: the backend no longer knows the job (result TTL elapsed or backend restarted). */
export type TrackedStatus = JobState | 'expired'

export interface TrackedJob {
  id: string
  kind: JobKind
  family: FamilyId
  /** Jobs submitted together (a batch, a comparison) share a group. */
  groupId: string
  /** Short human label: the prompt, a merge's output name, or a download's file name. */
  label: string
  /** Checkpoint used (generation), base model (merges), or Civitai model name (downloads). */
  model?: string
  /** Expected output size, so placeholders can reserve the right aspect ratio. */
  width?: number
  height?: number
  /** Submitted request payload (large fields like images stripped), for retry and remix. */
  request: Record<string, unknown>
  status: TrackedStatus
  submittedAt: string
  startedAt?: string | null
  completedAt?: string | null
  error?: string | null
  /** 1-based position among queued jobs of the same lane; undefined once running. */
  queuePosition?: number
  lane?: JobLane
  /** Whether cancelling stops the job while it runs, rather than only discarding its result. */
  cancellable?: boolean
  /** 0-1, when the backend reports it (0 otherwise until the job completes). */
  progress?: number
  detail?: JobDetail | null
}

export const JOB_TOOL: Record<JobKind, ToolId> = {
  generate: 'generate',
  compare: 'generate',
  inpaint: 'inpaint',
  merge: 'merge',
  'batch-merge': 'merge',
  'recipe-merge': 'merge',
  download: 'models',
}

const STORAGE_KEY = 'sd-webapp.jobs'
const MAX_TRACKED = 200

export function isActive(job: Pick<TrackedJob, 'status'>): boolean {
  return job.status === 'queued' || job.status === 'running'
}

/** Progress (0-1) when the backend reports it for a running job, else undefined. */
export function reportedProgress(job: Pick<TrackedJob, 'status' | 'progress'>): number | undefined {
  return job.status === 'running' && job.progress && job.progress > 0 && job.progress < 1 ? job.progress : undefined
}

function load(): TrackedJob[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]')
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

let jobs: TrackedJob[] = load()
const listeners = new Set<() => void>()
const removeListeners = new Set<(removed: TrackedJob[]) => void>()

function commit(next: TrackedJob[]) {
  // Newest first; drop the oldest finished jobs beyond the cap.
  if (next.length > MAX_TRACKED) {
    const keep = next.filter(isActive)
    const finished = next.filter((j) => !isActive(j))
    const dropped = finished.slice(MAX_TRACKED - keep.length)
    next = next.filter((j) => !dropped.includes(j))
    removeListeners.forEach((fn) => fn(dropped))
  }
  jobs = next
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(jobs))
  } catch {
    // Quota exceeded: keep working in memory.
  }
  listeners.forEach((notify) => notify())
}

export const jobStore = {
  getAll: () => jobs,

  add(newJobs: TrackedJob[]) {
    commit([...newJobs, ...jobs])
  },

  update(patches: Map<string, Partial<TrackedJob>>) {
    if (!patches.size) return
    commit(jobs.map((j) => (patches.has(j.id) ? { ...j, ...patches.get(j.id) } : j)))
  },

  remove(ids: Iterable<string>) {
    const idSet = new Set(ids)
    const removed = jobs.filter((j) => idSet.has(j.id))
    if (!removed.length) return
    commit(jobs.filter((j) => !idSet.has(j.id)))
    removeListeners.forEach((fn) => fn(removed))
  },

  clearFinished(predicate: (job: TrackedJob) => boolean = () => true) {
    jobStore.remove(jobs.filter((j) => !isActive(j) && predicate(j)).map((j) => j.id))
  },

  subscribe(notify: () => void) {
    listeners.add(notify)
    return () => {
      listeners.delete(notify)
    }
  },

  /** Called with jobs leaving the store, e.g. to release their cached images. */
  onRemove(fn: (removed: TrackedJob[]) => void) {
    removeListeners.add(fn)
    return () => {
      removeListeners.delete(fn)
    }
  },
}

/** All tracked jobs, newest first. Re-renders only when the store changes. */
export function useTrackedJobs(): TrackedJob[] {
  return useSyncExternalStore(jobStore.subscribe, jobStore.getAll)
}
