import { useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef, useSyncExternalStore } from 'react'
import { useNavigate } from 'react-router'
import { toast } from 'sonner'

import { useJobs } from '@/api/queries'
import type { JobState } from '@/api/types'
import { toolPath } from '@/lib/navigation'

import { isActive, JOB_TOOL, jobStore, useTrackedJobs, type TrackedJob } from './job-store'

const NOTIFY_KEY = 'sd-webapp.notify'
const EXPIRY_GRACE_MS = 3_000
const notifyListeners = new Set<() => void>()

/** Opt-in browser notifications for jobs finishing while the tab is hidden. */
export const notifyPreference = {
  get: () => localStorage.getItem(NOTIFY_KEY) === 'on' && 'Notification' in window,
  async set(on: boolean) {
    if (on && 'Notification' in window && Notification.permission !== 'granted') {
      on = (await Notification.requestPermission()) === 'granted'
    }
    localStorage.setItem(NOTIFY_KEY, on ? 'on' : 'off')
    notifyListeners.forEach((fn) => fn())
    return on
  },
  subscribe(fn: () => void) {
    notifyListeners.add(fn)
    return () => {
      notifyListeners.delete(fn)
    }
  },
}

export function useNotifyPreference() {
  return useSyncExternalStore(notifyPreference.subscribe, notifyPreference.get)
}

function describeGroup(jobs: TrackedJob[]): string {
  const { kind, label } = jobs[0]
  if (kind === 'merge' || kind === 'recipe-merge') return `Merge finished: ${label}`
  if (kind === 'batch-merge') return 'Batch merge finished'
  if (kind === 'inpaint') return 'Inpainting ready'
  const done = jobs.filter((j) => j.status === 'completed').length
  return done === 1 ? 'Image ready' : `${done} images ready`
}

/**
 * Keeps tracked jobs in sync with the backend. Mount once (in the app shell).
 *
 * Polls GET /jobs/ (statuses only, no images) while anything is active,
 * announces finished submissions, and refreshes data they affect.
 */
export function useJobSync() {
  const tracked = useTrackedJobs()
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const hasActive = tracked.some(isActive)
  const {
    data: backendJobs,
    isSuccess,
    dataUpdatedAt,
  } = useJobs({ enabled: hasActive, refetchInterval: hasActive ? 1_000 : false })
  const announcedGroups = useRef(new Set<string>())

  useEffect(() => {
    if (!isSuccess || !backendJobs) return
    const byId = new Map(backendJobs.map((j) => [j.job_id, j]))
    const queued = backendJobs
      .filter((j) => j.status === 'queued')
      .sort((a, b) => a.created_at.localeCompare(b.created_at))
      .map((j) => j.job_id)

    const patches = new Map<string, Partial<TrackedJob>>()
    const finished: TrackedJob[] = []
    for (const job of jobStore.getAll()) {
      if (!isActive(job)) continue
      const remote = byId.get(job.id)
      if (!remote) {
        // A response that started before this job was submitted can't contain it.
        if (dataUpdatedAt < Date.parse(job.submittedAt) + EXPIRY_GRACE_MS) continue
        patches.set(job.id, { status: 'expired', error: 'The backend no longer knows this job (restarted?)' })
        finished.push({ ...job, status: 'expired' })
        continue
      }
      const status = remote.status as JobState
      const position = queued.indexOf(job.id)
      const patch: Partial<TrackedJob> = {
        status,
        startedAt: remote.started_at,
        completedAt: remote.completed_at,
        error: remote.error,
        queuePosition: position >= 0 ? position + 1 : undefined,
      }
      if (status !== job.status || patch.queuePosition !== job.queuePosition) patches.set(job.id, patch)
      if (!isActive({ status })) finished.push({ ...job, ...patch })
    }
    jobStore.update(patches)
    if (!finished.length) return

    // Refresh data the finished jobs changed.
    if (finished.some((j) => j.kind.endsWith('merge') && j.status === 'completed')) {
      queryClient.invalidateQueries({ queryKey: ['models'] })
    }
    if (
      finished.some(
        (j) => (j.kind === 'generate' || j.kind === 'compare' || j.kind === 'inpaint') && j.status === 'completed',
      )
    ) {
      queryClient.invalidateQueries({ queryKey: ['outputs'] })
    }

    for (const job of finished) {
      if (job.status === 'failed') {
        toast.error('Job failed', { description: job.error ?? job.label })
      }
    }

    // One announcement per submission, once every job in it is done.
    const all = jobStore.getAll()
    for (const groupId of new Set(finished.map((j) => j.groupId))) {
      const group = all.filter((j) => j.groupId === groupId)
      if (group.some(isActive) || announcedGroups.current.has(groupId)) continue
      announcedGroups.current.add(groupId)
      if (!group.some((j) => j.status === 'completed')) continue
      const title = describeGroup(group)
      const path = toolPath(JOB_TOOL[group[0].kind], group[0].family)
      // Merge titles already name the output; generation titles don't name the prompt.
      const description = group[0].kind.endsWith('merge') ? undefined : group[0].label
      toast.success(title, {
        description,
        action: { label: 'View', onClick: () => navigate(path) },
      })
      if (document.hidden && notifyPreference.get() && Notification.permission === 'granted') {
        const notification = new Notification(title, { body: description, icon: '/favicon.svg' })
        notification.onclick = () => {
          window.focus()
          navigate(path)
        }
      }
    }
  }, [backendJobs, isSuccess, dataUpdatedAt, navigate, queryClient])
}
