import { queryOptions, useInfiniteQuery, useQuery } from '@tanstack/react-query'
import { useSyncExternalStore } from 'react'
import { toast } from 'sonner'

import { api, unwrap } from '@/api/client'
import { queryClient } from '@/api/query-client'
import type { OutputItem } from '@/api/schema'
import type { OutputMetadata } from '@/lib/generation'

const PAGE_SIZE = 24
const UNDO_WINDOW_MS = 5_000

// Keys start with 'outputs' so the job sync's invalidation refreshes the gallery.
export const outputKeys = {
  all: ['outputs'] as const,
  list: ['outputs', 'list'] as const,
  metadata: (id: string) => ['outputs', 'metadata', id] as const,
}

export function useOutputs() {
  return useInfiniteQuery({
    queryKey: outputKeys.list,
    queryFn: ({ pageParam }) =>
      unwrap(api.GET('/api/v1/outputs/', { params: { query: { limit: PAGE_SIZE, offset: pageParam } } })),
    initialPageParam: 0,
    getNextPageParam: (last) =>
      last.offset + last.items.length < last.total ? last.offset + last.items.length : undefined,
  })
}

export function outputMetadataQuery(id: string) {
  return queryOptions({
    queryKey: outputKeys.metadata(id),
    queryFn: async () =>
      (await unwrap(
        api.GET('/api/v1/outputs/{output_id}/metadata', { params: { path: { output_id: id } } }),
      )) as OutputMetadata,
    // Saved metadata never changes.
    staleTime: Infinity,
  })
}

export function useOutputMetadata(id: string | undefined) {
  return useQuery({ ...outputMetadataQuery(id ?? ''), enabled: !!id })
}

/**
 * Deletes wait out an undo window before hitting the backend. Timers live at
 * module level so a delete still happens if the user leaves the gallery.
 */
const pending = new Map<string, ReturnType<typeof setTimeout>>()
let pendingSnapshot: ReadonlySet<string> = new Set()
const listeners = new Set<() => void>()

function publish() {
  pendingSnapshot = new Set(pending.keys())
  listeners.forEach((notify) => notify())
}

async function performDelete(id: string) {
  try {
    await unwrap(api.DELETE('/api/v1/outputs/{output_id}', { params: { path: { output_id: id } } }))
    // Keep the item hidden until the refreshed list no longer contains it.
    await queryClient.invalidateQueries({ queryKey: outputKeys.list })
    queryClient.removeQueries({ queryKey: outputKeys.metadata(id) })
  } catch (error) {
    toast.error('Could not delete image', { description: (error as Error).message })
  } finally {
    pending.delete(id)
    publish()
  }
}

export function scheduleDelete(item: OutputItem) {
  if (pending.has(item.id)) return
  pending.set(
    item.id,
    setTimeout(() => performDelete(item.id), UNDO_WINDOW_MS),
  )
  publish()
  toast('Image deleted', {
    duration: UNDO_WINDOW_MS,
    action: { label: 'Undo', onClick: () => undoDelete(item.id) },
  })
}

export function undoDelete(id: string) {
  const timer = pending.get(id)
  if (!timer) return
  clearTimeout(timer)
  pending.delete(id)
  publish()
}

/** Ids whose deletion is pending; hidden from the grid and the lightbox. */
export function usePendingDeletes(): ReadonlySet<string> {
  return useSyncExternalStore(
    (notify) => {
      listeners.add(notify)
      return () => listeners.delete(notify)
    },
    () => pendingSnapshot,
  )
}

// Closing the tab mid-undo-window: send the deletes anyway (best effort).
window.addEventListener('pagehide', () => {
  for (const [id, timer] of pending) {
    clearTimeout(timer)
    fetch(`/api/v1/outputs/${id}`, { method: 'DELETE', keepalive: true }).catch(() => {})
  }
  pending.clear()
})
