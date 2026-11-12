import { useCallback, useSyncExternalStore } from 'react'

/**
 * Persisted form drafts, keyed per tool and family (e.g. `generate:sdxl`).
 *
 * Keeps half-written prompts and settings across navigation and reloads, and
 * lets other pages (gallery "remix", merge "use in generate") write into a
 * tool's form before navigating to it.
 */

const PREFIX = 'sd-webapp.draft.'
const listeners = new Map<string, Set<() => void>>()
const cache = new Map<string, unknown>()

function read<T>(key: string, fallback: () => T): T {
  if (!cache.has(key)) {
    let value: T = fallback()
    try {
      const stored = localStorage.getItem(PREFIX + key)
      // Merge over defaults so drafts saved by older versions gain new fields.
      if (stored) value = { ...value, ...JSON.parse(stored) }
    } catch {
      // Corrupt draft: start fresh.
    }
    cache.set(key, value)
  }
  return cache.get(key) as T
}

export function writeDraft<T extends object>(key: string, fallback: () => T, patch: Partial<T>) {
  const next = { ...read(key, fallback), ...patch }
  cache.set(key, next)
  try {
    localStorage.setItem(PREFIX + key, JSON.stringify(next))
  } catch {
    // Quota exceeded: the draft still lives in memory.
  }
  listeners.get(key)?.forEach((notify) => notify())
}

export function useDraft<T extends object>(key: string, fallback: () => T): [T, (patch: Partial<T>) => void] {
  const subscribe = useCallback(
    (notify: () => void) => {
      if (!listeners.has(key)) listeners.set(key, new Set())
      listeners.get(key)!.add(notify)
      return () => {
        listeners.get(key)!.delete(notify)
      }
    },
    [key],
  )
  // `fallback` is only used the first time a key is read, so it needn't be stable.
  const value = useSyncExternalStore(subscribe, () => read(key, fallback))
  const update = useCallback((patch: Partial<T>) => writeDraft(key, fallback, patch), [key]) // eslint-disable-line react-hooks/exhaustive-deps
  return [value, update]
}
