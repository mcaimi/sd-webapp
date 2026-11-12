import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { TrackedJob } from './job-store'

const job = (id: string, status: TrackedJob['status'] = 'queued'): TrackedJob => ({
  id,
  kind: 'generate',
  family: 'sdxl',
  groupId: 'g',
  label: id,
  request: {},
  status,
  submittedAt: new Date().toISOString(),
})

// The store reads localStorage at import time, so import a fresh copy per test.
async function freshStore() {
  vi.resetModules()
  return (await import('./job-store')).jobStore
}

describe('jobStore', () => {
  beforeEach(() => localStorage.clear())

  it('adds newest first, updates and persists', async () => {
    const store = await freshStore()
    store.add([job('a')])
    store.add([job('b')])
    store.update(new Map([['a', { status: 'running' }]]))
    expect(store.getAll().map((j) => [j.id, j.status])).toEqual([
      ['b', 'queued'],
      ['a', 'running'],
    ])
    const reloaded = await freshStore()
    expect(reloaded.getAll().map((j) => j.id)).toEqual(['b', 'a'])
  })

  it('notifies removal listeners and clears only finished jobs', async () => {
    const store = await freshStore()
    const removed = vi.fn()
    store.onRemove(removed)
    store.add([job('done', 'completed'), job('active', 'running'), job('bad', 'failed')])
    store.clearFinished()
    expect(store.getAll().map((j) => j.id)).toEqual(['active'])
    expect(removed.mock.calls[0][0].map((j: TrackedJob) => j.id).sort()).toEqual(['bad', 'done'])
  })

  it('caps history by dropping the oldest finished jobs, never active ones', async () => {
    const store = await freshStore()
    const removed = vi.fn()
    store.onRemove(removed)
    store.add(Array.from({ length: 200 }, (_, i) => job(`old-${i}`, 'completed')))
    store.add([job('new', 'queued')])
    const ids = store.getAll().map((j) => j.id)
    expect(ids).toHaveLength(200)
    expect(ids[0]).toBe('new')
    expect(ids).not.toContain('old-199')
    expect(removed).toHaveBeenCalledWith([expect.objectContaining({ id: 'old-199' })])
  })

  it('survives corrupt storage', async () => {
    localStorage.setItem('sd-webapp.jobs', '{not json')
    expect((await freshStore()).getAll()).toEqual([])
  })
})
