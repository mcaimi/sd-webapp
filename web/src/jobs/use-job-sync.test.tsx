import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { TrackedJob } from './job-store'

const toast = vi.hoisted(() => ({ success: vi.fn(), error: vi.fn() }))
vi.mock('sonner', () => ({ toast }))

type RemoteJob = {
  job_id: string
  status: string
  created_at: string
  started_at?: string | null
  error?: string | null
  lane?: string
  cancellable?: boolean
  progress?: number
  detail?: Record<string, unknown> | null
}
let remote: RemoteJob[] = []

const tracked = (id: string, extra: Partial<TrackedJob> = {}): TrackedJob => ({
  id,
  kind: 'generate',
  family: 'sdxl',
  groupId: 'group-1',
  label: 'a cat',
  request: {},
  status: 'queued',
  // Submitted well before the poll, so a missing job counts as expired.
  submittedAt: new Date(Date.now() - 60_000).toISOString(),
  ...extra,
})

async function setup(jobs: TrackedJob[]) {
  vi.resetModules()
  const { jobStore } = await import('./job-store')
  const { useJobSync } = await import('./use-job-sync')
  jobStore.add(jobs)
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const wrapper = ({ children }: { children: ReactNode }) => (
    <MemoryRouter>
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    </MemoryRouter>
  )
  renderHook(() => useJobSync(), { wrapper })
  return jobStore
}

beforeEach(() => {
  remote = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: Request) => {
      if (new URL(input.url).pathname === '/api/v1/jobs/') {
        return new Response(JSON.stringify(remote.map((j) => ({ job_type: 'txt2img', progress: 0, ...j }))), {
          headers: { 'content-type': 'application/json' },
        })
      }
      return new Response('{}', { status: 404 })
    }),
  )
})
afterEach(() => {
  vi.unstubAllGlobals()
  toast.success.mockReset()
  toast.error.mockReset()
})

describe('useJobSync', () => {
  it('updates statuses and queue positions from the backend list', async () => {
    remote = [
      { job_id: 'a', status: 'running', created_at: '2026-01-01T00:00:00', started_at: '2026-01-01T00:00:01' },
      { job_id: 'b', status: 'queued', created_at: '2026-01-01T00:00:02' },
    ]
    const store = await setup([tracked('a'), tracked('b')])
    await waitFor(() => expect(store.getAll().find((j) => j.id === 'b')?.queuePosition).toBe(1))
    expect(store.getAll().find((j) => j.id === 'a')?.status).toBe('running')
  })

  it("counts queue positions within the job's lane and tracks progress", async () => {
    remote = [
      { job_id: 'dl-running', status: 'running', created_at: '2026-01-01T00:00:00', lane: 'io', cancellable: true },
      { job_id: 'dl-queued', status: 'queued', created_at: '2026-01-01T00:00:01', lane: 'io', cancellable: true },
      { job_id: 'gen', status: 'queued', created_at: '2026-01-01T00:00:02', lane: 'gpu' },
      {
        job_id: 'mine',
        status: 'running',
        created_at: '2026-01-01T00:00:03',
        lane: 'io',
        cancellable: true,
        progress: 0.25,
        detail: { phase: 'downloading', bytes_done: 25, bytes_total: 100 },
      },
    ]
    const store = await setup([tracked('gen'), tracked('mine', { groupId: 'group-2' })])
    await waitFor(() => expect(store.getAll().find((j) => j.id === 'gen')?.queuePosition).toBe(1))
    expect(store.getAll().find((j) => j.id === 'mine')).toMatchObject({
      status: 'running',
      lane: 'io',
      cancellable: true,
      progress: 0.25,
      detail: { phase: 'downloading', bytes_done: 25, bytes_total: 100 },
    })
  })

  it('announces a submission once, when all of its jobs are done', async () => {
    remote = [
      { job_id: 'a', status: 'completed', created_at: '2026-01-01T00:00:00' },
      { job_id: 'b', status: 'failed', created_at: '2026-01-01T00:00:01', error: 'OOM' },
    ]
    const store = await setup([tracked('a'), tracked('b')])
    await waitFor(() => expect(store.getAll().every((j) => j.status !== 'queued')).toBe(true))
    expect(toast.error).toHaveBeenCalledWith('Job failed', { description: 'OOM' })
    expect(toast.success).toHaveBeenCalledTimes(1)
    expect(toast.success.mock.calls[0][0]).toBe('Image ready')
  })

  it('announces finished downloads by file name', async () => {
    remote = [{ job_id: 'dl', status: 'completed', created_at: '2026-01-01T00:00:00', lane: 'io', cancellable: true }]
    const store = await setup([tracked('dl', { kind: 'download', family: 'sd15', label: 'DreamShaper_8.safetensors' })])
    await waitFor(() => expect(store.getAll()[0].status).toBe('completed'))
    expect(toast.success).toHaveBeenCalledWith(
      'Downloaded DreamShaper_8.safetensors',
      expect.objectContaining({ description: undefined }),
    )
  })

  it('expires jobs the backend no longer knows, but not freshly submitted ones', async () => {
    remote = []
    const store = await setup([
      tracked('gone'),
      tracked('fresh', { groupId: 'group-2', submittedAt: new Date(Date.now() + 60_000).toISOString() }),
    ])
    await waitFor(() => expect(store.getAll().find((j) => j.id === 'gone')?.status).toBe('expired'))
    expect(store.getAll().find((j) => j.id === 'fresh')?.status).toBe('queued')
  })
})
