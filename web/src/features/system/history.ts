import { useSyncExternalStore } from 'react'

import type { SystemInfo } from '@/api/schema'
import type { GpuStats, HostStats } from '@/api/types'

/** ~3 minutes at the System page's 3s refresh. */
export const HISTORY_LENGTH = 60

export interface Sample {
  /** Epoch ms when the sample was received. */
  t: number
  cpu: number
  memory: number
  /** Undefined when the backend can't report GPU memory. */
  gpu?: number
}

// Module-level ring buffer: history survives leaving and re-opening the page.
let samples: Sample[] = []
let lastRecorded = 0
const listeners = new Set<() => void>()

export function recordSample(info: SystemInfo, receivedAt: number) {
  if (receivedAt <= lastRecorded) return
  lastRecorded = receivedAt
  const host = info.host_stats as unknown as HostStats
  const gpu = info.gpu_stats as unknown as GpuStats
  const sample: Sample = {
    t: receivedAt,
    cpu: host.cpu_percent,
    memory: host.memory_percent,
    gpu: gpu.available && gpu.total_bytes ? ((gpu.used_bytes ?? 0) / gpu.total_bytes) * 100 : undefined,
  }
  samples = [...samples, sample].slice(-HISTORY_LENGTH)
  listeners.forEach((notify) => notify())
}

export function useHistory(): Sample[] {
  return useSyncExternalStore(
    (notify) => {
      listeners.add(notify)
      return () => listeners.delete(notify)
    },
    () => samples,
  )
}
