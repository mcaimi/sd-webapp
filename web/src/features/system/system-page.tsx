import { Boxes, Cpu, ListOrdered, Workflow } from 'lucide-react'
import { motion } from 'motion/react'
import { useEffect } from 'react'

import { useSystemInfo } from '@/api/queries'
import type { CacheStats, GpuStats, HostStats, QueueStats } from '@/api/types'
import { Meter } from '@/components/meter'
import { PageHeader } from '@/components/page-header'
import { QueryError } from '@/components/query-error'
import { StatTile } from '@/components/stat-tile'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { formatBytes, formatPercent } from '@/lib/format'

import { HISTORY_LENGTH, recordSample, useHistory } from './history'
import { LineChart } from './line-chart'

const stagger = {
  hidden: {},
  show: { transition: { staggerChildren: 0.05 } },
}
const rise = {
  hidden: { opacity: 0, y: 8 },
  show: { opacity: 1, y: 0 },
}

export function SystemPage() {
  const { data, error, isPending, refetch, dataUpdatedAt } = useSystemInfo({ refetchInterval: 3_000 })
  const history = useHistory()

  useEffect(() => {
    if (data) recordSample(data, dataUpdatedAt)
  }, [data, dataUpdatedAt])

  if (error) {
    return (
      <>
        <PageHeader title="System" description="Backend health and resource usage." />
        <QueryError error={error} onRetry={() => refetch()} title="Backend unreachable" />
      </>
    )
  }

  if (isPending) {
    return (
      <>
        <PageHeader title="System" description="Backend health and resource usage." />
        <div className="grid grid-cols-2 gap-3 md:gap-4 xl:grid-cols-4">
          {Array.from({ length: 4 }, (_, i) => (
            <Skeleton key={i} className="h-28 rounded-xl" />
          ))}
        </div>
        <Skeleton className="mt-4 h-64 rounded-xl" />
      </>
    )
  }

  const gpu = data.gpu_stats as unknown as GpuStats
  const host = data.host_stats as unknown as HostStats
  const queue = data.queue_stats as unknown as QueueStats
  const cache = data.cache_stats as unknown as CacheStats
  const gpuPercent = gpu.total_bytes ? ((gpu.used_bytes ?? 0) / gpu.total_bytes) * 100 : 0

  return (
    <>
      <PageHeader title="System" description="Backend health and resource usage, refreshed every few seconds." />
      <motion.div variants={stagger} initial="hidden" animate="show" className="space-y-4">
        <div className="grid grid-cols-2 gap-3 md:gap-4 xl:grid-cols-4">
          <motion.div variants={rise}>
            <StatTile
              label="Device"
              value={data.device.toUpperCase()}
              detail={[data.gpu_name ?? data.vendor, data.dtype.replace('torch.', '')].filter(Boolean).join(' · ')}
              icon={Cpu}
            />
          </motion.div>
          <motion.div variants={rise}>
            <StatTile
              label="Queue"
              value={`${queue.queue_size} queued`}
              detail={`${queue.total_jobs} jobs since start`}
              icon={ListOrdered}
            />
          </motion.div>
          <motion.div variants={rise}>
            <StatTile
              label="Workers"
              value={String(queue.num_workers)}
              detail={queue.workers_running ? 'Running' : 'Stopped'}
              icon={Workflow}
            />
          </motion.div>
          <motion.div variants={rise}>
            <StatTile
              label="Model cache"
              value={`${cache.cached_models} / ${cache.max_models}`}
              detail={cache.cache_keys.length ? cache.cache_keys.join(', ') : 'No models loaded'}
              icon={Boxes}
            />
          </motion.div>
        </div>

        <motion.div variants={rise}>
          <Card>
            <CardHeader>
              <CardTitle>Resources</CardTitle>
              <CardDescription>Memory and compute on the generation host.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-6 md:grid-cols-2">
              {gpu.available && gpu.total_bytes ? (
                <Meter
                  label="GPU memory"
                  percent={gpuPercent}
                  valueLabel={`${formatBytes(gpu.used_bytes ?? 0)} of ${formatBytes(gpu.total_bytes)}`}
                  hint={gpu.total_is_approximate ? 'Total is approximate (shared memory)' : undefined}
                />
              ) : (
                <UnavailableStat label="GPU memory" reason="Not reported by the backend" />
              )}
              {gpu.compute?.available && gpu.compute.utilization_percent !== undefined ? (
                <Meter
                  label="GPU compute"
                  percent={gpu.compute.utilization_percent}
                  valueLabel={formatPercent(gpu.compute.utilization_percent)}
                />
              ) : (
                <UnavailableStat label="GPU compute" reason={gpu.compute?.reason ?? 'Not reported by the backend'} />
              )}
              <Meter
                label="Host memory"
                percent={host.memory_percent}
                valueLabel={`${formatBytes(host.memory_used_bytes)} of ${formatBytes(host.memory_total_bytes)}`}
              />
              <Meter
                label="CPU"
                percent={host.cpu_percent}
                valueLabel={formatPercent(host.cpu_percent)}
                hint={[
                  `${host.cpu_count} cores`,
                  host.load_average && `load ${host.load_average.map((l) => l.toFixed(2)).join(' / ')}`,
                ]
                  .filter(Boolean)
                  .join(' · ')}
              />
            </CardContent>
          </Card>
        </motion.div>

        <motion.div variants={rise}>
          <Card>
            <CardHeader>
              <CardTitle>History</CardTitle>
              <CardDescription>
                Last {Math.round((HISTORY_LENGTH * 3) / 60)} minutes, sampled every 3 seconds.
              </CardDescription>
            </CardHeader>
            <CardContent className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
              <LineChart
                title="CPU"
                points={history.map((h) => ({ t: h.t, value: h.cpu }))}
                capacity={HISTORY_LENGTH}
                span="3 min"
              />
              <LineChart
                title="Host memory"
                points={history.map((h) => ({ t: h.t, value: h.memory }))}
                capacity={HISTORY_LENGTH}
                span="3 min"
              />
              {history.some((h) => h.gpu !== undefined) && (
                <LineChart
                  title="GPU memory"
                  points={history.filter((h) => h.gpu !== undefined).map((h) => ({ t: h.t, value: h.gpu! }))}
                  capacity={HISTORY_LENGTH}
                  span="3 min"
                />
              )}
            </CardContent>
          </Card>
        </motion.div>
      </motion.div>
    </>
  )
}

function UnavailableStat({ label, reason }: { label: string; reason: string }) {
  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between gap-2 text-sm">
        <span className="font-medium">{label}</span>
        <span className="text-muted-foreground">n/a</span>
      </div>
      <div className="h-2 rounded-full bg-muted" aria-hidden />
      <p className="text-xs text-muted-foreground">{reason}</p>
    </div>
  )
}
