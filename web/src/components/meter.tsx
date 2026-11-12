import { CircleAlert, TriangleAlert } from 'lucide-react'
import { motion } from 'motion/react'

import { cn } from '@/lib/utils'

type Severity = 'ok' | 'warn' | 'danger'

const FILL: Record<Severity, string> = {
  ok: 'bg-meter-ok',
  warn: 'bg-meter-warn',
  danger: 'bg-meter-danger',
}
// The track is a lighter step of the fill's own hue, so state reads across the whole bar.
const TRACK: Record<Severity, string> = {
  ok: 'bg-meter-ok-track',
  warn: 'bg-meter-warn-track',
  danger: 'bg-meter-danger-track',
}

interface MeterProps {
  label: string
  /** 0-100 */
  percent: number
  /** Text shown on the right, e.g. "12.3 GB of 32 GB". */
  valueLabel: string
  hint?: string
  warnAt?: number
  dangerAt?: number
}

export function Meter({ label, percent, valueLabel, hint, warnAt = 75, dangerAt = 90 }: MeterProps) {
  const clamped = Math.min(Math.max(percent, 0), 100)
  const severity: Severity = clamped >= dangerAt ? 'danger' : clamped >= warnAt ? 'warn' : 'ok'

  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between gap-2 text-sm">
        <span className="font-medium">{label}</span>
        <span className="text-muted-foreground">{valueLabel}</span>
      </div>
      <div
        role="meter"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(clamped)}
        aria-valuetext={valueLabel}
        className={cn('h-2 overflow-hidden rounded-full transition-colors', TRACK[severity])}
      >
        <motion.div
          className={cn('h-full rounded-full transition-colors', FILL[severity])}
          initial={false}
          animate={{ width: `${clamped}%` }}
          transition={{ type: 'spring', stiffness: 120, damping: 20 }}
        />
      </div>
      {(severity !== 'ok' || hint) && (
        <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
          {severity === 'danger' && <CircleAlert className="size-3.5 text-meter-danger" aria-hidden />}
          {severity === 'warn' && <TriangleAlert className="size-3.5 text-meter-warn" aria-hidden />}
          {severity === 'danger' ? 'Critical' : severity === 'warn' ? 'High' : null}
          {severity !== 'ok' && hint && ' · '}
          {hint}
        </p>
      )}
    </div>
  )
}
