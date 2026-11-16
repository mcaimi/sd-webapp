const BYTE_UNITS = ['B', 'KB', 'MB', 'GB', 'TB']

/** 34359738368 -> "32 GB" (binary multiples, labelled the way OS tools do). */
export function formatBytes(bytes: number, fractionDigits = 1): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B'
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), BYTE_UNITS.length - 1)
  const value = bytes / 1024 ** exponent
  const digits = exponent === 0 || value >= 100 ? 0 : fractionDigits
  return `${value.toFixed(digits).replace(/\.0+$/, '')} ${BYTE_UNITS[exponent]}`
}

/** 1234 -> "1.2K", 2_500_000 -> "2.5M" */
export function formatCount(value: number): string {
  return new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(value)
}

export function formatPercent(value: number): string {
  return `${Math.round(value)}%`
}

/** 75_000 -> "1:15" */
export function formatElapsed(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000))
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`
}

/** "2026-11-09T14:20:00" -> "5 min ago" */
export function formatRelative(iso: string, now = Date.now()): string {
  const seconds = Math.round((now - Date.parse(iso)) / 1000)
  if (seconds < 45) return 'just now'
  const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' })
  if (seconds < 3600) return rtf.format(-Math.round(seconds / 60), 'minute')
  if (seconds < 86400) return rtf.format(-Math.round(seconds / 3600), 'hour')
  return rtf.format(-Math.round(seconds / 86400), 'day')
}
