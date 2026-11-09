import { modelDisplayName } from '@/lib/models'

export type MergeMethod = 'linear' | 'slerp' | 'additive' | 'subtract'

export const MERGE_METHODS: { id: MergeMethod; label: string; description: string }[] = [
  { id: 'linear', label: 'Linear', description: 'Weighted average of A and B. Predictable, the usual choice.' },
  { id: 'slerp', label: 'SLERP', description: 'Spherical blend that better keeps the character of both models.' },
  { id: 'additive', label: 'Additive', description: "Adds B's features on top of A." },
  { id: 'subtract', label: 'Subtract', description: "Removes B's features from A." },
]

/** Linear/SLERP interpolate between the two models; additive/subtract apply B at a strength. */
export function isBlend(method: MergeMethod): boolean {
  return method === 'linear' || method === 'slerp'
}

export function ensureSafetensors(name: string): string {
  const trimmed = name.trim()
  if (!trimmed) return ''
  return trimmed.endsWith('.safetensors') ? trimmed : `${trimmed}.safetensors`
}

/** "models/checkpoints/sdxl/batch/merged.safetensors" -> "merged.safetensors" */
export function basename(path: string): string {
  return path.split(/[\\/]/).pop() ?? path
}

export function suggestOutputName(base: string, target: string, method: MergeMethod, alpha: number): string {
  if (!base) return ''
  const parts = [modelDisplayName(base), target && modelDisplayName(target), method, Math.round(alpha * 100)]
  return `${parts.filter((p) => p !== '' && p !== undefined).join('_')}.safetensors`
}

/** Same-origin URL for downloading a checkpoint; the backend resolves it by basename. */
export function checkpointDownloadUrl(family: string, path: string): string {
  return `/api/v1/models/${family}/checkpoints/${encodeURIComponent(basename(path))}/download`
}
