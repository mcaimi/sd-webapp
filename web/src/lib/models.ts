import type { FluxLayout, ModelEntry } from '@/api/types'

import type { FamilyId, ResourceType } from './families'

/** "juggernautXL_v9.safetensors" -> "juggernautXL_v9" */
export function modelDisplayName(name: string): string {
  return name.replace(/\.safetensors$/, '')
}

/** "DreamShaper · 8", for models the backend matched on Civitai. */
export function civitaiLabel(model: Pick<ModelEntry, 'civitai'>): string | undefined {
  const info = model.civitai
  if (!info?.model_name) return undefined
  return [info.model_name, info.version_name].filter(Boolean).join(' · ')
}

/** Preview image saved next to a model, available when `has_preview` is set. */
export function modelPreviewUrl(family: FamilyId, resource: ResourceType, name: string): string {
  return `/api/v1/models/${family}/${resource}/${encodeURIComponent(name)}/preview`
}

const FLUX_COMPONENTS = [
  ['transformer', 'Transformer'],
  ['clip_l', 'CLIP-L'],
  ['t5xxl', 'T5-XXL'],
  ['vae', 'VAE'],
] as const

/** Components a Flux checkpoint bundles, and those it takes from the shared components directory. */
export function fluxComponents(layout: Partial<FluxLayout>): { bundled: string[]; shared: string[] } {
  const bundled = FLUX_COMPONENTS.filter(([key]) => layout[key]).map(([, label]) => label)
  const shared = FLUX_COMPONENTS.filter(([key]) => !layout[key]).map(([, label]) => label)
  return { bundled, shared }
}
