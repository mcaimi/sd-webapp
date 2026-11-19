/**
 * Model family registry.
 *
 * Every per-family difference lives here, so pages are written once and
 * parameterised by family instead of being copied per architecture.
 * Adding a family the backend supports means adding one entry.
 */

export type FamilyId = 'sd15' | 'sdxl' | 'illustrious' | 'flux' | 'zimage'

/** Z-Image checkpoint variant, sent as `variant`. */
export type VariantId = 'turbo' | 'base'

export interface Variant {
  id: VariantId
  label: string
  /** Sampling defaults applied when the variant is selected. */
  steps: number
  cfgScale: number
  /** Civitai base models of this variant, to preselect it from a checkpoint's sidecar. */
  civitaiBaseModels: string[]
}

export interface Resolution {
  label: string
  width: number
  height: number
}

export interface Family {
  id: FamilyId
  label: string
  shortLabel: string
  description: string
  /** txt2img defaults; mirror sensible values for the architecture, not the API's generic ones. */
  defaults: {
    width: number
    height: number
    steps: number
    cfgScale: number
    scheduler: string
  }
  inpaintDefaults: {
    steps: number
    cfgScale: number
  }
  /** Aspect-ratio presets near the architecture's native pixel count. */
  resolutions: Resolution[]
  /** Slider ranges and wording for the sampling controls. */
  sampling: {
    maxSteps: number
    cfg: { label: string; min: number; max: number; hint?: string }
  }
  /** Flux's `true_cfg_scale`: >1.0 enables the negative prompt. */
  trueCfgScale?: { default: number; min: number; max: number }
  /**
   * When the negative prompt is honoured: always, only with True CFG above 1
   * (Flux), or only with CFG above 0 (Z-Image).
   */
  negativePrompt: 'always' | 'true-cfg' | 'cfg'
  /** Inpaint re-noising `strength`, for families whose inpaint request accepts it. */
  inpaintStrength?: { default: number }
  /** Checkpoint variants; the first is the default and matches `defaults`. */
  variants?: Variant[]
}

const SD_SAMPLING: Family['sampling'] = { maxSteps: 100, cfg: { label: 'CFG scale', min: 0, max: 20 } }

const SDXL_RESOLUTIONS: Resolution[] = [
  { label: '1:1', width: 1024, height: 1024 },
  { label: '2:3', width: 832, height: 1216 },
  { label: '3:2', width: 1216, height: 832 },
  { label: '3:4', width: 896, height: 1152 },
  { label: '4:3', width: 1152, height: 896 },
  { label: '9:16', width: 768, height: 1344 },
  { label: '16:9', width: 1344, height: 768 },
]

export const FAMILIES: Record<FamilyId, Family> = {
  sd15: {
    id: 'sd15',
    label: 'Stable Diffusion 1.5',
    shortLabel: 'SD 1.5',
    description: 'Fast 512px base model with a huge LoRA ecosystem.',
    defaults: { width: 512, height: 768, steps: 30, cfgScale: 7, scheduler: 'DPM++ 2M' },
    inpaintDefaults: { steps: 20, cfgScale: 7 },
    resolutions: [
      { label: '1:1', width: 512, height: 512 },
      { label: '2:3', width: 512, height: 768 },
      { label: '3:2', width: 768, height: 512 },
      { label: '3:4', width: 576, height: 768 },
      { label: '4:3', width: 768, height: 576 },
    ],
    sampling: SD_SAMPLING,
    negativePrompt: 'always',
  },
  sdxl: {
    id: 'sdxl',
    label: 'Stable Diffusion XL',
    shortLabel: 'SDXL',
    description: '1024px model with stronger composition and prompt following.',
    defaults: { width: 832, height: 1216, steps: 20, cfgScale: 7, scheduler: 'DPM++ 2M' },
    inpaintDefaults: { steps: 20, cfgScale: 7 },
    resolutions: SDXL_RESOLUTIONS,
    sampling: SD_SAMPLING,
    negativePrompt: 'always',
  },
  illustrious: {
    id: 'illustrious',
    label: 'Illustrious XL',
    shortLabel: 'Illustrious',
    description: 'SDXL-based model tuned for illustration and anime styles.',
    defaults: { width: 832, height: 1216, steps: 28, cfgScale: 5.5, scheduler: 'Euler a' },
    inpaintDefaults: { steps: 24, cfgScale: 5.5 },
    resolutions: SDXL_RESOLUTIONS,
    sampling: SD_SAMPLING,
    negativePrompt: 'always',
  },
  flux: {
    id: 'flux',
    label: 'Flux',
    shortLabel: 'Flux',
    description: 'Flow-matching transformer with excellent text rendering.',
    defaults: { width: 1024, height: 1024, steps: 28, cfgScale: 3.5, scheduler: 'FlowMatch Euler' },
    inpaintDefaults: { steps: 28, cfgScale: 3.5 },
    resolutions: SDXL_RESOLUTIONS,
    sampling: {
      maxSteps: 50,
      cfg: { label: 'Guidance', min: 1, max: 10, hint: 'Distilled guidance; 3–4 is typical.' },
    },
    trueCfgScale: { default: 1, min: 1, max: 10 },
    negativePrompt: 'true-cfg',
    inpaintStrength: { default: 0.85 },
  },
  zimage: {
    id: 'zimage',
    label: 'Z-Image',
    shortLabel: 'Z-Image',
    description: 'Single-stream transformer; Turbo renders in a handful of steps.',
    defaults: { width: 1024, height: 1024, steps: 8, cfgScale: 0, scheduler: 'FlowMatch Euler' },
    inpaintDefaults: { steps: 8, cfgScale: 0 },
    resolutions: SDXL_RESOLUTIONS,
    sampling: {
      maxSteps: 50,
      cfg: { label: 'CFG scale', min: 0, max: 10, hint: 'Above 0 enables the negative prompt; Turbo works best at 0.' },
    },
    negativePrompt: 'cfg',
    inpaintStrength: { default: 1 },
    variants: [
      { id: 'turbo', label: 'Turbo', steps: 8, cfgScale: 0, civitaiBaseModels: ['ZImageTurbo'] },
      { id: 'base', label: 'Base', steps: 28, cfgScale: 4, civitaiBaseModels: ['ZImageBase'] },
    ],
  },
}

/** Whether the family honours the negative prompt with these settings. */
export function negativePromptActive(family: FamilyId, params: { cfgScale: number; trueCfgScale: number }): boolean {
  const rule = FAMILIES[family].negativePrompt
  return rule === 'always' || (rule === 'cfg' ? params.cfgScale > 0 : params.trueCfgScale > 1)
}

/** Why the negative prompt is hidden, for families that don't always honour it. */
export function negativePromptNote(family: FamilyId): string {
  const { shortLabel, negativePrompt } = FAMILIES[family]
  return negativePrompt === 'cfg'
    ? `${shortLabel} ignores negative prompts unless CFG is above 0.`
    : `${shortLabel} ignores negative prompts unless True CFG is above 1.`
}

/** The variant a checkpoint's Civitai base model identifies, if any. */
export function variantForBaseModel(family: FamilyId, baseModel: string | null | undefined): VariantId | undefined {
  if (!baseModel) return undefined
  return FAMILIES[family].variants?.find((v) => v.civitaiBaseModels.includes(baseModel))?.id
}

export const FAMILY_LIST: Family[] = Object.values(FAMILIES)

export const DEFAULT_FAMILY: FamilyId = 'sdxl'

export function isFamilyId(value: string | undefined): value is FamilyId {
  return value !== undefined && value in FAMILIES
}

/** Resource directories exposed by GET /models/{family}/{resource}. */
export type ResourceType = 'checkpoints' | 'loras' | 'vaes'

export const RESOURCE_LABELS: Record<ResourceType, string> = {
  checkpoints: 'Checkpoints',
  loras: 'LoRAs',
  vaes: 'VAEs',
}

export function isResourceType(value: string | null | undefined): value is ResourceType {
  return !!value && value in RESOURCE_LABELS
}
