/**
 * Model family registry.
 *
 * Every per-family difference lives here, so pages are written once and
 * parameterised by family instead of being copied per architecture.
 * Adding a family the backend supports means adding one entry.
 */

export type FamilyId = 'sd15' | 'sdxl' | 'illustrious' | 'flux'

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
  /**
   * Fixed scheduler list, for families whose schedulers aren't returned by
   * GET /info/schedulers (that endpoint only lists the SD schedulers).
   * When undefined, the API list is used.
   */
  schedulers?: string[]
  /** Flux's `true_cfg_scale`: >1.0 enables the negative prompt. */
  trueCfgScale?: { default: number; min: number; max: number }
  /** Whether the negative prompt is honoured without extra settings. */
  negativePrompt: boolean
}

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
    negativePrompt: true,
  },
  sdxl: {
    id: 'sdxl',
    label: 'Stable Diffusion XL',
    shortLabel: 'SDXL',
    description: '1024px model with stronger composition and prompt following.',
    defaults: { width: 832, height: 1216, steps: 20, cfgScale: 7, scheduler: 'DPM++ 2M' },
    inpaintDefaults: { steps: 20, cfgScale: 7 },
    resolutions: SDXL_RESOLUTIONS,
    negativePrompt: true,
  },
  illustrious: {
    id: 'illustrious',
    label: 'Illustrious XL',
    shortLabel: 'Illustrious',
    description: 'SDXL-based model tuned for illustration and anime styles.',
    defaults: { width: 832, height: 1216, steps: 28, cfgScale: 5.5, scheduler: 'Euler a' },
    inpaintDefaults: { steps: 24, cfgScale: 5.5 },
    resolutions: SDXL_RESOLUTIONS,
    negativePrompt: true,
  },
  flux: {
    id: 'flux',
    label: 'Flux',
    shortLabel: 'Flux',
    description: 'Flow-matching transformer with excellent text rendering.',
    defaults: { width: 1024, height: 1024, steps: 28, cfgScale: 3.5, scheduler: 'FlowMatch Euler' },
    inpaintDefaults: { steps: 28, cfgScale: 3.5 },
    resolutions: SDXL_RESOLUTIONS,
    schedulers: ['FlowMatch Euler', 'FlowMatch Heun'],
    trueCfgScale: { default: 1, min: 1, max: 10 },
    negativePrompt: false,
  },
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
