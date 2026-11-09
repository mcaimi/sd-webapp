import type { GenerationInstance } from '@/api/types'

import type { FamilyId } from './families'
import { FAMILIES } from './families'

/** Form-level generation parameters, shared by the generate and inpaint tools. */
export interface GenerationParams {
  prompt: string
  negativePrompt: string
  model: string
  loras: { name: string; strength: number }[]
  /** Custom VAE; empty string means the checkpoint's own. */
  vae: string
  width: number
  height: number
  steps: number
  cfgScale: number
  /** Flux only. */
  trueCfgScale: number
  scheduler: string
  /** Seed used when `seedLocked`; otherwise each image gets a random seed. */
  seed: number
  seedLocked: boolean
  /** Number of images per submission. */
  count: number
}

export const MAX_LORAS = 5
export const MAX_BATCH = 8
export const SEED_MAX = 2 ** 32 - 1

export function defaultParams(family: FamilyId): GenerationParams {
  const { defaults, trueCfgScale } = FAMILIES[family]
  return {
    prompt: '',
    negativePrompt: '',
    model: '',
    loras: [],
    vae: '',
    width: defaults.width,
    height: defaults.height,
    steps: defaults.steps,
    cfgScale: defaults.cfgScale,
    trueCfgScale: trueCfgScale?.default ?? 1,
    scheduler: defaults.scheduler,
    seed: randomSeed(),
    seedLocked: false,
    count: 1,
  }
}

export function randomSeed(): number {
  return crypto.getRandomValues(new Uint32Array(1))[0]
}

/**
 * Seeds for a submission of `count` images. A locked seed gives consecutive
 * seeds (seed, seed+1, ...) so a batch stays reproducible without being identical.
 */
export function seedsFor(params: Pick<GenerationParams, 'seed' | 'seedLocked' | 'count'>): number[] {
  return Array.from({ length: params.count }, (_, i) =>
    params.seedLocked ? (params.seed + i) % (SEED_MAX + 1) : randomSeed(),
  )
}

/** Fields common to every family's generate/inpaint request, in API (snake_case) form. */
export function toApiParams(family: FamilyId, params: GenerationParams, seed: number) {
  return {
    positive_prompt: params.prompt.trim(),
    negative_prompt: params.negativePrompt.trim(),
    model_checkpoint: params.model,
    steps: params.steps,
    cfg_scale: params.cfgScale,
    seed,
    scheduler: params.scheduler,
    loras: params.loras.length ? params.loras : null,
    custom_vae: params.vae || null,
    ...(family === 'flux' ? { true_cfg_scale: params.trueCfgScale } : {}),
  }
}

/**
 * Rough CLIP token count. CLIP's BPE splits words and punctuation; most common
 * English words are one token, long or rare words several. Good enough to warn
 * about the 77-token window without shipping a tokenizer.
 */
export function estimateTokens(text: string): number {
  const pieces: string[] = text.toLowerCase().match(/[a-z]+|\d|[^\sa-z\d]/g) ?? []
  return pieces.reduce((sum, piece) => sum + (piece.length > 7 ? Math.ceil(piece.length / 5) : 1), 0)
}

export const CLIP_TOKEN_LIMIT = 75

/** Request payload fields as submitted to the API (see `toApiParams`). */
interface ApiGenerationFields {
  positive_prompt?: string
  negative_prompt?: string
  model_checkpoint?: string
  width?: number
  height?: number
  steps?: number
  cfg_scale?: number
  true_cfg_scale?: number
  seed?: number
  scheduler?: string
  loras?: { name: string; strength: number }[] | null
  custom_vae?: string | null
}

/** Rebuild form params from a submitted request, to re-run or tweak a past job. */
export function paramsFromRequest(request: Record<string, unknown>): Partial<GenerationParams> {
  const r = request as ApiGenerationFields
  const params: Partial<GenerationParams> = {
    prompt: r.positive_prompt,
    negativePrompt: r.negative_prompt,
    model: r.model_checkpoint,
    width: r.width,
    height: r.height,
    steps: r.steps,
    cfgScale: r.cfg_scale,
    trueCfgScale: r.true_cfg_scale,
    scheduler: r.scheduler,
    loras: r.loras ?? [],
    vae: r.custom_vae ?? '',
  }
  if (r.seed !== undefined && r.seed >= 0) Object.assign(params, { seed: r.seed, seedLocked: true })
  return Object.fromEntries(Object.entries(params).filter(([, v]) => v !== undefined))
}

/**
 * Rebuild form params from persisted output metadata (gallery). The backend
 * doesn't record which checkpoint produced an output, so the model is left alone.
 */
export function paramsFromOutputMetadata(metadata: OutputMetadata): Partial<GenerationParams> {
  const instance = metadata.generation?.instances?.[0]
  if (!instance) return {}
  return paramsFromRequest({
    positive_prompt: instance.prompt,
    negative_prompt: instance.negative_prompt,
    width: instance.width,
    height: instance.height,
    steps: instance.num_inference_steps,
    cfg_scale: instance.guidance_scale,
    seed: instance.seed,
    scheduler: instance.scheduler ?? undefined,
    loras: Object.values(metadata.loras ?? {}).map((l) => ({ name: l.name, strength: l.merge_strength })),
    custom_vae: metadata.vae,
  })
}

/** Metadata JSON persisted next to each output image (api/services/generation.py). */
export interface OutputMetadata {
  generation?: { instances?: GenerationInstance[] }
  loras?: Record<string, { name: string; lora_path?: string; merge_strength: number }>
  vae?: string | null
}
