import { describe, expect, it } from 'vitest'

import {
  appendToPrompt,
  defaultParams,
  estimateTokens,
  modelPatch,
  paramsFromOutputMetadata,
  paramsFromRequest,
  SEED_MAX,
  seedsFor,
  toApiParams,
  variantPatch,
} from './generation'

describe('seedsFor', () => {
  it('uses consecutive seeds when locked, wrapping at the 32-bit limit', () => {
    expect(seedsFor({ seed: 41, seedLocked: true, count: 3 })).toEqual([41, 42, 43])
    expect(seedsFor({ seed: SEED_MAX, seedLocked: true, count: 2 })).toEqual([SEED_MAX, 0])
  })

  it('rolls independent random seeds when unlocked', () => {
    const seeds = seedsFor({ seed: 1, seedLocked: false, count: 4 })
    expect(seeds).toHaveLength(4)
    expect(new Set(seeds).size).toBe(4)
    seeds.forEach((s) => expect(s).toBeLessThanOrEqual(SEED_MAX))
  })
})

describe('toApiParams', () => {
  const params = { ...defaultParams('sdxl'), prompt: '  a cat ', model: 'm.safetensors' }

  it('maps form params to the API shape', () => {
    expect(toApiParams('sdxl', params, 7)).toEqual({
      positive_prompt: 'a cat',
      negative_prompt: '',
      model_checkpoint: 'm.safetensors',
      steps: params.steps,
      cfg_scale: params.cfgScale,
      seed: 7,
      scheduler: params.scheduler,
      loras: null,
      custom_vae: null,
    })
  })

  it('only sends true_cfg_scale for Flux', () => {
    expect(toApiParams('flux', { ...params, trueCfgScale: 2 }, 1)).toMatchObject({ true_cfg_scale: 2 })
    expect(toApiParams('sdxl', params, 1)).not.toHaveProperty('true_cfg_scale')
  })

  it('only sends variant for Z-Image', () => {
    expect(toApiParams('zimage', { ...defaultParams('zimage'), variant: 'base' }, 1)).toMatchObject({ variant: 'base' })
    expect(toApiParams('flux', params, 1)).not.toHaveProperty('variant')
  })
})

describe('variants', () => {
  it('applies the variant sampling defaults', () => {
    expect(variantPatch('zimage', 'base')).toEqual({ variant: 'base', steps: 28, cfgScale: 4 })
    expect(variantPatch('sdxl', 'base')).toEqual({})
  })

  it('picks the variant from the checkpoint Civitai base model', () => {
    const current = defaultParams('zimage')
    expect(modelPatch('zimage', { name: 'b', civitai: { base_model: 'ZImageBase' } }, current)).toEqual({
      model: 'b',
      variant: 'base',
      steps: 28,
      cfgScale: 4,
    })
    // Same variant or unknown base model: sampling settings are left alone.
    expect(modelPatch('zimage', { name: 't', civitai: { base_model: 'ZImageTurbo' } }, current)).toEqual({ model: 't' })
    expect(modelPatch('zimage', { name: 'x' }, current)).toEqual({ model: 'x' })
  })
})

describe('paramsFromRequest', () => {
  it('round-trips a submitted request back to form params with the seed locked', () => {
    const form = {
      ...defaultParams('sd15'),
      prompt: 'p',
      model: 'x.safetensors',
      loras: [{ name: 'l', strength: 0.5 }],
      vae: 'v',
    }
    const request = { ...toApiParams('sd15', form, 99), width: 512, height: 768 }
    expect(paramsFromRequest(request)).toMatchObject({
      prompt: 'p',
      model: 'x.safetensors',
      loras: [{ name: 'l', strength: 0.5 }],
      vae: 'v',
      width: 512,
      height: 768,
      seed: 99,
      seedLocked: true,
    })
  })

  it('leaves the seed alone for random-seed requests', () => {
    expect(paramsFromRequest({ seed: -1 })).not.toHaveProperty('seed')
  })
})

describe('paramsFromOutputMetadata', () => {
  it('reads the backend persisted layout and never sets a model', () => {
    const params = paramsFromOutputMetadata({
      generation: {
        instances: [
          {
            prompt: 'lighthouse',
            negative_prompt: 'blurry',
            num_inference_steps: 28,
            width: 832,
            height: 1216,
            guidance_scale: 5,
            seed: 123,
            scheduler: 'Euler a',
          },
        ],
      },
      loras: { lora_0: { name: 'detail.safetensors', merge_strength: 0.8 } },
      vae: null,
    })
    expect(params).toMatchObject({
      prompt: 'lighthouse',
      negativePrompt: 'blurry',
      steps: 28,
      cfgScale: 5,
      seed: 123,
      seedLocked: true,
      scheduler: 'Euler a',
      loras: [{ name: 'detail.safetensors', strength: 0.8 }],
      vae: '',
    })
    expect(params).not.toHaveProperty('model')
  })

  it('returns nothing for metadata without a generation instance', () => {
    expect(paramsFromOutputMetadata({})).toEqual({})
  })
})

describe('estimateTokens', () => {
  it('counts words and punctuation, with extra tokens for long words', () => {
    expect(estimateTokens('')).toBe(0)
    expect(estimateTokens('a red fox, sitting')).toBe(5)
    expect(estimateTokens('photorealistic')).toBeGreaterThan(1)
  })
})

describe('appendToPrompt', () => {
  it('appends comma-separated words, skipping ones already present', () => {
    expect(appendToPrompt('a cat, Watercolor', ['watercolor', 'wcstyle'])).toBe('a cat, Watercolor, wcstyle')
    expect(appendToPrompt('a cat,', ['wcstyle'])).toBe('a cat, wcstyle')
    expect(appendToPrompt('', ['wcstyle', ' '])).toBe('wcstyle')
  })

  it('leaves the prompt untouched when nothing is new', () => {
    expect(appendToPrompt('a cat, wcstyle ', ['WCSTYLE'])).toBe('a cat, wcstyle ')
  })
})
