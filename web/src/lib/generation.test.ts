import { describe, expect, it } from 'vitest'

import {
  defaultParams,
  estimateTokens,
  paramsFromOutputMetadata,
  paramsFromRequest,
  SEED_MAX,
  seedsFor,
  toApiParams,
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
