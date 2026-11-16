import { describe, expect, it } from 'vitest'

import type { ModelCivitaiInfo } from '@/api/types'

import { civitaiLabel, fluxComponents, modelPreviewUrl } from './models'

const civitai = (info: Partial<ModelCivitaiInfo>): ModelCivitaiInfo => ({ trained_words: [], nsfw: false, ...info })

describe('civitaiLabel', () => {
  it('joins model and version names', () => {
    expect(civitaiLabel({ civitai: civitai({ model_name: 'DreamShaper', version_name: '8' }) })).toBe('DreamShaper · 8')
    expect(civitaiLabel({ civitai: civitai({ model_name: 'DreamShaper' }) })).toBe('DreamShaper')
  })

  it('is undefined for models not matched on Civitai', () => {
    expect(civitaiLabel({ civitai: civitai({}) })).toBeUndefined()
    expect(civitaiLabel({})).toBeUndefined()
  })
})

describe('modelPreviewUrl', () => {
  it('encodes the model name', () => {
    expect(modelPreviewUrl('sd15', 'loras', 'my lora#1.safetensors')).toBe(
      '/api/v1/models/sd15/loras/my%20lora%231.safetensors/preview',
    )
  })
})

describe('fluxComponents', () => {
  it('splits bundled and shared components', () => {
    expect(fluxComponents({ transformer: true, clip_l: false, t5xxl: false, vae: true })).toEqual({
      bundled: ['Transformer', 'VAE'],
      shared: ['CLIP-L', 'T5-XXL'],
    })
  })
})
