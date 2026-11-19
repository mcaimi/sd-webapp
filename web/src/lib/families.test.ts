import { describe, expect, it } from 'vitest'

import {
  DEFAULT_FAMILY,
  FAMILIES,
  FAMILY_LIST,
  isFamilyId,
  isResourceType,
  negativePromptActive,
  variantForBaseModel,
} from './families'

describe('family registry', () => {
  it('keys entries by their own id', () => {
    for (const [key, family] of Object.entries(FAMILIES)) expect(family.id).toBe(key)
  })

  it.each(FAMILY_LIST)('$id defaults to one of its resolution presets', (family) => {
    const { width, height } = family.defaults
    expect(family.resolutions).toContainEqual(expect.objectContaining({ width, height }))
  })

  it.each(FAMILY_LIST.filter((f) => f.variants))('$id defaults match its first variant', (family) => {
    const [first] = family.variants!
    expect(family.defaults).toMatchObject({ steps: first.steps, cfgScale: first.cfgScale })
  })

  it('validates family ids', () => {
    expect(isFamilyId(DEFAULT_FAMILY)).toBe(true)
    expect(isFamilyId('flux')).toBe(true)
    expect(isFamilyId('zimage')).toBe(true)
    expect(isFamilyId('sd3')).toBe(false)
    expect(isFamilyId(undefined)).toBe(false)
  })
})

describe('negativePromptActive', () => {
  it('follows each family rule', () => {
    expect(negativePromptActive('sdxl', { cfgScale: 0, trueCfgScale: 1 })).toBe(true)
    expect(negativePromptActive('flux', { cfgScale: 3.5, trueCfgScale: 1 })).toBe(false)
    expect(negativePromptActive('flux', { cfgScale: 3.5, trueCfgScale: 2 })).toBe(true)
    expect(negativePromptActive('zimage', { cfgScale: 0, trueCfgScale: 1 })).toBe(false)
    expect(negativePromptActive('zimage', { cfgScale: 4, trueCfgScale: 1 })).toBe(true)
  })
})

describe('variantForBaseModel', () => {
  it('maps Civitai base models to variants', () => {
    expect(variantForBaseModel('zimage', 'ZImageBase')).toBe('base')
    expect(variantForBaseModel('zimage', 'SDXL 1.0')).toBeUndefined()
    expect(variantForBaseModel('sdxl', 'ZImageBase')).toBeUndefined()
    expect(variantForBaseModel('zimage', null)).toBeUndefined()
  })
})

describe('isResourceType', () => {
  it('accepts the backend resource directories only', () => {
    expect(isResourceType('loras')).toBe(true)
    expect(isResourceType('embeddings')).toBe(false)
    expect(isResourceType(null)).toBe(false)
  })
})
