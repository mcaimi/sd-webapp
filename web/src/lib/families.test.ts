import { describe, expect, it } from 'vitest'

import { DEFAULT_FAMILY, FAMILIES, FAMILY_LIST, isFamilyId, isResourceType } from './families'

describe('family registry', () => {
  it('keys entries by their own id', () => {
    for (const [key, family] of Object.entries(FAMILIES)) expect(family.id).toBe(key)
  })

  it.each(FAMILY_LIST)('$id defaults to one of its resolution presets', (family) => {
    const { width, height } = family.defaults
    expect(family.resolutions).toContainEqual(expect.objectContaining({ width, height }))
  })

  it.each(FAMILY_LIST.filter((f) => f.schedulers))('$id default scheduler is in its scheduler list', (family) => {
    expect(family.schedulers).toContain(family.defaults.scheduler)
  })

  it('validates family ids', () => {
    expect(isFamilyId(DEFAULT_FAMILY)).toBe(true)
    expect(isFamilyId('flux')).toBe(true)
    expect(isFamilyId('sd3')).toBe(false)
    expect(isFamilyId(undefined)).toBe(false)
  })
})

describe('isResourceType', () => {
  it('accepts the backend resource directories only', () => {
    expect(isResourceType('loras')).toBe(true)
    expect(isResourceType('embeddings')).toBe(false)
    expect(isResourceType(null)).toBe(false)
  })
})
