import { describe, expect, it } from 'vitest'

import { formatBytes, formatCount, formatPercent } from './format'

describe('formatBytes', () => {
  it.each([
    [0, '0 B'],
    [512, '512 B'],
    [393216, '384 KB'],
    [1536, '1.5 KB'],
    [16506732544, '15.4 GB'],
    [34359738368, '32 GB'],
  ])('%d -> %s', (bytes, expected) => {
    expect(formatBytes(bytes)).toBe(expected)
  })
})

describe('formatPercent', () => {
  it('rounds to whole percent', () => {
    expect(formatPercent(55.2)).toBe('55%')
  })
})

describe('formatCount', () => {
  it.each([
    [0, '0'],
    [999, '999'],
    [1234, '1.2K'],
    [2_500_000, '2.5M'],
  ])('%d -> %s', (value, expected) => {
    expect(formatCount(value)).toBe(expected)
  })
})
