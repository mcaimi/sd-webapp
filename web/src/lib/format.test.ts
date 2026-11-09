import { describe, expect, it } from 'vitest'

import { formatBytes, formatPercent } from './format'

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
