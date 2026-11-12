import { describe, expect, it } from 'vitest'

import { maxSideFor, workingSize } from './image-size'

describe('workingSize', () => {
  it('scales the longest side down to the limit, keeping aspect ratio', () => {
    expect(workingSize(4000, 3000, 1024)).toEqual({ width: 1024, height: 768 })
    expect(workingSize(3000, 4000, 768)).toEqual({ width: 576, height: 768 })
  })

  it('never upscales small images', () => {
    expect(workingSize(512, 512, 1024)).toEqual({ width: 512, height: 512 })
  })

  it('rounds both sides to multiples of 8', () => {
    const { width, height } = workingSize(1001, 667, 1024)
    expect(width % 8).toBe(0)
    expect(height % 8).toBe(0)
    expect({ width, height }).toEqual({ width: 1000, height: 664 })
  })

  it('does not exceed the limit after rounding', () => {
    expect(workingSize(1023, 10, 1020).width).toBeLessThanOrEqual(1020)
  })

  it('keeps tiny dimensions valid', () => {
    expect(workingSize(2000, 3, 1024).height).toBe(8)
  })
})

describe('maxSideFor', () => {
  it('uses 768 for SD 1.5 and 1024 otherwise', () => {
    expect(maxSideFor('sd15')).toBe(768)
    expect(maxSideFor('sdxl')).toBe(1024)
    expect(maxSideFor('flux')).toBe(1024)
  })
})
