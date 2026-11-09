import { describe, expect, it } from 'vitest'

import { hasPaint, invertMask, pushBounded } from './mask-utils'

describe('hasPaint', () => {
  it('detects any non-transparent pixel', () => {
    const rgba = new Uint8ClampedArray(16)
    expect(hasPaint(rgba)).toBe(false)
    rgba[11] = 1
    expect(hasPaint(rgba)).toBe(true)
  })
})

describe('invertMask', () => {
  it('flips alpha and recolours every pixel', () => {
    const rgba = new Uint8ClampedArray([0, 0, 0, 0, 10, 20, 30, 255, 1, 2, 3, 100])
    invertMask(rgba, [9, 8, 7])
    expect([...rgba]).toEqual([9, 8, 7, 255, 9, 8, 7, 0, 9, 8, 7, 155])
  })
})

describe('pushBounded', () => {
  it('drops the oldest item past the limit', () => {
    const stack = [1, 2, 3]
    pushBounded(stack, 4, 3)
    expect(stack).toEqual([2, 3, 4])
  })
})
