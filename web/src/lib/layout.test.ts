import { describe, expect, it } from 'vitest'

import { DEFAULT_LAYOUT, withMenu, withParams } from './layout'

describe('layout', () => {
  it('keeps the menu within its range', () => {
    expect(withMenu(DEFAULT_LAYOUT, 5).menu).toBe(12)
    expect(withMenu(DEFAULT_LAYOUT, 60).menu).toBe(35)
  })

  it('shrinks the parameters when the menu leaves too little room for results', () => {
    expect(withMenu({ menu: 20, params: 60 }, 35)).toEqual({ menu: 35, params: 50 })
    expect(withMenu(DEFAULT_LAYOUT, 30)).toEqual({ menu: 30, params: 50 })
  })

  it('keeps room for the results when widening the parameters', () => {
    expect(withParams(DEFAULT_LAYOUT, 90).params).toBe(65)
    expect(withParams(DEFAULT_LAYOUT, 5).params).toBe(20)
  })
})
