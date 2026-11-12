import { act, renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

async function freshDrafts() {
  vi.resetModules()
  return import('./drafts')
}

describe('drafts', () => {
  it('persists patches and shares updates between hooks with the same key', async () => {
    const { useDraft } = await freshDrafts()
    const fallback = () => ({ prompt: '', steps: 20 })
    const a = renderHook(() => useDraft('generate:sdxl', fallback))
    const b = renderHook(() => useDraft('generate:sdxl', fallback))
    act(() => a.result.current[1]({ prompt: 'cat' }))
    expect(b.result.current[0]).toEqual({ prompt: 'cat', steps: 20 })
    expect(JSON.parse(localStorage.getItem('sd-webapp.draft.generate:sdxl')!)).toEqual({ prompt: 'cat', steps: 20 })
  })

  it('merges stored drafts over defaults so new fields get values', async () => {
    localStorage.setItem('sd-webapp.draft.generate:sd15', JSON.stringify({ prompt: 'saved' }))
    const { useDraft } = await freshDrafts()
    const { result } = renderHook(() => useDraft('generate:sd15', () => ({ prompt: '', count: 1 })))
    expect(result.current[0]).toEqual({ prompt: 'saved', count: 1 })
  })

  it('lets non-hook code write a draft (remix)', async () => {
    const { useDraft, writeDraft } = await freshDrafts()
    const fallback = () => ({ prompt: '' })
    const { result } = renderHook(() => useDraft('generate:flux', fallback))
    act(() => writeDraft('generate:flux', fallback, { prompt: 'from gallery' }))
    expect(result.current[0].prompt).toBe('from gallery')
  })
})
