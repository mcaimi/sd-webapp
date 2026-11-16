import { describe, expect, it } from 'vitest'

import { ApiError } from '@/api/client'
import type { CivitaiFile } from '@/api/types'

import {
  canCheckLayout,
  civitaiThumbUrl,
  downloadConflict,
  downloadProgressText,
  htmlToText,
  isNsfwPreview,
} from './civitai'

describe('civitaiThumbUrl', () => {
  it('asks Civitai for a resized rendition', () => {
    expect(civitaiThumbUrl('https://image.civitai.com/x/abc/original=true/1.jpeg')).toBe(
      'https://image.civitai.com/x/abc/width=450/1.jpeg',
    )
    expect(civitaiThumbUrl('/local/1.png')).toBe('/local/1.png')
  })
})

describe('isNsfwPreview', () => {
  it('flags previews rated above the SFW level', () => {
    expect(isNsfwPreview({ nsfw_level: 1 })).toBe(false)
    expect(isNsfwPreview({ nsfw_level: 4 })).toBe(true)
    expect(isNsfwPreview(null)).toBe(false)
  })
})

describe('htmlToText', () => {
  it('keeps text and line breaks, never markup', () => {
    const html = '<p>Hello <b>world</b></p><p>Line<br>break</p><img src=x onerror="alert(1)"><script>bad()</script>'
    expect(htmlToText(html)).toBe('Hello world\nLine\nbreak\nbad()')
    expect(htmlToText(null)).toBe('')
  })
})

describe('canCheckLayout', () => {
  const file = { target_resource_type: 'checkpoints', format: 'SafeTensor' } as CivitaiFile
  it('only allows Flux safetensors checkpoints', () => {
    expect(canCheckLayout({ model_type: 'flux' }, file)).toBe(true)
    expect(canCheckLayout({ model_type: 'sdxl' }, file)).toBe(false)
    expect(canCheckLayout({ model_type: 'flux' }, { ...file, format: 'PickleTensor' })).toBe(false)
  })
})

describe('downloadConflict', () => {
  it('reads the 409 detail shapes', () => {
    const warnings = ['Made for SDXL 1.0 (sdxl), not sd15']
    expect(downloadConflict(new ApiError(409, { message: 'Base model mismatch', warnings }))).toEqual({
      kind: 'mismatch',
      message: 'Base model mismatch',
      warnings,
    })
    expect(downloadConflict(new ApiError(409, { message: 'Already downloaded', path: '/m/a.safetensors' }))).toEqual({
      kind: 'downloaded',
      message: 'Already downloaded',
      path: '/m/a.safetensors',
    })
    expect(downloadConflict(new ApiError(409, { message: 'Busy', job_id: 'j1' }))).toEqual({
      kind: 'in-progress',
      message: 'Busy',
      jobId: 'j1',
    })
  })

  it('ignores other errors', () => {
    expect(downloadConflict(new ApiError(400, 'bad'))).toBeUndefined()
    expect(downloadConflict(new Error('x'))).toBeUndefined()
  })
})

describe('downloadProgressText', () => {
  it('describes bytes, speed and time left', () => {
    const detail = {
      phase: 'downloading',
      bytes_done: 1.5 * 2 ** 30,
      bytes_total: 3 * 2 ** 30,
      speed_bps: 2 ** 24,
      eta_s: 80,
    }
    expect(downloadProgressText(detail)).toBe('1.5 GB of 3 GB · 16 MB/s · 1:20 left')
    expect(downloadProgressText({ ...detail, eta_s: null, speed_bps: 0 })).toBe('1.5 GB of 3 GB')
    expect(downloadProgressText({ ...detail, phase: 'verifying' })).toBe('Verifying 3 GB…')
    expect(downloadProgressText(null)).toBeUndefined()
  })
})
