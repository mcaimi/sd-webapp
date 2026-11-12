import { describe, expect, it } from 'vitest'

import { basename, checkpointDownloadUrl, ensureSafetensors, isBlend, suggestOutputName } from './merge-utils'

describe('merge utils', () => {
  it('adds the .safetensors extension once', () => {
    expect(ensureSafetensors('merged')).toBe('merged.safetensors')
    expect(ensureSafetensors(' merged.safetensors ')).toBe('merged.safetensors')
    expect(ensureSafetensors('   ')).toBe('')
  })

  it('takes the basename of backend paths', () => {
    expect(basename('models/checkpoints/sdxl/batch_merged/a.safetensors')).toBe('a.safetensors')
    expect(basename('a.safetensors')).toBe('a.safetensors')
  })

  it('suggests a name from the inputs', () => {
    expect(suggestOutputName('a.safetensors', 'b.safetensors', 'slerp', 0.35)).toBe('a_b_slerp_35.safetensors')
    expect(suggestOutputName('a.safetensors', '', 'linear', 0.5)).toBe('a_linear_50.safetensors')
    expect(suggestOutputName('', 'b.safetensors', 'linear', 0.5)).toBe('')
  })

  it('builds download URLs by basename', () => {
    expect(checkpointDownloadUrl('sdxl', 'models/checkpoints/sdxl/sub/my model.safetensors')).toBe(
      '/api/v1/models/sdxl/checkpoints/my%20model.safetensors/download',
    )
  })

  it('distinguishes blends from strength-based methods', () => {
    expect(isBlend('linear')).toBe(true)
    expect(isBlend('subtract')).toBe(false)
  })
})
