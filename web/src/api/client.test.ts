import { describe, expect, it } from 'vitest'

import { ApiError, unwrap } from './client'

const response = (status: number) => new Response(null, { status })

describe('unwrap', () => {
  it('returns data for successful responses', async () => {
    await expect(unwrap(Promise.resolve({ data: { ok: 1 }, response: response(200) }))).resolves.toEqual({ ok: 1 })
  })

  it('throws ApiError with a FastAPI string detail', async () => {
    const result = unwrap(Promise.resolve({ error: { detail: 'Job not found' }, response: response(404) }))
    await expect(result).rejects.toBeInstanceOf(ApiError)
    await expect(result).rejects.toMatchObject({ status: 404, message: 'Job not found' })
  })

  it('formats FastAPI validation errors', async () => {
    const detail = [{ loc: ['body', 'model_checkpoint'], msg: 'Field required', type: 'missing' }]
    await expect(unwrap(Promise.resolve({ error: { detail }, response: response(422) }))).rejects.toThrow(
      'model_checkpoint: Field required',
    )
  })
})
