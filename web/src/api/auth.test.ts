import { afterEach, describe, expect, it, vi } from 'vitest'

import { auth } from './auth'

function serverAnswers(...answers: (number | object)[]) {
  const fetch = vi.fn<typeof globalThis.fetch>()
  for (const answer of answers) {
    fetch.mockResolvedValueOnce(
      typeof answer === 'number' ? new Response(null, { status: answer }) : Response.json(answer),
    )
  }
  vi.stubGlobal('fetch', fetch)
  return fetch
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('auth', () => {
  it('signs out when the session is not authenticated', async () => {
    serverAnswers({ authenticated: false })
    await auth.check()
    expect(auth.get()).toBe('signed-out')
  })

  it('stays usable behind a server without the gate', async () => {
    serverAnswers(404)
    await auth.check()
    expect(auth.get()).toBe('signed-in')
  })

  it('shares one request between concurrent checks', async () => {
    const fetch = serverAnswers({ authenticated: true })
    await Promise.all([auth.check(), auth.check()])
    expect(fetch).toHaveBeenCalledTimes(1)
  })

  it('sends the token as a bearer header and reports a rejected token', async () => {
    const fetch = serverAnswers(401, 204)
    expect(await auth.signIn('wrong')).toBe(false)
    expect(await auth.signIn('  right-token \n')).toBe(true)
    expect(auth.get()).toBe('signed-in')
    expect(fetch.mock.calls[1][1]).toMatchObject({ method: 'POST', headers: { Authorization: 'Bearer right-token' } })
  })

  it('signs out', async () => {
    serverAnswers(204)
    await auth.signOut()
    expect(auth.get()).toBe('signed-out')
  })
})
