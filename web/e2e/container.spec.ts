import { expect, test } from './fixtures'

// Behaviour provided by the nginx container, not the app. Runs only against a
// deployment: E2E_BASE_URL=http://localhost:8080 npm run e2e
test.skip(!process.env.E2E_BASE_URL, 'needs E2E_BASE_URL pointing at the container')

test('serves the SPA with security headers and deep-link fallback', async ({ request }) => {
  const index = await request.get('/generate/flux')
  expect(index.status()).toBe(200)
  expect(await index.text()).toContain('<div id="root">')
  const headers = index.headers()
  expect(headers['cache-control']).toBe('no-cache')
  expect(headers['content-security-policy']).toContain("script-src 'self'")
  expect(headers['x-content-type-options']).toBe('nosniff')
  expect(headers['x-frame-options']).toBe('DENY')
})

test('caches hashed assets forever', async ({ request }) => {
  const html = await (await request.get('/')).text()
  const asset = html.match(/src="(\/assets\/[^"]+\.js)"/)![1]
  const res = await request.get(asset)
  expect(res.status()).toBe(200)
  expect(res.headers()['cache-control']).toContain('immutable')
  expect(res.headers()['content-security-policy']).toBeTruthy()
  expect((await request.get('/assets/missing.js')).status()).toBe(404)
})

test('health endpoint and API proxy', async ({ request }) => {
  expect(await (await request.get('/healthz')).text()).toBe('ok\n')
  const health = await request.get('/api/v1/system/health')
  expect(health.status()).toBe(200)
  expect(await health.json()).toMatchObject({ status: 'healthy' })
})

test('accepts large inpainting payloads', async ({ request }) => {
  // ~5 MB of base64, like a 1024px PNG plus mask. nginx's default limit would answer 413.
  const res = await request.post('/api/v1/sdxl/inpaint', {
    data: { positive_prompt: 'x', image_base64: 'A'.repeat(5 * 1024 * 1024) },
  })
  expect(res.status()).not.toBe(413)
  expect(res.status()).toBe(422)
})
