import { E2E_TOKEN } from '../playwright.config'
import { expect, expectAccessible, test } from './fixtures'

test.beforeEach(async ({ context }) => {
  await context.clearCookies()
})

// Rejected tokens and API calls without a session are logged by the browser as failed requests.
function dropUnauthorized(errors: string[]) {
  const others = errors.filter((e) => !e.includes('status of 401'))
  errors.splice(0, errors.length, ...others)
}

test('asks for the access token before showing the app', async ({ page, request, consoleErrors }) => {
  await page.goto('/generate/sdxl')
  const token = page.getByLabel('Access token')
  await expect(token).toBeVisible()
  await expect(page.getByRole('navigation', { name: 'Breadcrumb' })).toHaveCount(0)
  await expectAccessible(page, 'token prompt')
  expect((await request.get('/api/v1/system/health')).status()).toBe(401)

  await token.fill('not-the-token')
  await page.getByRole('button', { name: 'Continue' }).click()
  await expect(page.getByRole('alert')).toHaveText("That token isn't valid.")
  dropUnauthorized(consoleErrors)

  await token.fill(E2E_TOKEN)
  await page.getByRole('button', { name: 'Continue' }).click()
  await expect(page.getByText('API online')).toBeVisible()

  // The session survives a reload.
  await page.reload()
  await expect(page.getByText('API online')).toBeVisible()

  await page.getByRole('button', { name: 'Sign out' }).click()
  await expect(page.getByLabel('Access token')).toBeVisible()
  await page.reload()
  await expect(page.getByLabel('Access token')).toBeVisible()
})

test('asks again when the session is lost while using the app', async ({ page, context, consoleErrors }) => {
  await page.goto('/generate/sdxl')
  await page.getByLabel('Access token').fill(E2E_TOKEN)
  await page.getByRole('button', { name: 'Continue' }).click()
  await expect(page.getByText('API online')).toBeVisible()

  // e.g. the server restarted with a new token.
  await context.clearCookies()
  await expect(page.getByLabel('Access token')).toBeVisible({ timeout: 15_000 })
  dropUnauthorized(consoleErrors)
})
