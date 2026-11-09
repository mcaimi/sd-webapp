import type { APIRequestContext } from '@playwright/test'

import { MOCK_URL } from '../playwright.config'
import { expect, test } from './fixtures'

const outputCount = async (request: APIRequestContext) =>
  ((await (await request.get(`${MOCK_URL}/api/v1/outputs/?limit=1`)).json()) as { total: number }).total

test('browses, remixes and opens metadata', async ({ page }) => {
  await page.goto('/gallery')
  const grid = page.getByRole('list', { name: 'Saved images' })
  await expect(grid.getByRole('listitem').first()).toBeVisible()
  // Infinite scroll loads the second page (30 seeded outputs, 24 per page).
  await page.mouse.wheel(0, 10_000)
  await expect(grid.getByRole('listitem')).toHaveCount(30)

  await grid
    .getByRole('button', { name: /^Open image/ })
    .first()
    .click()
  const viewer = page.getByRole('dialog')
  await expect(viewer.getByText('Scheduler')).toBeVisible()
  await viewer.getByRole('button', { name: /Remix/ }).click()
  await expect(page).toHaveURL(/\/generate\//)
  await expect(page.getByLabel('Prompt', { exact: true })).not.toHaveValue('')
})

test('delete can be undone, and otherwise really deletes', async ({ page, request }) => {
  await page.goto('/gallery')
  const grid = page.getByRole('list', { name: 'Saved images' })
  await expect(grid.getByRole('listitem').first()).toBeVisible()
  const total = await outputCount(request)

  await grid.getByRole('listitem').first().hover()
  await grid
    .getByRole('button', { name: /^Delete/ })
    .first()
    .click()
  await page.getByRole('button', { name: 'Undo' }).click()
  await page.waitForTimeout(6_000)
  expect(await outputCount(request)).toBe(total)

  await grid.getByRole('listitem').first().hover()
  await grid
    .getByRole('button', { name: /^Delete/ })
    .first()
    .click()
  await expect.poll(() => outputCount(request), { timeout: 10_000 }).toBe(total - 1)
})
