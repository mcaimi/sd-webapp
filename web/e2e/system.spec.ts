import { expect, test } from './fixtures'

test('shows live stats and keyboard-explorable history', async ({ page }) => {
  await page.goto('/system')
  await expect(page.getByText('NVIDIA RTX 4090 (mock)')).toBeVisible()
  await expect(page.getByRole('meter', { name: 'Host memory' })).toBeVisible()
  // History needs a couple of samples (3s refresh).
  await page.waitForTimeout(7_000)
  const chart = page
    .getByRole('img', { name: /^CPU/ })
    .or(page.getByRole('group', { name: /^CPU/ }))
    .first()
  await expect(chart).toBeVisible()
  await chart.focus()
  await page.keyboard.press('End')
  await expect(page.getByRole('table', { name: /CPU/ }).or(page.locator('table')).first()).toBeAttached()
})
