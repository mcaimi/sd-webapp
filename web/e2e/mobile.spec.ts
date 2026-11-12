import { expect, test, waitForJobsToSettle } from './fixtures'

test('generates from the mobile bottom bar with settings in a sheet', async ({ page }) => {
  await page.goto('/generate/flux')
  await page.getByLabel('Prompt', { exact: true }).fill('a neon sign that says HELLO')

  await page.getByRole('button', { name: 'Generation settings' }).click()
  const sheet = page.getByRole('dialog', { name: /Settings/ })
  await expect(sheet.getByText('True CFG')).toBeVisible()
  await page.keyboard.press('Escape')

  await page
    .getByRole('button', { name: /^Generate$/ })
    .last()
    .click()
  await waitForJobsToSettle(page)
  await expect(page.getByRole('button', { name: /^Open image/ })).toBeVisible()
})

test('navigates with the sidebar drawer and has no horizontal overflow', async ({ page }) => {
  await page.goto('/generate/sdxl')
  await page.getByRole('button', { name: 'Toggle Sidebar' }).first().click()
  await page.getByRole('link', { name: 'Gallery' }).click()
  await expect(page).toHaveURL(/\/gallery$/)
  await expect(page.getByRole('link', { name: 'Gallery' })).toBeHidden()

  for (const path of ['/generate/sdxl', '/inpaint/sdxl', '/merge/sdxl', '/gallery', '/system']) {
    await page.goto(path)
    await page.waitForTimeout(500)
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)
    expect(overflow, `horizontal overflow on ${path}`).toBeLessThanOrEqual(0)
  }
})
