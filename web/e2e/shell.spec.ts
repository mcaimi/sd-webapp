import { expect, test } from './fixtures'

test('redirects to generate for the default family', async ({ page }) => {
  await page.goto('/')
  await expect(page).toHaveURL(/\/generate\/sdxl$/)
  await expect(page.getByRole('navigation', { name: 'Breadcrumb' })).toContainText('SDXL')
  await expect(page.getByText('API online')).toBeVisible()
})

test('family switcher keeps the current tool and is remembered', async ({ page }) => {
  await page.goto('/merge/sdxl')
  await page.getByRole('button', { name: /SD Studio/ }).click()
  await page.getByRole('menuitem', { name: /Flux/ }).click()
  await expect(page).toHaveURL(/\/merge\/flux$/)

  await page.goto('/')
  await expect(page).toHaveURL(/\/generate\/flux$/)
})

test('command palette navigates and switches theme', async ({ page }) => {
  await page.goto('/generate/sdxl')
  // Wait for the lazy-loaded page so the shortcut listener is attached.
  await expect(page.getByLabel('Prompt', { exact: true })).toBeVisible()
  await page.keyboard.press('ControlOrMeta+k')
  const palette = page.getByRole('dialog')
  await palette.getByPlaceholder(/Search pages/).fill('system')
  await page.keyboard.press('Enter')
  await expect(page).toHaveURL(/\/system$/)

  await page.keyboard.press('ControlOrMeta+k')
  await palette.getByPlaceholder(/Search pages/).fill('light')
  await palette.getByRole('option', { name: 'Light theme' }).click()
  await expect(page.locator('html')).not.toHaveClass(/dark/)
  await page.reload()
  await expect(page.locator('html')).not.toHaveClass(/dark/)
})

test('unknown routes and families show a 404', async ({ page }) => {
  await page.goto('/generate/sd3')
  await expect(page.getByRole('heading', { name: 'Page not found' })).toBeVisible()
  await page.goto('/nope')
  await expect(page.getByRole('heading', { name: 'Page not found' })).toBeVisible()
})
