import { expect, expectAccessible, test, waitForJobsToSettle } from './fixtures'

const PAGES = ['/generate/sdxl', '/inpaint/sdxl', '/merge/sdxl', '/models/sd15', '/gallery', '/system']

for (const theme of ['dark', 'light'] as const) {
  test.describe(`${theme} theme`, () => {
    test.beforeEach(async ({ page }) => {
      await page.addInitScript((t) => localStorage.setItem('theme', t), theme)
    })

    for (const path of PAGES) {
      test(`${path} has no serious accessibility violations`, async ({ page }) => {
        await page.goto(path)
        await expect(page.getByRole('heading').or(page.getByRole('tab')).first()).toBeVisible()
        // Let entry animations finish so contrast is measured on final colours.
        await page.waitForTimeout(800)
        await expectAccessible(page, `${path} (${theme})`)
      })
    }
  })
}

test('lightbox and jobs panel are accessible', async ({ page }) => {
  await page.goto('/generate/sd15')
  await page.getByLabel('Prompt', { exact: true }).fill('accessibility check')
  await page.getByRole('button', { name: /^Generate$/ }).click()
  await waitForJobsToSettle(page)
  await page
    .getByRole('button', { name: /^Open image/ })
    .first()
    .click()
  await page.waitForTimeout(600)
  await expectAccessible(page, 'lightbox')
  await page.keyboard.press('Escape')

  await page.getByRole('button', { name: /^Jobs \(/ }).click()
  await page.waitForTimeout(400)
  await expectAccessible(page, 'jobs panel')
})

test('Civitai browser and model sheet are accessible', async ({ page }) => {
  await page.goto('/models/sd15')
  await page.getByRole('tab', { name: 'Browse Civitai' }).click()
  await expect(page.getByRole('list', { name: 'Civitai models' }).getByRole('listitem').first()).toBeVisible()
  await page.waitForTimeout(800)
  await expectAccessible(page, 'Civitai browser')

  await page.getByRole('list', { name: 'Civitai models' }).getByRole('button').first().click()
  await expect(page.getByRole('dialog').getByRole('heading', { name: 'Files' })).toBeVisible()
  await page.waitForTimeout(600)
  await expectAccessible(page, 'Civitai model sheet')
})
