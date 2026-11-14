import { expect, test, trackedJobs } from './fixtures'

test('lists installed models with their Civitai details and identifies unlinked ones', async ({ page }) => {
  await page.goto('/models/sd15')
  const installed = page.getByRole('list', { name: 'Installed Checkpoints' })
  await expect(installed.getByRole('listitem').filter({ hasText: 'dreamshaper_8' })).toContainText('DreamShaper · 8')

  const deliberate = installed.getByRole('listitem').filter({ hasText: 'deliberate_v3' })
  await expect(deliberate).toContainText('Not linked to Civitai')
  await deliberate.getByRole('button', { name: 'Identify' }).click()
  await expect(page.getByText('Found deliberate_v3 on Civitai')).toBeVisible()
  await expect(deliberate).toContainText('Deliberate · v3')

  await page.getByRole('radio', { name: 'LoRAs' }).click()
  await expect(page.getByRole('list', { name: 'Installed LoRAs' })).toContainText('2 trigger words')
})

test('searches Civitai, downloads a file after confirming the base model, and uses it', async ({
  page,
  consoleErrors,
}) => {
  await page.goto('/models/sd15')
  await page.getByRole('tab', { name: 'Browse Civitai' }).click()
  const results = page.getByRole('list', { name: 'Civitai models' })
  await expect(results.getByRole('listitem')).toHaveCount(24)

  await page.getByRole('searchbox', { name: 'Search Civitai' }).fill('aurora')
  await expect(results.getByRole('listitem')).toHaveCount(3)
  await results.getByRole('button', { name: /Aurora SD15 1/ }).click()

  const sheet = page.getByRole('dialog', { name: 'Aurora SD15 1' })
  await expect(sheet.getByText("Can't download: PickleTensor format not supported").first()).toBeVisible()

  // The v1.0 SDXL version is for another family: downloading asks first.
  await sheet.getByRole('combobox', { name: 'Version' }).click()
  await page.getByRole('option', { name: /v1\.0 SDXL/ }).click()
  await expect(sheet.getByText(/This version is made for Stable Diffusion XL/)).toBeVisible()
  await sheet.getByRole('button', { name: /^Download AuroraSD151_v10SDXL\.safetensors/ }).click()
  const confirm = page.getByRole('alertdialog', { name: 'Different base model' })
  await expect(confirm).toContainText('Made for SDXL 1.0 (sdxl), not sd15')
  await confirm.getByRole('button', { name: 'Download anyway' }).click()
  // The browser logs the mismatch answer (409) that triggered the confirmation; it's expected.
  const conflict = consoleErrors.findIndex((e) => e.includes('409 (Conflict)'))
  expect(conflict).toBeGreaterThanOrEqual(0)
  consoleErrors.splice(conflict, 1)

  await expect(sheet.getByText(/ of 2 GB/).first()).toBeVisible()
  await expect(sheet.getByText('Downloaded')).toBeVisible({ timeout: 15_000 })
  await expect(page.getByText('Downloaded AuroraSD151_v10SDXL.safetensors').first()).toBeVisible()
  const jobs = await trackedJobs(page)
  expect(jobs).toContainEqual(expect.objectContaining({ kind: 'download', status: 'completed' }))
  await page.keyboard.press('Escape')

  await page.goto('/generate/sd15')
  await page.getByRole('combobox', { name: 'Checkpoint' }).click()
  await expect(page.getByRole('option', { name: /AuroraSD151_v10SDXL/ })).toContainText('Aurora SD15 1 · v1.0 SDXL')
})

test('downloads can be cancelled while they run', async ({ page }) => {
  await page.goto('/models/sd15')
  await page.getByRole('tab', { name: 'Browse Civitai' }).click()
  await page.getByRole('list', { name: 'Civitai models' }).getByRole('button').first().click()
  const sheet = page.getByRole('dialog')
  await sheet.getByRole('button', { name: /^Download .*_v20\.safetensors/ }).click()

  await expect(sheet.getByText(/ of 2 GB/).first()).toBeVisible()
  await sheet.getByRole('button', { name: 'Cancel' }).first().click()
  await page.keyboard.press('Escape')
  const downloads = page.getByRole('region', { name: 'Downloads' })
  await expect(downloads.getByText('Cancelled')).toBeVisible()
})

test('NSFW previews stay blurred until revealed', async ({ page }) => {
  await page.goto('/models/sd15')
  await page.getByRole('tab', { name: 'Browse Civitai' }).click()
  await page.getByText('Include NSFW').click()
  const card = page
    .getByRole('list', { name: 'Civitai models' })
    .getByRole('listitem')
    .filter({ hasText: 'NSFW' })
    .first()
  await expect(card.locator('img')).toHaveClass(/blur/)

  await card.getByRole('button').click()
  const sheet = page.getByRole('dialog')
  await sheet.getByRole('button', { name: /NSFW preview · Show/ }).click()
  await expect(sheet.locator('img').first()).not.toHaveClass(/blur/)
})
