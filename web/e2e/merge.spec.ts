import { expect, test, waitForJobsToSettle } from './fixtures'

test('single merge, then use the result in generate', async ({ page }) => {
  await page.goto('/merge/sd15')
  const name = page.getByLabel('Output filename')
  await name.fill('e2e_merged.safetensors')
  await page.getByRole('button', { name: /^Merge$/ }).click()
  await waitForJobsToSettle(page)

  const jobs = page.getByRole('region', { name: 'Merge jobs' })
  const download = jobs.getByRole('link', { name: /Download/ }).or(jobs.getByRole('button', { name: /Download/ }))
  await expect(download.first()).toBeVisible()

  await jobs.getByRole('button', { name: 'Use in Generate' }).click()
  await expect(page).toHaveURL(/\/generate\/sd15$/)
  await expect(page.getByRole('combobox', { name: 'Checkpoint' })).toContainText('e2e_merged')
})

test('asks before overwriting an existing checkpoint', async ({ page }) => {
  await page.goto('/merge/sd15')
  await page.getByLabel('Output filename').fill('dreamshaper_8.safetensors')
  await page.getByRole('button', { name: /^Merge$/ }).click()

  const dialog = page.getByRole('alertdialog')
  await expect(dialog).toContainText('Overwrite dreamshaper_8.safetensors?')
  await dialog.getByRole('button', { name: 'Keep existing' }).click()
  await expect(dialog).toBeHidden()
  expect(await page.evaluate(() => localStorage.getItem('sd-webapp.jobs'))).toBeNull()
})

test('builds and runs a recipe', async ({ page }) => {
  await page.goto('/merge/sdxl')
  await page.getByRole('tab', { name: 'Recipe' }).click()
  await page.getByRole('button', { name: 'Add step' }).click()
  await page.getByRole('button', { name: 'Add step' }).click()
  await expect(page.getByRole('button', { name: 'Remove step 2' })).toBeVisible()
  await page.getByRole('button', { name: 'Move step 2 up' }).click()
  await page.getByLabel('Output filename').fill('e2e_recipe')
  await page
    .getByRole('button', { name: /^(Run|Merge|Execute)/ })
    .last()
    .click()
  await waitForJobsToSettle(page)
  await expect(page.getByRole('region', { name: 'Merge jobs' }).getByText('e2e_recipe').first()).toBeVisible()
})
