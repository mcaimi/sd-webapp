import { expect, test, trackedJobs, waitForJobsToSettle } from './fixtures'

test('generates a batch, survives a reload and opens the lightbox', async ({ page }) => {
  await page.goto('/generate/sdxl')
  await page.getByLabel('Prompt', { exact: true }).fill('a lighthouse at dusk')
  const images = page.getByRole('slider', { name: 'Images' })
  await images.focus()
  await page.keyboard.press('ArrowRight')
  await page.getByRole('button', { name: 'Generate 2' }).click()

  const results = page.getByRole('region', { name: 'Results' })
  await expect(results.getByRole('button', { name: /^Open image/ })).toHaveCount(2, { timeout: 15_000 })
  await expect(page.getByText('2 images ready')).toBeVisible()

  await page.reload()
  await expect(results.getByRole('button', { name: /^Open image/ })).toHaveCount(2)

  await results
    .getByRole('button', { name: /^Open image/ })
    .first()
    .click()
  const viewer = page.getByRole('dialog')
  await expect(viewer.getByRole('term').filter({ hasText: /^Seed$/ })).toBeVisible()
  await expect(viewer.getByRole('button', { name: 'Next image' })).toBeVisible()
  await page.keyboard.press('ArrowRight')
  await expect(viewer.getByRole('button', { name: 'Previous image' })).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(viewer).toBeHidden()
})

test('reports failures and allows retry', async ({ page }) => {
  await page.goto('/generate/sd15')
  await page.getByLabel('Prompt', { exact: true }).fill('this will fail')
  await page.getByLabel('Prompt', { exact: true }).press('ControlOrMeta+Enter')
  await expect(page.getByText('CUDA out of memory (mock failure)').first()).toBeVisible({ timeout: 15_000 })
  await expect(page.getByRole('button', { name: 'Retry' })).toBeVisible()
})

test('reuse settings restores the prompt and locks the seed', async ({ page }) => {
  await page.goto('/generate/sd15')
  const prompt = page.getByLabel('Prompt', { exact: true })
  await prompt.fill('original prompt')
  await page.getByRole('button', { name: /^Generate$/ }).click()
  await waitForJobsToSettle(page)

  await prompt.fill('something else')
  await page.getByRole('button', { name: 'Reuse settings' }).first().click()
  await expect(prompt).toHaveValue('original prompt')
  await expect(page.getByRole('button', { name: 'Unlock seed' })).toBeVisible()
})

test('compares two checkpoints with the same seed', async ({ page }) => {
  await page.goto('/generate/sdxl')
  await page.getByRole('radio', { name: 'Compare models' }).click()
  await page.getByLabel('Prompt', { exact: true }).fill('portrait of an astronaut')
  await page.getByRole('checkbox').nth(0).check()
  await page.getByRole('checkbox').nth(1).check()
  await page.getByRole('button', { name: /^Compare 2 models/ }).click()
  await waitForJobsToSettle(page)

  const jobs = await trackedJobs(page)
  expect(jobs.filter((j) => j.kind === 'compare' && j.status === 'completed')).toHaveLength(2)
  const seeds = await page.evaluate(() =>
    (JSON.parse(localStorage.getItem('sd-webapp.jobs')!) as { request: { seed: number } }[]).map((j) => j.request.seed),
  )
  expect(new Set(seeds).size).toBe(1)
})

test('queued jobs can be cancelled from the jobs panel', async ({ page }) => {
  await page.goto('/generate/sd15')
  const prompt = page.getByLabel('Prompt', { exact: true })
  // A batch of deliberately slow jobs ("slow" in the prompt) keeps the next one queued.
  await prompt.fill('slow batch ahead')
  await page.getByRole('slider', { name: 'Images' }).focus()
  for (let i = 0; i < 3; i++) await page.keyboard.press('ArrowRight')
  await prompt.press('ControlOrMeta+Enter')
  await page.getByRole('slider', { name: 'Images' }).focus()
  await page.keyboard.press('Home')
  await prompt.fill('second job to cancel')
  await prompt.press('ControlOrMeta+Enter')

  await page.getByRole('button', { name: /^Jobs \(/ }).click()
  const row = page
    .getByRole('dialog', { name: 'Jobs' })
    .getByRole('listitem')
    .filter({ hasText: 'second job to cancel' })
  await row.getByRole('button', { name: 'Cancel job' }).click()
  await expect(row.getByText('Cancelled')).toBeVisible()
})

test('discarding a running job asks first and explains the backend keeps running it', async ({ page }) => {
  await page.goto('/generate/sd15')
  const prompt = page.getByLabel('Prompt', { exact: true })
  await prompt.fill('slow job to discard')
  await prompt.press('ControlOrMeta+Enter')

  const results = page.getByRole('region', { name: 'Results' })
  await expect(results.getByText(/^Running/)).toBeVisible({ timeout: 15_000 })
  await results.getByRole('button', { name: 'Discard' }).click()

  const confirm = page.getByRole('alertdialog', { name: "Discard this job's result?" })
  await expect(confirm).toContainText('keeps running to completion')
  await confirm.getByRole('button', { name: 'Discard result' }).click()
  await expect(results.getByText('Cancelled')).toBeVisible()
  await waitForJobsToSettle(page)
})

test('adds a LoRA and inserts its Civitai trigger words into the prompt', async ({ page }) => {
  await page.goto('/generate/sd15')
  const prompt = page.getByLabel('Prompt', { exact: true })
  await prompt.fill('a harbour')

  await page.getByRole('button', { name: 'Add' }).click()
  await page.getByRole('option', { name: /Watercolor Style/ }).click()
  await page.getByRole('button', { name: 'wcstyle' }).click()
  await expect(prompt).toHaveValue('a harbour, wcstyle')
  await page.getByRole('button', { name: 'All' }).click()
  await expect(prompt).toHaveValue('a harbour, wcstyle, watercolor painting')
})
