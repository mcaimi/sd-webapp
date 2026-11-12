import { expect, test, trackedJobs, waitForJobsToSettle } from './fixtures'
import { pngFile } from './png'

test('uploads an image, paints a mask and inpaints', async ({ page }) => {
  await page.goto('/inpaint/sdxl')
  await page.getByLabel('Choose an image to inpaint').setInputFiles(pngFile(1600, 1200))
  // Downscaled to SDXL's native size.
  await expect(page.getByText(/working at 1024×768/)).toBeVisible()

  const canvas = page.getByLabel('Mask canvas: paint over the area to regenerate')
  const box = (await canvas.boundingBox())!
  await page.mouse.move(box.x + box.width * 0.3, box.y + box.height * 0.3)
  await page.mouse.down()
  await page.mouse.move(box.x + box.width * 0.6, box.y + box.height * 0.6, { steps: 10 })
  await page.mouse.up()

  await page.getByLabel('Prompt', { exact: true }).fill('a red door')
  await page.getByRole('button', { name: /^Inpaint$/ }).click()
  await waitForJobsToSettle(page)
  expect((await trackedJobs(page))[0]).toMatchObject({ kind: 'inpaint', status: 'completed' })

  await page
    .getByRole('region', { name: 'Results' })
    .getByRole('button', { name: /^Open image/ })
    .click()
  const compare = page.getByRole('slider', { name: 'Compare original and inpainted image' })
  await expect(compare).toBeVisible()
  const before = await compare.getAttribute('aria-valuenow')
  await compare.focus()
  await page.keyboard.press('ArrowLeft')
  await expect(compare).not.toHaveAttribute('aria-valuenow', before!)
})

test('explains why submitting is not possible yet', async ({ page }) => {
  await page.goto('/inpaint/sd15')
  await expect(page.getByText(/choose an image/i).first()).toBeVisible()
})
