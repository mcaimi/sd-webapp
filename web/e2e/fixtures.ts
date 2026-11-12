import AxeBuilder from '@axe-core/playwright'
import { test as base, expect, type Page } from '@playwright/test'

import { MOCK_URL } from '../playwright.config'

/**
 * Every test starts from a clean mock backend and empty browser storage, and
 * fails if the page logs a console error.
 */
export const test = base.extend<{ consoleErrors: string[] }>({
  consoleErrors: [
    async ({ page, request }, use) => {
      await request.post(`${MOCK_URL}/__mock/reset`)
      const errors: string[] = []
      page.on('console', (msg) => msg.type() === 'error' && errors.push(msg.text()))
      page.on('pageerror', (err) => errors.push(String(err)))
      await use(errors)
      expect(errors, 'console errors').toEqual([])
    },
    { auto: true },
  ],
})

export { expect }

/** Fail on serious or critical WCAG 2.1 AA violations. */
export async function expectAccessible(page: Page, context = '') {
  const results = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']).analyze()
  const blocking = results.violations.filter((v) => v.impact === 'serious' || v.impact === 'critical')
  const summary = blocking.map(
    (v) => `${v.id} (${v.impact}): ${v.help}\n  ${v.nodes.map((n) => n.target.join(' ')).join('\n  ')}`,
  )
  expect(summary, `accessibility violations ${context}`).toEqual([])
}

/** Wait until every tracked job has finished (completed, failed, …). */
export async function waitForJobsToSettle(page: Page, timeout = 20_000) {
  await page.waitForFunction(
    () => {
      const jobs = JSON.parse(localStorage.getItem('sd-webapp.jobs') ?? '[]') as { status: string }[]
      return jobs.length > 0 && jobs.every((j) => j.status !== 'queued' && j.status !== 'running')
    },
    null,
    { timeout },
  )
}

export async function trackedJobs(page: Page) {
  return page.evaluate(
    () =>
      JSON.parse(localStorage.getItem('sd-webapp.jobs') ?? '[]') as { kind: string; status: string; label: string }[],
  )
}
