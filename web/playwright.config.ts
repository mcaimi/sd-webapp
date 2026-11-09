import { defineConfig, devices } from '@playwright/test'

const CI = !!process.env.CI
const MOCK_PORT = 8020
const APP_PORT = 4173
// Use a locally installed Chromium instead of Playwright's download, e.g.
// CHROMIUM_PATH=/Applications/Chromium.app/Contents/MacOS/Chromium npm run e2e
const executablePath = process.env.CHROMIUM_PATH
// Test an already-running deployment (e.g. the production container) instead of
// starting vite preview. It must proxy /api to the mock on MOCK_PORT.
const externalBaseUrl = process.env.E2E_BASE_URL

export const MOCK_URL = `http://localhost:${MOCK_PORT}`

/**
 * End-to-end tests run the production build (vite preview) against the mock
 * backend, so they exercise the shipped bundle without a GPU.
 */
export default defineConfig({
  testDir: 'e2e',
  // The mock has one sequential job queue and shared state: run tests one at a time.
  workers: 1,
  fullyParallel: false,
  retries: CI ? 1 : 0,
  forbidOnly: CI,
  reporter: CI ? [['github'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: externalBaseUrl ?? `http://localhost:${APP_PORT}`,
    trace: 'retain-on-failure',
    launchOptions: executablePath ? { executablePath } : {},
  },
  projects: [
    { name: 'desktop', use: { ...devices['Desktop Chrome'] }, testIgnore: /mobile\.spec/ },
    { name: 'mobile', use: { ...devices['Pixel 7'] }, testMatch: /mobile\.spec/ },
  ],
  webServer: [
    {
      command: 'node scripts/mock-api.mjs',
      env: { MOCK_PORT: String(MOCK_PORT), MOCK_SPEED: '8' },
      url: `${MOCK_URL}/api/v1/system/health`,
      reuseExistingServer: !CI,
    },
    ...(externalBaseUrl
      ? []
      : [
          {
            command: `npm run build && npx vite preview --port ${APP_PORT} --strictPort`,
            env: { SD_API_URL: MOCK_URL },
            url: `http://localhost:${APP_PORT}`,
            reuseExistingServer: !CI,
            timeout: 120_000,
          },
        ]),
  ],
})
