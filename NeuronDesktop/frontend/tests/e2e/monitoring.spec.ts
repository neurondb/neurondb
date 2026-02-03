import { test, expect } from '@playwright/test'

test.describe('Monitoring', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/auth/me**', async (route) => {
      await route.fulfill({ status: 200, json: { id: 'test', username: 'testuser' } })
    })
    await page.route('**/profiles**', async (route) => {
      await route.fulfill({ json: [{ id: 'p1', name: 'Default', is_default: true }] })
    })
  })

  test('should display monitoring page', async ({ page }) => {
    await page.goto('/monitoring')
    await page.waitForLoadState('domcontentloaded')
    await expect(page.getByText(/Monitoring|Metrics/i)).toBeVisible({ timeout: 10000 })
  })
})
