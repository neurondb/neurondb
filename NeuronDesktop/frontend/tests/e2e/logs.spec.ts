import { test, expect } from '@playwright/test'

test.describe('Logs', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/auth/me**', async (route) => {
      await route.fulfill({ status: 200, json: { id: 'test', username: 'testuser' } })
    })
    await page.route('**/profiles**', async (route) => {
      await route.fulfill({ json: [{ id: 'p1', name: 'Default', is_default: true }] })
    })
    await page.route('**/request-logs**', async (route) => {
      await route.fulfill({ json: { logs: [], total: 0 } })
    })
  })

  test('should display logs page', async ({ page }) => {
    await page.goto('/logs')
    await page.waitForLoadState('domcontentloaded')
    await expect(page.getByText(/Logs|Request|Inspector/i)).toBeVisible({ timeout: 10000 })
  })
})
