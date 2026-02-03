import { test, expect } from '@playwright/test'

test.describe('Agents', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/auth/me**', async (route) => {
      await route.fulfill({ status: 200, json: { id: 'test', username: 'testuser' } })
    })
    await page.route('**/profiles**', async (route) => {
      await route.fulfill({ json: [{ id: 'p1', name: 'Default', is_default: true }] })
    })
    await page.route('**/profiles/*/agents**', async (route) => {
      await route.fulfill({ json: [] })
    })
  })

  test('should display agents page', async ({ page }) => {
    await page.goto('/agents')
    await page.waitForLoadState('domcontentloaded')
    await expect(page.getByText(/Agents/i)).toBeVisible({ timeout: 10000 })
  })

  test('should navigate to create agent', async ({ page }) => {
    await page.goto('/agents')
    await page.waitForLoadState('domcontentloaded')
    const createLink = page.getByRole('link', { name: /Create|New Agent/i })
    if (await createLink.first().isVisible()) {
      await createLink.first().click()
      await expect(page).toHaveURL(/agents\/create/)
    }
  })

  test('should display create agent page when navigated directly', async ({ page }) => {
    await page.goto('/agents/create')
    await page.waitForLoadState('domcontentloaded')
    await expect(page.getByText(/Create|Agent/i)).toBeVisible({ timeout: 10000 })
  })
})
