import { test, expect } from '@playwright/test'

test.describe('MCP Console', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/auth/me**', async (route) => {
      await route.fulfill({ status: 200, json: { id: 'test', username: 'testuser' } })
    })
    await page.route('**/profiles**', async (route) => {
      await route.fulfill({ json: [{ id: 'p1', name: 'Default', is_default: true }] })
    })
    await page.route('**/profiles/*/mcp/**', async (route) => {
      await route.fulfill({ json: { tools: [], threads: [] } })
    })
    await page.route('**/profiles/*/models**', async (route) => {
      await route.fulfill({ json: [] })
    })
  })

  test('should display MCP console page', async ({ page }) => {
    await page.goto('/mcp')
    await page.waitForLoadState('domcontentloaded')
    await expect(page.getByText(/MCP|Console|Chat/i)).toBeVisible({ timeout: 10000 })
  })

  test('should have profile selector', async ({ page }) => {
    await page.goto('/mcp')
    await page.waitForLoadState('domcontentloaded')
    await expect(page.getByText(/Profile|Select/i)).toBeVisible({ timeout: 10000 })
  })

  test('should navigate to MCP sub-routes', async ({ page }) => {
    await page.goto('/mcp/tools')
    await expect(page).toHaveURL(/mcp\/tools/)

    await page.goto('/mcp/datasets')
    await expect(page).toHaveURL(/mcp\/datasets/)
  })
})
