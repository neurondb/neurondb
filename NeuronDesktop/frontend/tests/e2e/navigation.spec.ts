import { test, expect } from '@playwright/test'

test.describe('Navigation', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/auth/me**', async (route) => {
      await route.fulfill({ status: 200, json: { id: 'test', username: 'testuser' } })
    })
    await page.route('**/factory/setup-state**', async (route) => {
      await route.fulfill({ json: { setup_complete: true } })
    })
    await page.route('**/profiles**', async (route) => {
      const data = route.request().method() === 'POST' ? { id: 'p1', name: 'Default' } : [{ id: 'p1', name: 'Default', is_default: true }]
      await route.fulfill({ json: data })
    })
    await page.route('**/dashboard**', async (route) => {
      await route.fulfill({
        json: {
          system_metrics: { total_requests: 0, successful_requests: 0 },
          neurondb_stats: { collections_count: 0, total_vectors: 0, indexes_count: 0, avg_query_time: 0 },
          health_status: { components: {} },
        },
      })
    })
  })

  test('sidebar links navigate correctly', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('domcontentloaded')

    const linksToTest = [
      { name: 'Home', urlPart: '/', text: /NeuronDesktop|Features/ },
      { name: 'Dashboard', urlPart: 'dashboard', text: /Dashboard|System Metrics|Loading/ },
      { name: 'Settings', urlPart: 'settings', text: /Settings|Profiles|Modules/ },
    ]

    for (const link of linksToTest) {
      const navLink = page.getByRole('link', { name: new RegExp(link.name, 'i') })
      await navLink.first().click()
      await expect(page).toHaveURL(new RegExp(link.urlPart))
      await expect(page.getByText(link.text)).toBeVisible({ timeout: 10000 })
      if (link.urlPart !== '/') {
        await page.goto('/')
        await page.waitForLoadState('domcontentloaded')
      }
    }
  })

  test('home page feature cards navigate correctly', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('domcontentloaded')

    const card = page.getByRole('link', { name: /MCP Console/i })
    await expect(card.first()).toBeVisible()
    await card.first().click()
    await expect(page).toHaveURL(/mcp/)
  })

  test('no 404 on valid routes', async ({ page }) => {
    const routes = ['/', '/dashboard', '/setup', '/mcp', '/models', '/settings']
    for (const route of routes) {
      const response = await page.goto(route)
      expect(response?.status()).not.toBe(404)
    }
  })
})
