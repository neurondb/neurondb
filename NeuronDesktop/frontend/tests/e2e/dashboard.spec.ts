import { test, expect } from '@playwright/test'

test.describe('Dashboard', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/auth/me**', async (route) => {
      await route.fulfill({ status: 200, json: { id: 'test', username: 'testuser' } })
    })
    await page.route('**/profiles**', async (route) => {
      await route.fulfill({ json: [{ id: 'p1', name: 'Default', is_default: true }] })
    })
    await page.route('**/profiles/*/dashboard**', async (route) => {
      await route.fulfill({
        json: {
          system_metrics: { total_requests: 100, successful_requests: 95 },
          neurondb_stats: { collections_count: 2, total_vectors: 1000, indexes_count: 1, avg_query_time: 5.2 },
          health_status: { components: { neurondb: 'healthy', neuronagent: 'unknown', neuronmcp: 'unknown' } },
        },
      })
    })
  })

  test('should display dashboard', async ({ page }) => {
    await page.goto('/dashboard')
    await page.waitForLoadState('domcontentloaded')
    await expect(page.getByText(/Dashboard|System Metrics|NeuronDB/i)).toBeVisible({ timeout: 10000 })
  })
})
