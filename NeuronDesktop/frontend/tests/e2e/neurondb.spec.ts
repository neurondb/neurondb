import { test, expect } from '@playwright/test'

test.describe('NeuronDB Console', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/auth/me**', async (route) => {
      await route.fulfill({ status: 200, json: { id: 'test', username: 'testuser' } })
    })
    await page.route('**/profiles**', async (route) => {
      await route.fulfill({ json: [{ id: 'p1', name: 'Default', is_default: true }] })
    })
    await page.route('**/profiles/*/neurondb/**', async (route) => {
      const url = route.request().url()
      if (url.includes('collections')) {
        await route.fulfill({ json: [] })
      } else if (url.includes('search')) {
        await route.fulfill({ json: [] })
      } else {
        await route.fulfill({ json: { data: [] } })
      }
    })
  })

  test('should display NeuronDB console', async ({ page }) => {
    await page.goto('/neurondb')
    await page.waitForLoadState('domcontentloaded')
    await expect(page.getByText(/NeuronDB|Search|Collections|SQL/i)).toBeVisible({ timeout: 10000 })
  })

  test('should have profile selector', async ({ page }) => {
    await page.goto('/neurondb')
    await page.waitForLoadState('domcontentloaded')
    await expect(page.getByText(/Profile|Select/i)).toBeVisible({ timeout: 10000 })
  })

  test('should navigate to NeuronDB sub-routes', async ({ page }) => {
    await page.goto('/neurondb/analytics')
    await expect(page).toHaveURL(/neurondb\/analytics/)

    await page.goto('/neurondb/vectors')
    await expect(page).toHaveURL(/neurondb\/vectors/)
  })
})
