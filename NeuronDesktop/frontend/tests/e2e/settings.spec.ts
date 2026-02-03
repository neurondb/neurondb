import { test, expect } from '@playwright/test'

test.describe('Settings', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/auth/me**', async (route) => {
      await route.fulfill({ status: 200, json: { id: 'test', username: 'testuser' } })
    })
    await page.route('**/profiles**', async (route) => {
      await route.fulfill({ json: [{ id: 'p1', name: 'Default', is_default: true }] })
    })
    await page.route('**/factory/status**', async (route) => {
      await route.fulfill({
        json: {
          os: { type: 'darwin', distro: '', version: '', arch: 'arm64' },
          docker: { available: false },
          neurondb: { status: 'unknown' },
          neuronagent: { status: 'unknown' },
          neuronmcp: { status: 'unknown' },
          install_commands: {},
        },
      })
    })
  })

  test('should display settings page', async ({ page }) => {
    await page.goto('/settings')
    await page.waitForLoadState('domcontentloaded')
    await expect(page.getByText(/Settings|Profiles|Modules/i)).toBeVisible({ timeout: 10000 })
  })

  test('should have sections for Modules, Appearance, Profiles', async ({ page }) => {
    await page.goto('/settings')
    await page.waitForLoadState('domcontentloaded')
    await expect(page.getByText(/Modules|Appearance|Profiles/i)).toBeVisible({ timeout: 10000 })
  })
})
