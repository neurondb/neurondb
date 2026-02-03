import { test, expect } from '@playwright/test'

test.describe('Command Palette', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/auth/me**', async (route) => {
      await route.fulfill({ status: 200, json: { id: 'test', username: 'testuser' } })
    })
    await page.route('**/factory/setup-state**', async (route) => {
      await route.fulfill({ json: { setup_complete: true } })
    })
    await page.route('**/profiles**', async (route) => {
      await route.fulfill({ json: [{ id: 'p1', name: 'Default', is_default: true }] })
    })
  })

  test('should open command palette with Cmd+K', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('domcontentloaded')
    await page.keyboard.press(process.platform === 'darwin' ? 'Meta+k' : 'Control+k')
    await expect(page.getByPlaceholder('Type a command or search...')).toBeVisible({ timeout: 3000 })
  })

  test('should navigate when selecting a command', async ({ page }) => {
    await page.goto('/')
    await page.waitForLoadState('domcontentloaded')
    await page.keyboard.press('Meta+k')
    await expect(page.getByPlaceholder('Type a command or search...')).toBeVisible({ timeout: 3000 })
    await page.getByPlaceholder('Type a command or search...').fill('dashboard')
    await page.getByText('Go to Dashboard').click()
    await expect(page).toHaveURL(/dashboard/)
  })
})
