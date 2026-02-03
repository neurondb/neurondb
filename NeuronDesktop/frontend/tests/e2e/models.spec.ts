import { test, expect } from '@playwright/test'

test.describe('Model Management', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/auth/me**', async (route) => {
      await route.fulfill({ status: 200, json: { id: 'test', username: 'testuser' } })
    })
    await page.route('**/profiles**', async (route) => {
      const data = route.request().method() === 'POST' ? { id: 'p1', name: 'Default' } : [{ id: 'p1', name: 'Default', is_default: true }]
      await route.fulfill({ json: data })
    })
    await page.route('**/profiles/*/models**', async (route) => {
      if (route.request().method() === 'POST') {
        await route.fulfill({ json: { id: '1', model_name: 'gpt-4', model_provider: 'openai' } })
      } else {
        await route.fulfill({ json: [] })
      }
    })
    await page.goto('/models')
    await page.waitForSelector('h1:has-text("Model Settings")', { timeout: 10000 })
  })

  test('should display models page', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'Model Settings' })).toBeVisible()
    await expect(page.getByText('Configure API keys and settings for AI models')).toBeVisible()
  })

  test('should open add model modal', async ({ page }) => {
    await page.getByRole('button', { name: 'Add Model' }).click()
    await expect(page.getByRole('heading', { name: 'Add Model Configuration' })).toBeVisible()
    await expect(page.getByText('Provider *')).toBeVisible()
    await expect(page.getByText('Model *')).toBeVisible()
  })

  test('should add a new model', async ({ page }) => {
    await page.getByRole('button', { name: 'Add Model' }).click()
    await page.locator('select').first().selectOption('openai')
    await page.locator('select').nth(1).selectOption('gpt-4')
    await page.getByPlaceholder('Enter API key').fill('sk-test-key')
    await page.getByRole('button', { name: 'Create' }).click()

    await expect(page.getByRole('heading', { name: 'Add Model Configuration' })).not.toBeVisible({ timeout: 5000 })
  })

  test('should display model list when models exist', async ({ page }) => {
    await page.route('**/profiles/*/models**', async (route) => {
      await route.fulfill({
        json: [
          { id: '1', model_name: 'gpt-4', model_provider: 'openai', api_key: 'sk-xxx', is_default: true, is_free: false },
        ],
      })
    })
    await page.reload()
    await page.waitForSelector('text=OpenAI - gpt-4', { timeout: 5000 })
    await expect(page.getByText('OpenAI - gpt-4')).toBeVisible()
  })
})
