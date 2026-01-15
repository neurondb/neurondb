import { test, expect } from '@playwright/test'

test.describe('Model Management', () => {
  test.beforeEach(async ({ page }) => {
    // Mock authentication
    await page.goto('/models')
    
    // Wait for page to load
    await page.waitForSelector('h1:has-text("Model & Key Management")', { timeout: 10000 })
  })

  test('should display models page', async ({ page }) => {
    await expect(page.getByText('Model & Key Management')).toBeVisible()
    await expect(page.getByText('Manage LLM models and API keys')).toBeVisible()
  })

  test('should open add model modal', async ({ page }) => {
    await page.click('button:has-text("Add Model")')
    
    await expect(page.getByText('Add Model')).toBeVisible()
    await expect(page.getByPlaceholderText('gpt-4')).toBeVisible()
  })

  test('should add a new model', async ({ page }) => {
    // Mock API response
    await page.route('**/api/v1/models', async route => {
      if (route.request().method() === 'POST') {
        await route.fulfill({ json: { data: { id: '1', name: 'gpt-4' } } })
      } else {
        await route.fulfill({ json: { data: [] } })
      }
    })

    await page.click('button:has-text("Add Model")')
    await page.fill('input[placeholder="gpt-4"]', 'gpt-4')
    await page.selectOption('select', 'openai')
    await page.click('button:has-text("Add Model"):not(:has-text("Add Model"))')

    // Should close modal and show success
    await expect(page.getByText('Add Model')).not.toBeVisible()
  })

  test('should set API key for model', async ({ page }) => {
    // Mock models list
    await page.route('**/api/v1/models', async route => {
      await route.fulfill({
        json: {
          data: [
            { id: '1', name: 'gpt-4', provider: 'openai', api_key_set: false }
          ]
        }
      })
    })

    await page.reload()
    await page.waitForSelector('text=gpt-4', { timeout: 5000 })

    // Click set API key button
    await page.click('button:has-text("Set API Key")')
    
    await expect(page.getByText('Set API Key for gpt-4')).toBeVisible()
    await expect(page.getByPlaceholderText('sk-...')).toBeVisible()
  })

  test('should toggle API key visibility', async ({ page }) => {
    await page.route('**/api/v1/models', async route => {
      await route.fulfill({
        json: {
          data: [
            { id: '1', name: 'gpt-4', provider: 'openai', api_key_set: false }
          ]
        }
      })
    })

    await page.reload()
    await page.waitForSelector('text=gpt-4', { timeout: 5000 })
    await page.click('button:has-text("Set API Key")')

    const keyInput = page.getByPlaceholderText('sk-...')
    await keyInput.fill('sk-test123')

    // Should be password type initially
    await expect(keyInput).toHaveAttribute('type', 'password')

    // Toggle visibility
    await page.click('button[aria-label*="eye"]')
    await expect(keyInput).toHaveAttribute('type', 'text')
  })
})








