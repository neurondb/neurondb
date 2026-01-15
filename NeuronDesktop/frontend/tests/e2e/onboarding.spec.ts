import { test, expect } from '@playwright/test'

test.describe('Onboarding Flow', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to onboarding page
    await page.goto('/onboarding')
  })

  test('should display onboarding wizard', async ({ page }) => {
    await expect(page.getByText('Welcome to NeuronDesktop')).toBeVisible()
    await expect(page.getByText('Database Connection')).toBeVisible()
  })

  test('should test database connection', async ({ page }) => {
    // Fill in database connection form
    await page.fill('input[placeholder="localhost"]', 'localhost')
    await page.fill('input[placeholder="5433"]', '5433')
    await page.fill('input[placeholder="neurondb"]', 'neurondb')
    await page.fill('input[type="password"]', 'testpassword')

    // Click test connection button
    await page.click('button:has-text("Test Connection")')

    // Wait for result (success or error)
    await page.waitForSelector('text=/Connection (successful|failed)/i', { timeout: 10000 })
  })

  test('should navigate through wizard steps', async ({ page }) => {
    // Step 1: Database Connection
    await expect(page.getByText('Database Connection')).toBeVisible()

    // Mock successful connection test
    await page.route('**/api/v1/neurondb/test-connection', async route => {
      await route.fulfill({ json: { success: true } })
    })

    await page.fill('input[placeholder="localhost"]', 'localhost')
    await page.fill('input[placeholder="5433"]', '5433')
    await page.fill('input[placeholder="neurondb"]', 'neurondb')
    await page.fill('input[type="password"]', 'testpassword')
    await page.click('button:has-text("Test Connection")')
    
    // Wait for success
    await page.waitForSelector('text=/Connection successful/i', { timeout: 5000 })

    // Proceed to next step
    await page.click('button:has-text("Next")')

    // Step 2: MCP Configuration
    await expect(page.getByText('MCP Configuration')).toBeVisible()
  })

  test('should complete onboarding flow', async ({ page }) => {
    // Mock all API calls
    await page.route('**/api/v1/neurondb/test-connection', async route => {
      await route.fulfill({ json: { success: true } })
    })
    
    await page.route('**/api/v1/agent/health', async route => {
      await route.fulfill({ json: { status: 'ok' } })
    })

    await page.route('**/api/v1/profiles', async route => {
      await route.fulfill({ json: { data: [] } })
    })

    // Complete step 1
    await page.fill('input[placeholder="localhost"]', 'localhost')
    await page.fill('input[placeholder="5433"]', '5433')
    await page.fill('input[placeholder="neurondb"]', 'neurondb')
    await page.fill('input[type="password"]', 'testpassword')
    await page.click('button:has-text("Test Connection")')
    await page.waitForSelector('text=/Connection successful/i', { timeout: 5000 })
    await page.click('button:has-text("Next")')

    // Step 2: Skip MCP (optional)
    await page.click('button:has-text("Next")')

    // Step 3: Skip Agent (optional)
    await page.click('button:has-text("Next")')

    // Step 4: Skip demo dataset
    await page.click('button:has-text("Next")')

    // Step 5: Complete
    await expect(page.getByText('Setup Complete!')).toBeVisible()
  })
})








