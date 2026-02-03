import { test, expect } from '@playwright/test'

test.describe('Setup / Factory Flow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/setup?new_user=true')
  })

  test('should display welcome step', async ({ page }) => {
    await expect(page.getByText('Welcome to NeuronDesktop')).toBeVisible()
    await expect(page.getByRole('button', { name: 'Get Started' })).toBeVisible()
  })

  test('should navigate from welcome to PostgreSQL configuration', async ({ page }) => {
    await page.getByRole('button', { name: 'Get Started' }).click()
    await expect(page.getByText('PostgreSQL Configuration')).toBeVisible()
    await expect(page.getByText('Configure your PostgreSQL connection')).toBeVisible()
  })

  test('should display PostgreSQL form and navigate to profile', async ({ page }) => {
    await page.getByRole('button', { name: 'Get Started' }).click()
    await expect(page.getByText('PostgreSQL Configuration')).toBeVisible()

    await page.getByPlaceholder('localhost').fill('localhost')
    await page.getByPlaceholder('5432').fill('5432')
    await page.getByPlaceholder('neurondb').first().fill('neurondb')
    await page.getByPlaceholder('Enter PostgreSQL password').fill('testpassword')

    await page.getByRole('button', { name: 'Next' }).click()
    await expect(page.getByRole('heading', { name: 'Create Profile' })).toBeVisible()
    await expect(page.getByText('Set up your connection profile')).toBeVisible()
  })

  test('should complete setup flow with mocked API', async ({ page }) => {
    await page.route('**/api/v1/profiles', async (route) => {
      if (route.request().method() === 'POST') {
        await route.fulfill({ json: { id: 'test-profile', name: 'Default' } })
      } else {
        await route.fulfill({ json: [] })
      }
    })
    await page.route('**/api/v1/factory/setup-state', async (route) => {
      await route.fulfill({ json: { setup_complete: true } })
    })

    await page.getByRole('button', { name: 'Get Started' }).click()
    await page.getByRole('button', { name: 'Next' }).click()
    await page.getByRole('button', { name: /Create Profile/ }).click()

    await expect(page.getByRole('heading', { name: 'Setup Complete!' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Go to Dashboard' })).toBeVisible()
  })

  test('should navigate back from PostgreSQL to welcome', async ({ page }) => {
    await page.getByRole('button', { name: 'Get Started' }).click()
    await expect(page.getByText('PostgreSQL Configuration')).toBeVisible()
    await page.getByRole('button', { name: 'Back' }).click()
    await expect(page.getByText('Welcome to NeuronDesktop')).toBeVisible()
  })
})
