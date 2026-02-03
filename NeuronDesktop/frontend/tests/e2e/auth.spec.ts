import { test, expect } from '@playwright/test'

test.describe('Authentication', () => {
  test('should display login form', async ({ page }) => {
    await page.goto('/login')
    await expect(page.getByText('NeuronDesktop')).toBeVisible()
    await expect(page.getByText('Sign in to your account')).toBeVisible()
    await expect(page.getByLabel(/username/i)).toBeVisible()
    await expect(page.getByLabel(/password/i)).toBeVisible()
    await expect(page.getByRole('button', { name: /Sign In with Password/i })).toBeVisible()
  })

  test('should toggle to signup mode', async ({ page }) => {
    await page.goto('/login')
    await expect(page.getByText('Sign in to your account')).toBeVisible()
    await page.getByRole('button', { name: /Sign up/i }).click()
    await expect(page.getByText('Create your account')).toBeVisible()
  })

  test('should login and redirect to home with mocked API', async ({ page }) => {
    await page.route('**/auth/login**', async (route) => {
      await route.fulfill({
        status: 200,
        json: { token: 'test-token', user_id: 'u1', username: 'testuser', profile_id: 'p1' },
      })
    })
    await page.route('**/auth/me**', async (route) => {
      await route.fulfill({ status: 200, json: { id: 'u1', username: 'testuser' } })
    })
    await page.route('**/factory/setup-state**', async (route) => {
      await route.fulfill({ json: { setup_complete: true } })
    })

    await page.goto('/login')
    await page.getByLabel(/username/i).fill('testuser')
    await page.getByLabel(/password/i).fill('password123')
    await page.getByRole('button', { name: /Sign In with Password/i }).click()

    await expect(page).toHaveURL(/\/(\?.*)?$/, { timeout: 10000 })
    await expect(page.getByText(/NeuronDesktop|Features/)).toBeVisible()
  })

  test('should show error on failed login', async ({ page }) => {
    await page.route('**/auth/login**', async (route) => {
      await route.fulfill({ status: 401, json: { error: 'Invalid credentials' } })
    })

    await page.goto('/login')
    await page.getByLabel(/username/i).fill('wronguser')
    await page.getByLabel(/password/i).fill('wrongpass')
    await page.getByRole('button', { name: /Sign In with Password/i }).click()

    await expect(page.getByText(/invalid|error|failed|credentials/i)).toBeVisible({ timeout: 5000 })
  })
})
