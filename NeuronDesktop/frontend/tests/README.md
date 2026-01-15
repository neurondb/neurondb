# Frontend Testing Guide

## Overview

This directory contains tests for the NeuronDesktop frontend, including unit tests, component tests, and end-to-end tests.

## Test Structure

```
tests/
├── setup.ts                    # Test setup and mocks
├── components/                 # Component unit tests
│   ├── ProfileSelector.test.tsx
│   └── OnboardingWizard.test.tsx
└── e2e/                        # End-to-end tests
    ├── onboarding.spec.ts
    └── models.spec.ts
```

## Running Tests

### Unit/Component Tests (Vitest)

```bash
# Run all tests
npm run test

# Run in watch mode
npm run test:watch

# Run with coverage
npm run test:coverage

# Run specific test file
npm run test ProfileSelector
```

### End-to-End Tests (Playwright)

```bash
# Run all E2E tests
npm run test:e2e

# Run in UI mode
npm run test:e2e:ui

# Run specific test file
npm run test:e2e tests/e2e/onboarding.spec.ts

# Run in headed mode
npm run test:e2e -- --headed
```

## Writing Tests

### Unit Tests

Use Vitest with React Testing Library for component tests:

```typescript
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import MyComponent from '@/components/MyComponent'

describe('MyComponent', () => {
  it('renders correctly', () => {
    render(<MyComponent />)
    expect(screen.getByText('Hello')).toBeInTheDocument()
  })
})
```

### E2E Tests

Use Playwright for end-to-end tests:

```typescript
import { test, expect } from '@playwright/test'

test('user can login', async ({ page }) => {
  await page.goto('/login')
  await page.fill('input[name="email"]', 'test@example.com')
  await page.fill('input[name="password"]', 'password')
  await page.click('button:has-text("Login")')
  await expect(page).toHaveURL('/')
})
```

## Test Coverage Goals

- **Unit Tests**: 80%+ coverage for components
- **E2E Tests**: Cover all critical user flows
- **Integration Tests**: Test API interactions

## Best Practices

1. **Test User Behavior**: Test what users see and do, not implementation details
2. **Use Data Attributes**: Add `data-testid` for stable selectors
3. **Mock External APIs**: Mock API calls in unit tests
4. **Test Error States**: Test error handling and edge cases
5. **Keep Tests Fast**: Unit tests should run in < 1 second
6. **Isolate Tests**: Each test should be independent

## Continuous Integration

Tests run automatically on:
- Pull requests
- Pushes to main branch
- Nightly builds

## Debugging Tests

### Vitest
```bash
# Run with debug output
npm run test -- --reporter=verbose

# Run single test
npm run test -- ProfileSelector.test.tsx
```

### Playwright
```bash
# Run with debug mode
npm run test:e2e -- --debug

# Run with trace viewer
npm run test:e2e -- --trace on
```

## Common Issues

### Tests Failing Due to Timing
- Use `waitFor` for async operations
- Increase timeout if needed
- Use `findBy*` queries for async elements

### Mock Not Working
- Check mock is in `setup.ts` or test file
- Verify mock path matches import path
- Clear mocks between tests with `vi.clearAllMocks()`

### E2E Tests Flaky
- Add explicit waits
- Use `waitForSelector` instead of `waitForTimeout`
- Check for race conditions








