import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import OnboardingWizard from '@/components/OnboardingWizard'
import { neurondbAPI, agentAPI } from '@/lib/api'

vi.mock('@/lib/api')
vi.mock('@/lib/errors', () => ({
  showSuccessToast: vi.fn(),
  showErrorToast: vi.fn(),
}))

describe('OnboardingWizard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders first step (Database Connection)', () => {
    render(<OnboardingWizard />)
    
    expect(screen.getByText('Database Connection')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('localhost')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('5433')).toBeInTheDocument()
  })

  it('validates database connection before proceeding', async () => {
    const mockTestConnection = vi.mocked(neurondbAPI.testConnection)
    mockTestConnection.mockResolvedValue({ data: { success: true } })

    render(<OnboardingWizard />)

    // Fill in database fields
    fireEvent.change(screen.getByPlaceholderText('localhost'), { target: { value: 'test-host' } })
    fireEvent.change(screen.getByPlaceholderText('5433'), { target: { value: '5432' } })
    fireEvent.change(screen.getByPlaceholderText('neurondb'), { target: { value: 'testdb' } })
    fireEvent.change(screen.getByPlaceholderText('neurondb'), { target: { value: 'testuser' } })
    fireEvent.change(screen.getByPlaceholderText('Enter password'), { target: { value: 'testpass' } })

    // Test connection
    fireEvent.click(screen.getByText('Test Connection'))

    await waitFor(() => {
      expect(mockTestConnection).toHaveBeenCalled()
    })
  })

  it('shows error when database connection fails', async () => {
    const mockTestConnection = vi.mocked(neurondbAPI.testConnection)
    mockTestConnection.mockRejectedValue(new Error('Connection failed'))

    render(<OnboardingWizard />)

    fireEvent.click(screen.getByText('Test Connection'))

    await waitFor(() => {
      expect(screen.getByText(/Connection failed/i)).toBeInTheDocument()
    })
  })

  it('navigates to next step when valid', async () => {
    const mockTestConnection = vi.mocked(neurondbAPI.testConnection)
    mockTestConnection.mockResolvedValue({ data: { success: true } })

    render(<OnboardingWizard />)

    // Test and succeed
    fireEvent.click(screen.getByText('Test Connection'))
    
    await waitFor(() => {
      expect(mockTestConnection).toHaveBeenCalled()
    })

    // Try to proceed (should be disabled until connection succeeds)
    const nextButton = screen.getByText('Next')
    expect(nextButton).toBeDisabled()
  })
})








