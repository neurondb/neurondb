import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import OnboardingWizard from '@/components/OnboardingWizard'
import { neurondbAPI } from '@/lib/api'

jest.mock('@/lib/api')
jest.mock('@/lib/errors', () => ({
  showSuccessToast: jest.fn(),
  showErrorToast: jest.fn(),
}))

describe('OnboardingWizard', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('renders first step (Database Connection)', () => {
    render(<OnboardingWizard />)
    expect(screen.getByRole('heading', { name: 'Database Connection' })).toBeInTheDocument()
    expect(screen.getByPlaceholderText('localhost')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('5433')).toBeInTheDocument()
  })

  it('validates database connection before proceeding', async () => {
    const mockTestConnection = jest.mocked(neurondbAPI.testConnection)
    mockTestConnection.mockResolvedValue({ data: { success: true } })

    render(<OnboardingWizard />)

    // Fill in database fields (use labels to disambiguate duplicate "neurondb" placeholders)
    fireEvent.change(screen.getByPlaceholderText('localhost'), { target: { value: 'test-host' } })
    fireEvent.change(screen.getByPlaceholderText('5433'), { target: { value: '5432' } })
    const dbInputs = screen.getAllByPlaceholderText('neurondb')
    fireEvent.change(dbInputs[0], { target: { value: 'testdb' } })
    fireEvent.change(dbInputs[1], { target: { value: 'testuser' } })
    fireEvent.change(screen.getByPlaceholderText('Enter password'), { target: { value: 'testpass' } })

    // Test connection
    fireEvent.click(screen.getByText('Test Connection'))

    await waitFor(() => {
      expect(mockTestConnection).toHaveBeenCalled()
    })
  })

  it('shows error when database connection fails', async () => {
    const mockTestConnection = jest.mocked(neurondbAPI.testConnection)
    mockTestConnection.mockRejectedValue(new Error('Connection failed'))

    render(<OnboardingWizard />)

    fireEvent.click(screen.getByText('Test Connection'))

    await waitFor(() => {
      expect(screen.getByText(/Connection failed/i)).toBeInTheDocument()
    })
  })

  it('enables Next after successful connection test', async () => {
    const mockTestConnection = jest.mocked(neurondbAPI.testConnection)
    mockTestConnection.mockResolvedValue({ data: { success: true } })

    render(<OnboardingWizard />)

    fireEvent.click(screen.getByText('Test Connection'))

    await waitFor(() => {
      expect(mockTestConnection).toHaveBeenCalled()
    })

    // Next button becomes enabled after connection succeeds
    const nextButton = screen.getByRole('button', { name: /Next/i })
    await waitFor(() => {
      expect(nextButton).not.toBeDisabled()
    })
  })
})










