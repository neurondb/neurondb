import { render, screen, act } from '@testing-library/react'
import ProfileSelector from '@/components/ProfileSelector'

describe('ProfileSelector', () => {
  const mockProfiles = [
    { id: '1', name: 'Profile 1', is_default: true, neurondb_dsn: 'postgres://u@host1/db', created_at: '2024-01-01T00:00:00Z' },
    { id: '2', name: 'Profile 2', is_default: false, neurondb_dsn: 'postgres://u@host2/db', created_at: '2024-01-01T00:00:00Z' },
  ]

  it('renders selected profile and shows options when opened', async () => {
    render(
      <ProfileSelector
        profiles={mockProfiles}
        selectedProfile="1"
        onSelect={jest.fn()}
      />
    )
    expect(screen.getByText('Profile 1')).toBeInTheDocument()
    await act(async () => {
      screen.getByRole('button', { name: /Profile 1/i }).click()
    })
    expect(screen.getByText('Profile 2')).toBeInTheDocument()
  })

  it('calls onSelect when profile changes', async () => {
    const onSelect = jest.fn()
    render(
      <ProfileSelector
        profiles={mockProfiles}
        selectedProfile="1"
        onSelect={onSelect}
      />
    )
    await act(async () => {
      screen.getByRole('button', { name: /Profile 1/i }).click()
    })
    await act(async () => {
      screen.getByText('Profile 2').click()
    })
    expect(onSelect).toHaveBeenCalledWith('2')
  })

  it('shows selected profile name in button', () => {
    render(
      <ProfileSelector
        profiles={mockProfiles}
        selectedProfile="1"
        onSelect={jest.fn()}
      />
    )
    expect(screen.getByRole('button', { name: /Profile 1/i })).toBeInTheDocument()
  })
})










