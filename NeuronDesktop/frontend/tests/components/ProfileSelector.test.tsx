import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import ProfileSelector from '@/components/ProfileSelector'

describe('ProfileSelector', () => {
  const mockProfiles = [
    { id: '1', name: 'Profile 1', is_default: true },
    { id: '2', name: 'Profile 2', is_default: false },
  ]

  it('renders profile options', () => {
    render(
      <ProfileSelector
        profiles={mockProfiles}
        selectedProfile="1"
        onSelect={vi.fn()}
      />
    )

    expect(screen.getByText('Profile 1')).toBeInTheDocument()
    expect(screen.getByText('Profile 2')).toBeInTheDocument()
  })

  it('calls onSelect when profile changes', () => {
    const onSelect = vi.fn()
    render(
      <ProfileSelector
        profiles={mockProfiles}
        selectedProfile="1"
        onSelect={onSelect}
      />
    )

    const select = screen.getByRole('combobox')
    select.value = '2'
    select.dispatchEvent(new Event('change'))

    expect(onSelect).toHaveBeenCalledWith('2')
  })

  it('shows default indicator', () => {
    render(
      <ProfileSelector
        profiles={mockProfiles}
        selectedProfile="1"
        onSelect={vi.fn()}
      />
    )

    expect(screen.getByText(/Profile 1.*Default/i)).toBeInTheDocument()
  })
})








