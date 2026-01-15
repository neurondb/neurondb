'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { findShortcut, shortcuts } from '@/lib/keyboard'

interface UseKeyboardShortcutsOptions {
  onCommandPalette?: () => void
  onGlobalSearch?: () => void
  enabled?: boolean
}

export function useKeyboardShortcuts({
  onCommandPalette,
  onGlobalSearch,
  enabled = true,
}: UseKeyboardShortcutsOptions = {}) {
  const router = useRouter()

  useEffect(() => {
    if (!enabled) return

    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger shortcuts when typing in inputs
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        (e.target as HTMLElement)?.isContentEditable
      ) {
        return
      }

      const shortcut = findShortcut(e)

      if (!shortcut) return

      e.preventDefault()

      switch (shortcut.action) {
        case 'command-palette':
          onCommandPalette?.()
          break
        case 'toggle-sidebar':
          // Toggle sidebar logic would go here
          break
        case 'view-dashboard':
          router.push('/dashboard')
          break
        case 'view-neurondb':
          router.push('/neurondb')
          break
        case 'view-mcp':
          router.push('/mcp')
          break
        case 'view-agents':
          router.push('/agents')
          break
        case 'save':
          // Save logic
          break
        case 'export':
          // Export logic
          break
        case 'find':
          onGlobalSearch?.()
          break
        default:
          break
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [enabled, onCommandPalette, onGlobalSearch, router])
}
