'use client'

import { useState } from 'react'
import CommandPalette from './CommandPalette'
import GlobalSearch from './GlobalSearch'
import HelpCenter from './HelpCenter'
import { useKeyboardShortcuts } from '@/lib/hooks/useKeyboardShortcuts'

interface AppWrapperProps {
  children: React.ReactNode
}

export default function AppWrapper({ children }: AppWrapperProps) {
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false)
  const [globalSearchOpen, setGlobalSearchOpen] = useState(false)
  const [helpCenterOpen, setHelpCenterOpen] = useState(false)

  useKeyboardShortcuts({
    onCommandPalette: () => setCommandPaletteOpen(true),
    onGlobalSearch: () => setGlobalSearchOpen(true),
  })

  return (
    <>
      {children}
      <CommandPalette
        isOpen={commandPaletteOpen}
        onClose={() => setCommandPaletteOpen(false)}
      />
      <GlobalSearch
        isOpen={globalSearchOpen}
        onClose={() => setGlobalSearchOpen(false)}
      />
      <HelpCenter
        isOpen={helpCenterOpen}
        onClose={() => setHelpCenterOpen(false)}
      />
    </>
  )
}


