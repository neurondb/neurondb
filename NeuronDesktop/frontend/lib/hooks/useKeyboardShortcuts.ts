'use client'

import { useEffect } from 'react'
import { useHotkeys } from 'react-hotkeys-hook'

export interface KeyboardShortcut {
  keys: string
  callback: () => void
  description?: string
  preventDefault?: boolean
}

export function useKeyboardShortcuts(shortcuts: KeyboardShortcut[]) {
  shortcuts.forEach((shortcut) => {
    useHotkeys(
      shortcut.keys,
      (event) => {
        if (shortcut.preventDefault !== false) {
          event.preventDefault()
        }
        shortcut.callback()
      },
      {
        enableOnFormTags: ['INPUT', 'TEXTAREA', 'SELECT'],
      }
    )
  })
}

export const COMMON_SHORTCUTS = {
  SAVE: 'ctrl+s, cmd+s',
  NEW: 'ctrl+n, cmd+n',
  SEARCH: 'ctrl+k, cmd+k',
  CLOSE: 'escape',
  REFRESH: 'ctrl+r, cmd+r',
  FULLSCREEN: 'f11',
} as const





