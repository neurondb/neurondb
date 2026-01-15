'use client'

import { getShortcutString, type KeyboardShortcut } from '@/lib/keyboard'

interface ShortcutHintProps {
  shortcut: KeyboardShortcut
  className?: string
}

export default function ShortcutHint({ shortcut, className = '' }: ShortcutHintProps) {
  return (
    <kbd
      className={`
        px-2 py-1 text-xs font-semibold
        text-slate-500 dark:text-slate-400
        bg-slate-100 dark:bg-slate-700
        border border-slate-300 dark:border-slate-600
        rounded
        ${className}
      `}
      title={shortcut.description}
    >
      {getShortcutString(shortcut)}
    </kbd>
  )
}


