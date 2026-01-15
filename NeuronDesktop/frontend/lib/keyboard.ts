export interface KeyboardShortcut {
  key: string
  ctrl?: boolean
  meta?: boolean
  shift?: boolean
  alt?: boolean
  action: string
  description: string
  category?: string
}

export const shortcuts: KeyboardShortcut[] = [
  // Navigation
  {
    key: 'k',
    meta: true,
    action: 'command-palette',
    description: 'Open command palette',
    category: 'Navigation',
  },
  {
    key: 'p',
    meta: true,
    action: 'quick-switch',
    description: 'Quick switch',
    category: 'Navigation',
  },
  {
    key: 'g',
    meta: true,
    action: 'go-to',
    description: 'Go to...',
    category: 'Navigation',
  },
  {
    key: 'b',
    meta: true,
    action: 'toggle-sidebar',
    description: 'Toggle sidebar',
    category: 'Navigation',
  },
  
  // Actions
  {
    key: 'n',
    meta: true,
    action: 'new',
    description: 'New item',
    category: 'Actions',
  },
  {
    key: 's',
    meta: true,
    action: 'save',
    description: 'Save',
    category: 'Actions',
  },
  {
    key: 'e',
    meta: true,
    action: 'export',
    description: 'Export',
    category: 'Actions',
  },
  {
    key: 'f',
    meta: true,
    action: 'find',
    description: 'Find',
    category: 'Actions',
  },
  
  // Editor
  {
    key: 'Enter',
    ctrl: true,
    action: 'execute',
    description: 'Execute query/command',
    category: 'Editor',
  },
  {
    key: 'Escape',
    action: 'cancel',
    description: 'Cancel/Close',
    category: 'Editor',
  },
  
  // View
  {
    key: '1',
    meta: true,
    action: 'view-dashboard',
    description: 'Go to Dashboard',
    category: 'View',
  },
  {
    key: '2',
    meta: true,
    action: 'view-neurondb',
    description: 'Go to NeuronDB',
    category: 'View',
  },
  {
    key: '3',
    meta: true,
    action: 'view-mcp',
    description: 'Go to MCP Console',
    category: 'View',
  },
  {
    key: '4',
    meta: true,
    action: 'view-agents',
    description: 'Go to Agents',
    category: 'View',
  },
]

export function getShortcutString(shortcut: KeyboardShortcut): string {
  const parts: string[] = []
  if (shortcut.meta) parts.push('⌘')
  if (shortcut.ctrl) parts.push('Ctrl')
  if (shortcut.alt) parts.push('Alt')
  if (shortcut.shift) parts.push('Shift')
  parts.push(shortcut.key)
  return parts.join(' + ')
}

export function matchesShortcut(
  event: KeyboardEvent,
  shortcut: KeyboardShortcut
): boolean {
  const metaMatch = shortcut.meta ? event.metaKey || event.ctrlKey : !event.metaKey && !event.ctrlKey
  const ctrlMatch = shortcut.ctrl ? event.ctrlKey : !event.ctrlKey
  const altMatch = shortcut.alt ? event.altKey : !event.altKey
  const shiftMatch = shortcut.shift ? event.shiftKey : !event.shiftKey
  const keyMatch = event.key.toLowerCase() === shortcut.key.toLowerCase()

  return metaMatch && ctrlMatch && altMatch && shiftMatch && keyMatch
}

export function findShortcut(event: KeyboardEvent): KeyboardShortcut | null {
  return shortcuts.find((shortcut) => matchesShortcut(event, shortcut)) || null
}

