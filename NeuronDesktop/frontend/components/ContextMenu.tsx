'use client'

import { useEffect, useRef, ReactNode } from 'react'
import { createPortal } from 'react-dom'

interface ContextMenuItem {
  label: string
  icon?: ReactNode
  action: () => void
  disabled?: boolean
  divider?: boolean
  danger?: boolean
}

interface ContextMenuProps {
  x: number
  y: number
  items: ContextMenuItem[]
  onClose: () => void
}

export default function ContextMenu({ x, y, items, onClose }: ContextMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        onClose()
      }
    }

    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose()
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    document.addEventListener('keydown', handleEscape)

    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
      document.removeEventListener('keydown', handleEscape)
    }
  }, [onClose])

  useEffect(() => {
    if (menuRef.current) {
      const rect = menuRef.current.getBoundingClientRect()
      const viewportWidth = window.innerWidth
      const viewportHeight = window.innerHeight

      let adjustedX = x
      let adjustedY = y

      if (x + rect.width > viewportWidth) {
        adjustedX = viewportWidth - rect.width - 8
      }
      if (y + rect.height > viewportHeight) {
        adjustedY = viewportHeight - rect.height - 8
      }

      menuRef.current.style.left = `${adjustedX}px`
      menuRef.current.style.top = `${adjustedY}px`
    }
  }, [x, y])

  const menu = (
    <div
      ref={menuRef}
      className="fixed z-50 min-w-[200px] bg-white dark:bg-slate-800 rounded-lg shadow-xl border border-slate-200 dark:border-slate-700 py-1 animate-scale-in"
      style={{ left: x, top: y }}
    >
      {items.map((item, index) => {
        if (item.divider) {
          return (
            <div
              key={`divider-${index}`}
              className="my-1 border-t border-slate-200 dark:border-slate-700"
            />
          )
        }

        return (
          <button
            key={index}
            onClick={() => {
              if (!item.disabled) {
                item.action()
                onClose()
              }
            }}
            disabled={item.disabled}
            className={`
              w-full px-4 py-2 text-left text-sm
              flex items-center gap-2
              transition-colors duration-150
              ${
                item.disabled
                  ? 'text-slate-400 dark:text-slate-600 cursor-not-allowed'
                  : item.danger
                  ? 'text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20'
                  : 'text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700'
              }
            `}
          >
            {item.icon && <span className="w-4 h-4">{item.icon}</span>}
            <span>{item.label}</span>
          </button>
        )
      })}
    </div>
  )

  return typeof window !== 'undefined' ? createPortal(menu, document.body) : null
}


