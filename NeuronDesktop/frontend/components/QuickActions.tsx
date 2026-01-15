'use client'

import { useState } from 'react'
import { PlusIcon } from '@heroicons/react/24/outline'

interface QuickAction {
  id: string
  label: string
  icon: React.ReactNode
  action: () => void
  color?: string
}

interface QuickActionsProps {
  actions: QuickAction[]
  position?: 'bottom-right' | 'bottom-left' | 'top-right' | 'top-left'
}

export default function QuickActions({
  actions,
  position = 'bottom-right',
}: QuickActionsProps) {
  const [isOpen, setIsOpen] = useState(false)

  const positionClasses = {
    'bottom-right': 'bottom-6 right-6',
    'bottom-left': 'bottom-6 left-6',
    'top-right': 'top-6 right-6',
    'top-left': 'top-6 left-6',
  }

  return (
    <div className={`fixed ${positionClasses[position]} z-40`}>
      {/* Action Buttons */}
      {isOpen && (
        <div className="mb-3 flex flex-col-reverse gap-2 animate-fade-in-up">
          {actions.map((action, index) => (
            <button
              key={action.id}
              onClick={() => {
                action.action()
                setIsOpen(false)
              }}
              className={`
                w-12 h-12 rounded-full shadow-lg
                flex items-center justify-center
                transition-all duration-200
                hover:scale-110
                ${action.color || 'bg-purple-600 hover:bg-purple-700 text-white'}
                animate-fade-in-up
              `}
              style={{ animationDelay: `${index * 50}ms` }}
              title={action.label}
            >
              {action.icon}
            </button>
          ))}
        </div>
      )}

      {/* Toggle Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`
          w-14 h-14 rounded-full shadow-lg
          bg-gradient-to-r from-purple-600 to-indigo-600
          hover:from-purple-700 hover:to-indigo-700
          text-white
          flex items-center justify-center
          transition-all duration-200
          ${isOpen ? 'rotate-45' : 'rotate-0'}
          hover:scale-110
        `}
        aria-label="Quick actions"
      >
        <PlusIcon className="w-6 h-6" />
      </button>
    </div>
  )
}


