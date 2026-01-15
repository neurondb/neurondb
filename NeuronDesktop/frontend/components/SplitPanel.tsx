'use client'

import { useState, useRef, useEffect, ReactNode } from 'react'
import { ChevronLeftIcon, ChevronRightIcon } from '@heroicons/react/24/outline'

interface SplitPanelProps {
  left: ReactNode
  right: ReactNode
  defaultLeftWidth?: number
  minLeftWidth?: number
  minRightWidth?: number
  orientation?: 'horizontal' | 'vertical'
  className?: string
}

export default function SplitPanel({
  left,
  right,
  defaultLeftWidth = 50,
  minLeftWidth = 20,
  minRightWidth = 20,
  orientation = 'horizontal',
  className = '',
}: SplitPanelProps) {
  const [leftWidth, setLeftWidth] = useState(defaultLeftWidth)
  const [isDragging, setIsDragging] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const isVertical = orientation === 'vertical'

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging || !containerRef.current) return

      const container = containerRef.current
      const rect = container.getBoundingClientRect()
      
      if (isVertical) {
        const totalHeight = rect.height
        const newTopHeight = ((e.clientY - rect.top) / totalHeight) * 100
        const clampedHeight = Math.max(
          minLeftWidth,
          Math.min(100 - minRightWidth, newTopHeight)
        )
        setLeftWidth(clampedHeight)
      } else {
        const totalWidth = rect.width
        const newLeftWidth = ((e.clientX - rect.left) / totalWidth) * 100
        const clampedWidth = Math.max(
          minLeftWidth,
          Math.min(100 - minRightWidth, newLeftWidth)
        )
        setLeftWidth(clampedWidth)
      }
    }

    const handleMouseUp = () => {
      setIsDragging(false)
    }

    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
      document.body.style.cursor = isVertical ? 'ns-resize' : 'ew-resize'
      document.body.style.userSelect = 'none'
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
  }, [isDragging, isVertical, minLeftWidth, minRightWidth])

  const handleMouseDown = () => {
    setIsDragging(true)
  }

  const toggleLeft = () => {
    if (leftWidth < 5) {
      setLeftWidth(defaultLeftWidth)
    } else {
      setLeftWidth(0)
    }
  }

  return (
    <div
      ref={containerRef}
      className={`flex ${isVertical ? 'flex-col' : 'flex-row'} h-full w-full ${className}`}
    >
      {/* Left Panel */}
      <div
        className={`${isVertical ? 'h-full' : 'h-full'} transition-all duration-200 ease-out overflow-hidden ${
          leftWidth < 5 ? 'hidden' : ''
        }`}
        style={
          isVertical
            ? { height: `${leftWidth}%` }
            : { width: `${leftWidth}%` }
        }
      >
        <div className="h-full w-full overflow-auto">{left}</div>
      </div>

      {/* Resizer */}
      <div
        onMouseDown={handleMouseDown}
        className={`
          ${isVertical ? 'h-1 w-full cursor-ns-resize' : 'w-1 h-full cursor-ew-resize'}
          bg-slate-200 dark:bg-slate-700
          hover:bg-purple-400 dark:hover:bg-purple-600
          transition-colors duration-200
          flex items-center justify-center
          group relative
          ${isDragging ? 'bg-purple-500 dark:bg-purple-500' : ''}
        `}
      >
        <button
          onClick={toggleLeft}
          className={`
            absolute ${isVertical ? 'left-2' : 'top-2'}
            p-1 rounded-md
            bg-slate-300 dark:bg-slate-600
            hover:bg-purple-400 dark:hover:bg-purple-600
            opacity-0 group-hover:opacity-100
            transition-opacity duration-200
            z-10
          `}
          aria-label={leftWidth < 5 ? 'Show left panel' : 'Hide left panel'}
        >
          {isVertical ? (
            leftWidth < 5 ? (
              <ChevronRightIcon className="w-4 h-4 rotate-90" />
            ) : (
              <ChevronLeftIcon className="w-4 h-4 rotate-90" />
            )
          ) : leftWidth < 5 ? (
            <ChevronRightIcon className="w-4 h-4" />
          ) : (
            <ChevronLeftIcon className="w-4 h-4" />
          )}
        </button>
      </div>

      {/* Right Panel */}
      <div
        className={`${isVertical ? 'h-full' : 'h-full'} flex-1 transition-all duration-200 ease-out overflow-hidden`}
        style={
          isVertical
            ? { height: `${100 - leftWidth}%` }
            : { width: `${100 - leftWidth}%` }
        }
      >
        <div className="h-full w-full overflow-auto">{right}</div>
      </div>
    </div>
  )
}

