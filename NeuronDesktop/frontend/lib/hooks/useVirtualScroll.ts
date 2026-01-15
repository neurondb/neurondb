'use client'

import { useState, useEffect, useRef, useMemo } from 'react'

interface UseVirtualScrollOptions {
  itemHeight: number
  containerHeight: number
  itemCount: number
  overscan?: number
}

export function useVirtualScroll({
  itemHeight,
  containerHeight,
  itemCount,
  overscan = 5,
}: UseVirtualScrollOptions) {
  const [scrollTop, setScrollTop] = useState(0)
  const containerRef = useRef<HTMLDivElement>(null)

  const totalHeight = itemCount * itemHeight
  const startIndex = Math.max(0, Math.floor(scrollTop / itemHeight) - overscan)
  const endIndex = Math.min(
    itemCount - 1,
    Math.ceil((scrollTop + containerHeight) / itemHeight) + overscan
  )
  const visibleItems = endIndex - startIndex + 1

  const visibleRange = useMemo(
    () => ({
      start: startIndex,
      end: endIndex,
      total: itemCount,
    }),
    [startIndex, endIndex, itemCount]
  )

  const offsetY = startIndex * itemHeight

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const handleScroll = () => {
      setScrollTop(container.scrollTop)
    }

    container.addEventListener('scroll', handleScroll, { passive: true })
    return () => container.removeEventListener('scroll', handleScroll)
  }, [])

  return {
    containerRef,
    visibleRange,
    offsetY,
    totalHeight,
    visibleItems,
  }
}


