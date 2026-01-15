'use client'

import { ReactNode } from 'react'
import { useVirtualScroll } from '@/lib/hooks/useVirtualScroll'

interface VirtualListProps<T> {
  items: T[]
  itemHeight: number
  containerHeight: number
  renderItem: (item: T, index: number) => ReactNode
  className?: string
}

export default function VirtualList<T>({
  items,
  itemHeight,
  containerHeight,
  renderItem,
  className = '',
}: VirtualListProps<T>) {
  const { containerRef, visibleRange, offsetY, totalHeight } = useVirtualScroll({
    itemHeight,
    containerHeight,
    itemCount: items.length,
  })

  const visibleItems = items.slice(visibleRange.start, visibleRange.end + 1)

  return (
    <div
      ref={containerRef}
      className={`overflow-auto ${className}`}
      style={{ height: containerHeight }}
    >
      <div style={{ height: totalHeight, position: 'relative' }}>
        <div style={{ transform: `translateY(${offsetY}px)` }}>
          {visibleItems.map((item, index) => (
            <div
              key={visibleRange.start + index}
              style={{ height: itemHeight }}
            >
              {renderItem(item, visibleRange.start + index)}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}


