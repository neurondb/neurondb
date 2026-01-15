'use client'

import { useSidebar } from '@/contexts/SidebarContext'
import { useEffect, useState, memo } from 'react'

export default memo(function MainContent({ children }: { children: React.ReactNode }) {
  const { isOpen } = useSidebar()
  const [isDesktop, setIsDesktop] = useState(false)

  useEffect(() => {
    const checkDesktop = () => {
      setIsDesktop(window.innerWidth >= 1024)
    }
    
    checkDesktop()
    window.addEventListener('resize', checkDesktop)
    return () => window.removeEventListener('resize', checkDesktop)
  }, [])

  return (
    <main
      className={`
        flex-1 overflow-auto bg-transparent dark:bg-slate-950/50 transition-all duration-300 ease-in-out page-enter
        ${isOpen && isDesktop ? 'lg:ml-64 xl:ml-72' : 'lg:ml-0'}
        min-h-[calc(100vh-3.5rem)] sm:min-h-[calc(100vh-3.5rem)]
      `}
    >
      <div className="w-full max-w-[1920px] mx-auto px-3 sm:px-4 md:px-6 lg:px-8 xl:px-10 2xl:px-12 3xl:px-16">
        <div className="w-full">
          {children}
        </div>
      </div>
    </main>
  )
})

