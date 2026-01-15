'use client'

import { useEffect, useState } from 'react'
import { useRouter, usePathname } from 'next/navigation'
import { checkAuth } from '@/lib/auth'
import { SidebarProvider } from '@/contexts/SidebarContext'
import Sidebar from '@/components/Sidebar'
import SidebarToggle from '@/components/SidebarToggle'
import MainContent from '@/components/MainContent'
import TopMenu from '@/components/TopMenu'
import Footer from '@/components/Footer'

export default function AuthGuard({ children }: { children: React.ReactNode }) {
  const router = useRouter()
  const pathname = usePathname()
  const [isChecking, setIsChecking] = useState(true)

  useEffect(() => {
    const checkAuthStatus = async () => {
      const isLoginPage = pathname === '/login'
      const isSetupPage = pathname === '/setup'
      
      // Add timeout to prevent infinite loading
      const timeoutId = setTimeout(() => {
        console.warn('AuthGuard: Authentication check timed out, allowing access')
        setIsChecking(false)
      }, 10000) // 10 second max wait
      
      try {
        // Check authentication via API (cookie-based)
        const isAuthenticated = await checkAuth()
        
        clearTimeout(timeoutId)
        
        if (!isAuthenticated && !isLoginPage && !isSetupPage) {
          router.push('/login')
          return
        }
        
        if (isAuthenticated && isLoginPage) {
          router.push('/')
          return
        }
        
        setIsChecking(false)
      } catch (error) {
        clearTimeout(timeoutId)
        console.error('AuthGuard: Error during auth check', error)
        // On error, allow access to login/setup pages, otherwise redirect to login
        if (!isLoginPage && !isSetupPage) {
          router.push('/login')
        } else {
          setIsChecking(false)
        }
      }
    }
    
    checkAuthStatus()
  }, [pathname, router])

  if (isChecking) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 dark:from-slate-900 dark:via-slate-900 dark:to-slate-900">
        <div className="text-gray-700 dark:text-slate-300">Loading...</div>
      </div>
    )
  }

  const isLoginPage = pathname === '/login'
  const isSetupPage = pathname === '/setup'
  
  if (isLoginPage || isSetupPage) {
    return <>{children}</>
  }

  // All pages are full screen (ChatGPT-style) - clean, minimal layout with minimal top nav
  return (
    <SidebarProvider>
      <div className="h-screen w-screen overflow-hidden bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 dark:from-slate-900 dark:via-slate-900 dark:to-slate-900 flex flex-col">
        <TopMenu />
        <div className="flex-1 overflow-hidden flex">
          <Sidebar />
          <MainContent>
            {children}
          </MainContent>
        </div>
        <Footer />
      </div>
    </SidebarProvider>
  )
}

