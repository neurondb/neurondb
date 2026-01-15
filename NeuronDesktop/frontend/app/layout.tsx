import type { Metadata } from 'next'
import AuthGuard from '@/components/AuthGuard'
import ThemeProvider from '@/components/ThemeProvider'
import ToastContainer from '@/components/Toast'
import AppWrapper from '@/components/AppWrapper'
import './globals.css'
import 'highlight.js/styles/github-dark.css'

export const metadata: Metadata = {
  title: 'NeuronDesktop - NeuronDB PostgreSQL AI Factory',
  description: 'NeuronDB PostgreSQL AI Factory - Unified interface for MCP servers, NeuronDB, and NeuronAgent',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>
        <ThemeProvider>
          <AuthGuard>
            <AppWrapper>
              {children}
            </AppWrapper>
            <ToastContainer />
          </AuthGuard>
        </ThemeProvider>
      </body>
    </html>
  )
}
