'use client'

import React, { Component, ErrorInfo, ReactNode } from 'react'
import { logger } from '@/lib/logger'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
  errorInfo: ErrorInfo | null
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    }
  }

  static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      error,
      errorInfo: null,
    }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    logger.error('ErrorBoundary caught an error', error, { componentStack: errorInfo.componentStack })
    this.setState({
      error,
      errorInfo,
    })

    // Log to error reporting service if available
    if (typeof window !== 'undefined' && window.location.hostname !== 'localhost') {
      // In production, send to error reporting service
      // Example: Sentry.captureException(error, { contexts: { react: errorInfo } })
    }
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }

      return (
        <div className="min-h-screen flex items-center justify-center bg-slate-50 dark:bg-slate-900 p-4">
          <div className="max-w-2xl w-full bg-white dark:bg-slate-800 rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-bold text-red-600 dark:text-red-400 mb-4">
              Something went wrong
            </h2>
            <p className="text-gray-700 dark:text-slate-300 mb-4">
              An unexpected error occurred. Please refresh the page or contact support if the problem persists.
            </p>
            {this.state.error && (
              <details className="mt-4">
                <summary className="cursor-pointer text-sm text-gray-600 dark:text-slate-400 mb-2">
                  Error Details
                </summary>
                <pre className="bg-slate-100 dark:bg-slate-900 p-4 rounded text-xs overflow-auto max-h-64">
                  {this.state.error.toString()}
                  {this.state.errorInfo && (
                    <>
                      {'\n\n'}
                      {this.state.errorInfo.componentStack}
                    </>
                  )}
                </pre>
              </details>
            )}
            <button
              onClick={() => {
                this.setState({ hasError: false, error: null, errorInfo: null })
                window.location.reload()
              }}
              className="mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
            >
              Reload Page
            </button>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}

