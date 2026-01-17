// Authentication utilities for NeuronDesktop (Cookie-based sessions with JWT fallback)

import { logger } from './logger'

const AUTH_TOKEN_KEY = 'neurondesk_auth_token'

// Check if user is authenticated by calling /auth/me endpoint
export async function checkAuth(): Promise<boolean> {
  if (typeof window === 'undefined') return false
  
  try {
    const apiUrl = `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8081/api/v1'}/auth/me`
    logger.debug('Checking authentication', { apiUrl })
    
    const token = getAuthToken()
    
    // Add timeout to prevent hanging
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 5000) // 5 second timeout
    
    try {
      const response = await fetch(apiUrl, {
        method: 'GET',
        credentials: 'include', // Send cookies
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}), // JWT if present; otherwise cookie-only (OIDC mode)
        },
      })
      
      clearTimeout(timeoutId)
      
      logger.debug('Auth check response', { status: response.status, statusText: response.statusText })
      
      if (!response.ok) {
        const errorText = await response.text()
        logger.error('Authentication failed', undefined, { status: response.status, error: errorText })
        return false
      }
      
      const data = await response.json()
      logger.debug('Authentication successful', { userId: data.id, username: data.username })
      return true
    } catch (fetchError: unknown) {
      clearTimeout(timeoutId)
      if (fetchError instanceof Error && fetchError.name === 'AbortError') {
        logger.error('Request timeout - API server may not be responding', fetchError)
      }
      throw fetchError
    }
  } catch (error) {
    logger.error('Error checking authentication', error)
    return false
  }
}

// Legacy JWT token functions (for backward compatibility)
export function getAuthToken(): string | null {
  if (typeof window === 'undefined') return null
  return localStorage.getItem(AUTH_TOKEN_KEY)
}

export function setAuthToken(token: string): void {
  if (typeof window === 'undefined') return
  localStorage.setItem(AUTH_TOKEN_KEY, token)
}

export function removeAuthToken(): void {
  if (typeof window === 'undefined') return
  localStorage.removeItem(AUTH_TOKEN_KEY)
}

export function getAuthHeaders(): Record<string, string> {
  // For cookie-based auth, no headers needed
  // Keep for JWT fallback
  const token = getAuthToken()
  if (!token) return {}
  
  return {
    'Authorization': `Bearer ${token}`,
  }
}

// Legacy API key functions for backwards compatibility (deprecated)
export function getAPIKey(): string | null {
  return getAuthToken()
}

export function setAPIKey(key: string): void {
  setAuthToken(key)
}

export function removeAPIKey(): void {
  removeAuthToken()
}

