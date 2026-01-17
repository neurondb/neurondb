/**
 * Centralized logging utility for NeuronDesktop
 * 
 * Provides consistent logging interface that:
 * - Only logs in development mode by default
 * - Supports different log levels
 * - Can be extended to send logs to external services
 */

type LogLevel = 'debug' | 'info' | 'warn' | 'error'

interface LogContext {
  [key: string]: unknown
}

class Logger {
  private isDevelopment: boolean
  private enabled: boolean

  constructor() {
    this.isDevelopment = process.env.NODE_ENV === 'development'
    this.enabled = true // Can be controlled via config if needed
  }

  /**
   * Enable or disable logging
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled
  }

  /**
   * Check if logging is enabled for a given level
   */
  private shouldLog(level: LogLevel): boolean {
    if (!this.enabled) return false
    
    // In production, only log errors and warnings
    if (!this.isDevelopment) {
      return level === 'error' || level === 'warn'
    }
    
    return true
  }

  /**
   * Format log message with context
   */
  private formatMessage(message: string, context?: LogContext): string {
    if (!context || Object.keys(context).length === 0) {
      return message
    }
    
    try {
      return `${message} ${JSON.stringify(context)}`
    } catch {
      return `${message} [Context serialization failed]`
    }
  }

  /**
   * Debug level logging (development only)
   */
  debug(message: string, context?: LogContext): void {
    if (this.shouldLog('debug')) {
      console.debug(`[DEBUG] ${message}`, context || '')
    }
  }

  /**
   * Info level logging (development only)
   */
  info(message: string, context?: LogContext): void {
    if (this.shouldLog('info')) {
      console.info(`[INFO] ${message}`, context || '')
    }
  }

  /**
   * Warning level logging (always enabled)
   */
  warn(message: string, context?: LogContext): void {
    if (this.shouldLog('warn')) {
      console.warn(`[WARN] ${message}`, context || '')
    }
  }

  /**
   * Error level logging (always enabled)
   */
  error(message: string, error?: Error | unknown, context?: LogContext): void {
    if (this.shouldLog('error')) {
      const errorDetails = error instanceof Error 
        ? { message: error.message, stack: error.stack, name: error.name }
        : error
      
      console.error(`[ERROR] ${message}`, {
        ...context,
        error: errorDetails
      })
    }
  }

  /**
   * Log with explicit level control
   */
  log(level: LogLevel, message: string, error?: Error | unknown, context?: LogContext): void {
    switch (level) {
      case 'debug':
        this.debug(message, context)
        break
      case 'info':
        this.info(message, context)
        break
      case 'warn':
        this.warn(message, context)
        break
      case 'error':
        this.error(message, error, context)
        break
    }
  }
}

// Export singleton instance
export const logger = new Logger()

// Export convenience functions
export const logDebug = (message: string, context?: LogContext) => logger.debug(message, context)
export const logInfo = (message: string, context?: LogContext) => logger.info(message, context)
export const logWarn = (message: string, context?: LogContext) => logger.warn(message, context)
export const logError = (message: string, error?: Error | unknown, context?: LogContext) => 
  logger.error(message, error, context)

// Export for testing
export { Logger }
