// Toast notification system for user-friendly error and success messages

export type ToastType = 'success' | 'error' | 'warning' | 'info'

export interface Toast {
  id: string
  type: ToastType
  message: string
  duration?: number
  timestamp: number
}

class ToastManager {
  private toasts: Toast[] = []
  private listeners: Set<(toasts: Toast[]) => void> = new Set()
  private idCounter = 0

  subscribe(listener: (toasts: Toast[]) => void) {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }

  private notify() {
    this.listeners.forEach(listener => listener([...this.toasts]))
  }

  show(type: ToastType, message: string, duration: number = 5000) {
    const id = `toast-${++this.idCounter}`
    const toast: Toast = {
      id,
      type,
      message,
      duration,
      timestamp: Date.now(),
    }

    this.toasts.push(toast)
    this.notify()

    if (duration > 0) {
      setTimeout(() => {
        this.remove(id)
      }, duration)
    }

    return id
  }

  remove(id: string) {
    this.toasts = this.toasts.filter(t => t.id !== id)
    this.notify()
  }

  clear() {
    this.toasts = []
    this.notify()
  }

  success(message: string, duration?: number) {
    return this.show('success', message, duration)
  }

  error(message: string, duration?: number) {
    return this.show('error', message, duration || 7000) // Errors stay longer
  }

  warning(message: string, duration?: number) {
    return this.show('warning', message, duration)
  }

  info(message: string, duration?: number) {
    return this.show('info', message, duration)
  }
}

export const toastManager = new ToastManager()







