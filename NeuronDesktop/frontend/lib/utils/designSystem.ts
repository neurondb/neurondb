/**
 * Design System Utilities
 * Provides consistent styling and theming utilities
 */

export const colors = {
  primary: {
    light: 'rgb(139, 92, 246)',
    dark: 'rgb(167, 139, 250)',
    hover: 'rgb(167, 139, 250)',
  },
  secondary: {
    light: 'rgb(99, 102, 241)',
    dark: 'rgb(129, 140, 248)',
  },
  success: {
    light: 'rgb(34, 197, 94)',
    dark: 'rgb(74, 222, 128)',
  },
  warning: {
    light: 'rgb(251, 191, 36)',
    dark: 'rgb(251, 191, 36)',
  },
  error: {
    light: 'rgb(239, 68, 68)',
    dark: 'rgb(248, 113, 113)',
  },
  info: {
    light: 'rgb(59, 130, 246)',
    dark: 'rgb(96, 165, 250)',
  },
} as const

export const spacing = {
  xs: '0.25rem',
  sm: '0.5rem',
  md: '1rem',
  lg: '1.5rem',
  xl: '2rem',
  '2xl': '3rem',
} as const

export const borderRadius = {
  sm: '0.25rem',
  md: '0.375rem',
  lg: '0.5rem',
  xl: '0.75rem',
  '2xl': '1rem',
  full: '9999px',
} as const

export const shadows = {
  sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
  md: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
  lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
  xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1)',
  '2xl': '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
} as const

export const transitions = {
  fast: '150ms',
  normal: '200ms',
  slow: '300ms',
} as const

export function getThemeColors(isDark: boolean) {
  return {
    background: isDark ? 'rgb(15, 23, 42)' : 'rgb(248, 250, 252)',
    card: isDark ? 'rgb(30, 41, 59)' : 'rgb(255, 255, 255)',
    border: isDark ? 'rgb(51, 65, 85)' : 'rgb(226, 232, 240)',
    text: {
      primary: isDark ? 'rgb(241, 245, 249)' : 'rgb(17, 24, 39)',
      secondary: isDark ? 'rgb(203, 213, 225)' : 'rgb(30, 41, 59)',
      muted: isDark ? 'rgb(148, 163, 184)' : 'rgb(71, 85, 105)',
    },
  }
}






