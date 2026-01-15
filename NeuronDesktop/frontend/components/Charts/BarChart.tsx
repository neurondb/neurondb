'use client'

import { BarChart as RechartsBarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { useTheme } from '@/contexts/ThemeContext'

interface BarChartProps {
  data: any[]
  bars: Array<{
    key: string
    name: string
    color?: string
  }>
  xAxisKey?: string
  height?: number
  showGrid?: boolean
  showLegend?: boolean
}

export default function BarChart({
  data,
  bars,
  xAxisKey = 'name',
  height = 300,
  showGrid = true,
  showLegend = true,
}: BarChartProps) {
  const { theme } = useTheme()
  const isDark = theme === 'dark'

  return (
    <ResponsiveContainer width="100%" height={height}>
      <RechartsBarChart
        data={data}
        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
      >
        {showGrid && (
          <CartesianGrid
            strokeDasharray="3 3"
            stroke={isDark ? 'rgb(51, 65, 85)' : 'rgb(226, 232, 240)'}
          />
        )}
        <XAxis
          dataKey={xAxisKey}
          stroke={isDark ? 'rgb(148, 163, 184)' : 'rgb(71, 85, 105)'}
          style={{ fontSize: '12px' }}
        />
        <YAxis
          stroke={isDark ? 'rgb(148, 163, 184)' : 'rgb(71, 85, 105)'}
          style={{ fontSize: '12px' }}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: isDark ? 'rgb(30, 41, 59)' : 'rgb(255, 255, 255)',
            border: `1px solid ${isDark ? 'rgb(51, 65, 85)' : 'rgb(226, 232, 240)'}`,
            borderRadius: '6px',
            color: isDark ? 'rgb(241, 245, 249)' : 'rgb(17, 24, 39)',
          }}
        />
        {showLegend && (
          <Legend
            wrapperStyle={{ fontSize: '12px' }}
          />
        )}
        {bars.map((bar) => (
          <Bar
            key={bar.key}
            dataKey={bar.key}
            name={bar.name}
            fill={bar.color || 'rgb(139, 92, 246)'}
            radius={[4, 4, 0, 0]}
          />
        ))}
      </RechartsBarChart>
    </ResponsiveContainer>
  )
}






