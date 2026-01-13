'use client'

import { LineChart as RechartsLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { useTheme } from '@/contexts/ThemeContext'

interface LineChartProps {
  data: any[]
  dataKey: string
  lines: Array<{
    key: string
    name: string
    color?: string
    strokeWidth?: number
  }>
  xAxisKey?: string
  height?: number
  showGrid?: boolean
  showLegend?: boolean
}

export default function LineChart({
  data,
  dataKey,
  lines,
  xAxisKey = 'time',
  height = 300,
  showGrid = true,
  showLegend = true,
}: LineChartProps) {
  const { theme } = useTheme()
  const isDark = theme === 'dark'

  return (
    <ResponsiveContainer width="100%" height={height}>
      <RechartsLineChart
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
            iconType="line"
          />
        )}
        {lines.map((line) => (
          <Line
            key={line.key}
            type="monotone"
            dataKey={line.key}
            name={line.name}
            stroke={line.color || 'rgb(139, 92, 246)'}
            strokeWidth={line.strokeWidth || 2}
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
          />
        ))}
      </RechartsLineChart>
    </ResponsiveContainer>
  )
}





