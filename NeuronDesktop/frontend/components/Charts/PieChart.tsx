'use client'

import { PieChart as RechartsPieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { useTheme } from '@/contexts/ThemeContext'

interface PieChartProps {
  data: Array<{ name: string; value: number }>
  colors?: string[]
  height?: number
  showLegend?: boolean
  innerRadius?: number
  outerRadius?: number
}

const DEFAULT_COLORS = [
  'rgb(139, 92, 246)', // purple
  'rgb(99, 102, 241)', // indigo
  'rgb(59, 130, 246)', // blue
  'rgb(34, 197, 94)',  // green
  'rgb(251, 191, 36)', // yellow
  'rgb(239, 68, 68)',  // red
]

export default function PieChart({
  data,
  colors = DEFAULT_COLORS,
  height = 300,
  showLegend = true,
  innerRadius = 0,
  outerRadius = 80,
}: PieChartProps) {
  const { theme } = useTheme()
  const isDark = theme === 'dark'

  return (
    <ResponsiveContainer width="100%" height={height}>
      <RechartsPieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          labelLine={false}
          label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
          outerRadius={outerRadius}
          innerRadius={innerRadius}
          fill="#8884d8"
          dataKey="value"
        >
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
          ))}
        </Pie>
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
      </RechartsPieChart>
    </ResponsiveContainer>
  )
}





