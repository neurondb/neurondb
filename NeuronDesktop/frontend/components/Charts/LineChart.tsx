'use client'

import { useState } from 'react'
import { LineChart as RechartsLineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Brush, ReferenceLine } from 'recharts'
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
  interactive?: boolean
  brush?: boolean
  referenceLines?: Array<{
    y?: number
    x?: string
    label?: string
    stroke?: string
  }>
}

export default function LineChart({
  data,
  dataKey,
  lines,
  xAxisKey = 'time',
  height = 300,
  showGrid = true,
  showLegend = true,
  interactive = true,
  brush = false,
  referenceLines = [],
}: LineChartProps) {
  const { theme } = useTheme()
  const isDark = theme === 'dark'
  const [activeIndex, setActiveIndex] = useState<number | null>(null)

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
        {referenceLines.map((refLine, idx) => (
          <ReferenceLine
            key={idx}
            y={refLine.y}
            x={refLine.x}
            label={refLine.label}
            stroke={refLine.stroke || (isDark ? 'rgb(148, 163, 184)' : 'rgb(71, 85, 105)')}
            strokeDasharray="3 3"
          />
        ))}
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
            isAnimationActive={interactive}
            animationDuration={300}
          />
        ))}
        {brush && (
          <Brush
            dataKey={xAxisKey}
            height={30}
            stroke={isDark ? 'rgb(148, 163, 184)' : 'rgb(71, 85, 105)'}
          />
        )}
      </RechartsLineChart>
    </ResponsiveContainer>
  )
}





