'use client'

interface DistanceMetricSelectorProps {
  value: string
  onChange: (value: string) => void
}

export default function DistanceMetricSelector({ value, onChange }: DistanceMetricSelectorProps) {
  const metrics = [
    { value: 'l2', label: 'L2 (Euclidean)', operator: '<->' },
    { value: 'cosine', label: 'Cosine', operator: '<=>' },
    { value: 'inner_product', label: 'Inner Product', operator: '<#>' },
    { value: 'l1', label: 'L1 (Manhattan)', operator: '<+>' },
    { value: 'hamming', label: 'Hamming', operator: '<~>' },
    { value: 'chebyshev', label: 'Chebyshev', operator: '<^>' },
    { value: 'minkowski', label: 'Minkowski', operator: '<*>' },
  ]
  
  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium">Distance Metric</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="input"
      >
        {metrics.map((metric) => (
          <option key={metric.value} value={metric.value}>
            {metric.label} ({metric.operator})
          </option>
        ))}
      </select>
    </div>
  )
}

