'use client'

interface VectorTypeSelectorProps {
  value: string
  onChange: (value: string) => void
}

export default function VectorTypeSelector({ value, onChange }: VectorTypeSelectorProps) {
  const types = [
    { value: 'vector', label: 'vector', description: 'Standard 32-bit floating-point vectors' },
    { value: 'vectorp', label: 'vectorp', description: 'Packed optimized storage' },
    { value: 'vecmap', label: 'vecmap', description: 'Sparse vectors (non-zero only)' },
    { value: 'vgraph', label: 'vgraph', description: 'Graph-based structures' },
    { value: 'rtext', label: 'rtext', description: 'Retrieval-optimized text' },
  ]
  
  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium">Vector Type</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="input"
      >
        {types.map((type) => (
          <option key={type.value} value={type.value}>
            {type.label} - {type.description}
          </option>
        ))}
      </select>
    </div>
  )
}

