'use client'

import { useState, useEffect, useRef } from 'react'

interface WorkflowNode {
  id: string
  type: 'agent' | 'tool' | 'http' | 'sql' | 'approval' | 'custom'
  label: string
  x?: number
  y?: number
  status?: 'pending' | 'running' | 'completed' | 'failed'
}

interface WorkflowEdge {
  from: string
  to: string
  condition?: string
}

interface WorkflowDAGProps {
  nodes: WorkflowNode[]
  edges: WorkflowEdge[]
  onNodeClick?: (node: WorkflowNode) => void
  onNodeAdd?: (type: string, x: number, y: number) => void
  editable?: boolean
}

export default function WorkflowDAG({
  nodes,
  edges,
  onNodeClick,
  onNodeAdd,
  editable = false,
}: WorkflowDAGProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [selectedNode, setSelectedNode] = useState<string | null>(null)

  const nodeColors: Record<string, string> = {
    agent: 'rgb(139, 92, 246)',
    tool: 'rgb(59, 130, 246)',
    http: 'rgb(34, 197, 94)',
    sql: 'rgb(249, 115, 22)',
    approval: 'rgb(236, 72, 153)',
    custom: 'rgb(107, 114, 128)',
  }

  const statusColors: Record<string, string> = {
    pending: 'rgb(156, 163, 175)',
    running: 'rgb(59, 130, 246)',
    completed: 'rgb(34, 197, 94)',
    failed: 'rgb(239, 68, 68)',
  }

  const handleNodeClick = (node: WorkflowNode) => {
    setSelectedNode(node.id)
    if (onNodeClick) {
      onNodeClick(node)
    }
  }

  /* Simple layout algorithm - arrange nodes in layers */
  const layoutNodes = (nodes: WorkflowNode[], edges: WorkflowEdge[]): WorkflowNode[] => {
    const nodeMap = new Map(nodes.map((n) => [n.id, n]))
    const layers: WorkflowNode[][] = []
    const visited = new Set<string>()

    /* Find root nodes (no incoming edges) */
    const rootNodes = nodes.filter(
      (node) => !edges.some((edge) => edge.to === node.id)
    )

    /* BFS to assign layers */
    let currentLayer = rootNodes
    let layerIndex = 0

    while (currentLayer.length > 0) {
      layers.push(currentLayer)
      currentLayer.forEach((node) => visited.add(node.id))

      const nextLayer: WorkflowNode[] = []
      currentLayer.forEach((node) => {
        edges
          .filter((edge) => edge.from === node.id)
          .forEach((edge) => {
            const targetNode = nodeMap.get(edge.to)
            if (targetNode && !visited.has(edge.to)) {
              nextLayer.push(targetNode)
              visited.add(edge.to)
            }
          })
      })

      currentLayer = nextLayer
      layerIndex++
    }

    /* Position nodes */
    const positionedNodes: WorkflowNode[] = []
    layers.forEach((layer, layerIdx) => {
      layer.forEach((node, nodeIdx) => {
        positionedNodes.push({
          ...node,
          x: layerIdx * 200 + 100,
          y: nodeIdx * 120 + 100,
        })
      })
    })

    return positionedNodes
  }

  const positionedNodes = layoutNodes(nodes, edges)

  return (
    <div className="w-full h-full border border-slate-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-800 overflow-auto">
      <svg
        ref={svgRef}
        width="100%"
        height="100%"
        viewBox="0 0 800 600"
        className="min-h-[600px]"
      >
        {/* Draw edges */}
        {edges.map((edge, idx) => {
          const fromNode = positionedNodes.find((n) => n.id === edge.from)
          const toNode = positionedNodes.find((n) => n.id === edge.to)
          if (!fromNode || !toNode || !fromNode.x || !fromNode.y || !toNode.x || !toNode.y) {
            return null
          }

          return (
            <line
              key={idx}
              x1={fromNode.x}
              y1={fromNode.y}
              x2={toNode.x}
              y2={toNode.y}
              stroke="rgb(156, 163, 175)"
              strokeWidth="2"
              markerEnd="url(#arrowhead)"
            />
          )
        })}

        {/* Arrow marker definition */}
        <defs>
          <marker
            id="arrowhead"
            markerWidth="10"
            markerHeight="10"
            refX="9"
            refY="3"
            orient="auto"
          >
            <polygon points="0 0, 10 3, 0 6" fill="rgb(156, 163, 175)" />
          </marker>
        </defs>

        {/* Draw nodes */}
        {positionedNodes.map((node) => {
          if (!node.x || !node.y) return null

          const isSelected = selectedNode === node.id
          const nodeColor = nodeColors[node.type] || nodeColors.custom
          const statusColor = node.status ? statusColors[node.status] : nodeColor

          return (
            <g key={node.id}>
              <circle
                cx={node.x}
                cy={node.y}
                r={30}
                fill={isSelected ? statusColor : nodeColor}
                stroke={isSelected ? 'rgb(59, 130, 246)' : 'transparent'}
                strokeWidth={isSelected ? 3 : 0}
                className="cursor-pointer"
                onClick={() => handleNodeClick(node)}
              />
              <text
                x={node.x}
                y={node.y + 50}
                textAnchor="middle"
                className="text-xs fill-slate-700 dark:fill-slate-300 pointer-events-none"
              >
                {node.label}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}
