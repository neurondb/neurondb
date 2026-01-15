export interface SearchResult {
  id: string
  type: 'mcp-tool' | 'neurondb-collection' | 'agent' | 'log' | 'page'
  title: string
  description?: string
  url?: string
  metadata?: Record<string, any>
}

export interface SearchIndex {
  tools: SearchResult[]
  collections: SearchResult[]
  agents: SearchResult[]
  pages: SearchResult[]
}

let searchIndex: SearchIndex = {
  tools: [],
  collections: [],
  agents: [],
  pages: [],
}

export function indexItem(item: SearchResult) {
  switch (item.type) {
    case 'mcp-tool':
      searchIndex.tools.push(item)
      break
    case 'neurondb-collection':
      searchIndex.collections.push(item)
      break
    case 'agent':
      searchIndex.agents.push(item)
      break
    case 'page':
      searchIndex.pages.push(item)
      break
  }
}

export function removeFromIndex(id: string, type: SearchResult['type']) {
  switch (type) {
    case 'mcp-tool':
      searchIndex.tools = searchIndex.tools.filter((item) => item.id !== id)
      break
    case 'neurondb-collection':
      searchIndex.collections = searchIndex.collections.filter((item) => item.id !== id)
      break
    case 'agent':
      searchIndex.agents = searchIndex.agents.filter((item) => item.id !== id)
      break
    case 'page':
      searchIndex.pages = searchIndex.pages.filter((item) => item.id !== id)
      break
  }
}

export function search(query: string): SearchResult[] {
  const queryLower = query.toLowerCase()
  const results: SearchResult[] = []

  const searchInArray = (items: SearchResult[]) => {
    return items.filter((item) => {
      const titleMatch = item.title.toLowerCase().includes(queryLower)
      const descMatch = item.description?.toLowerCase().includes(queryLower)
      return titleMatch || descMatch
    })
  }

  results.push(...searchInArray(searchIndex.tools))
  results.push(...searchInArray(searchIndex.collections))
  results.push(...searchInArray(searchIndex.agents))
  results.push(...searchInArray(searchIndex.pages))

  return results
}

export function clearIndex() {
  searchIndex = {
    tools: [],
    collections: [],
    agents: [],
    pages: [],
  }
}

export function getIndex(): SearchIndex {
  return { ...searchIndex }
}


