/**
 * Basic NeuronAgent usage example
 * 
 * This example demonstrates:
 * 1. Creating an agent
 * 2. Creating a session
 * 3. Sending messages
 * 4. Retrieving conversation history
 */

import { NeuronAgentClient } from '@neurondb/neuronagent'

async function main() {
  // Initialize client
  const client = new NeuronAgentClient({
    baseURL: 'http://localhost:8080',
    apiKey: 'your-api-key-here'
  })

  // Step 1: Create an agent
  console.log('Creating agent...')
  const agent = await client.agents.createAgent({
    name: 'example-agent',
    description: 'A simple example agent',
    systemPrompt: 'You are a helpful assistant that answers questions clearly and concisely.',
    modelName: 'gpt-4',
    enabledTools: ['sql', 'http'],
    config: {
      temperature: 0.7,
      maxTokens: 1000
    }
  })
  console.log(`Created agent: ${agent.id}`)

  // Step 2: Create a session
  console.log('\nCreating session...')
  const session = await client.sessions.createSession({
    agentId: agent.id,
    metadata: { userId: 'example-user' }
  })
  console.log(`Created session: ${session.id}`)

  // Step 3: Send messages
  console.log('\nSending messages...')
  
  const messages = [
    'Hello! Can you help me understand vector databases?',
    'What are the key advantages of using NeuronDB?',
    'How does HNSW indexing work?'
  ]

  for (const message of messages) {
    console.log(`\nUser: ${message}`)
    const response = await client.sessions.sendMessage(session.id, {
      content: message
    })
    console.log(`Agent: ${response.content}`)
  }

  // Step 4: Retrieve conversation history
  console.log('\n\nRetrieving conversation history...')
  const history = await client.sessions.getMessages(session.id)
  
  console.log(`\nTotal messages: ${history.length}`)
  for (const msg of history) {
    console.log(`\n[${msg.role}]: ${msg.content.substring(0, 100)}...`)
  }

  console.log('\nExample complete!')
}

main().catch(console.error)







