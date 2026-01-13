#!/usr/bin/env python3
"""
RAG Pipeline Example

This example demonstrates:
1. Ingesting documents
2. Generating embeddings
3. Creating vector indexes
4. Performing semantic search
5. Using RAG with an agent
"""

import psycopg2
from neuronagent import NeuronAgentClient

def main():
    # Connect to NeuronDB
    conn = psycopg2.connect(
        host="localhost",
        port=5433,
        database="neurondb",
        user="neurondb",
        password="neurondb"
    )

    # Step 1: Create documents table
    print("Creating documents table...")
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector(1536),
                metadata JSONB
            )
        """)
        conn.commit()

    # Step 2: Insert documents with embeddings
    print("Inserting documents...")
    documents = [
        "NeuronDB is a PostgreSQL extension for vector search and AI.",
        "HNSW indexing provides fast approximate nearest neighbor search.",
        "The extension supports 52+ ML algorithms for various tasks.",
    ]

    with conn.cursor() as cur:
        for doc in documents:
            # Generate embedding (using NeuronDB function)
            cur.execute("""
                INSERT INTO documents (content, embedding)
                VALUES (
                    %s,
                    neurondb_embed_text(%s, 'text-embedding-3-small')
                )
            """, (doc, doc))
        conn.commit()

    # Step 3: Create HNSW index
    print("Creating HNSW index...")
    with conn.cursor() as cur:
        cur.execute("""
            CREATE INDEX IF NOT EXISTS documents_embedding_idx
            ON documents
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """)
        conn.commit()

    # Step 4: Perform semantic search
    print("Performing semantic search...")
    query = "What is NeuronDB?"
    
    with conn.cursor() as cur:
        # Generate query embedding
        cur.execute("""
            SELECT neurondb_embed_text(%s, 'text-embedding-3-small')
        """, (query,))
        query_embedding = cur.fetchone()[0]

        # Search
        cur.execute("""
            SELECT id, content, embedding <=> %s AS distance
            FROM documents
            ORDER BY embedding <=> %s
            LIMIT 5
        """, (query_embedding, query_embedding))
        
        results = cur.fetchall()
        print(f"\nFound {len(results)} results:")
        for doc_id, content, distance in results:
            print(f"  [{distance:.4f}] {content[:80]}...")

    # Step 5: Use with agent for RAG
    print("\nUsing RAG with agent...")
    client = NeuronAgentClient(
        base_url="http://localhost:8080",
        api_key="your-api-key"
    )

    # Create agent with SQL tool enabled
    agent = client.agents.create_agent(
        name="rag-agent",
        system_prompt="""You are a helpful assistant that answers questions using 
        the documents in the database. Use the SQL tool to search for relevant 
        information before answering.""",
        model_name="gpt-4",
        enabled_tools=["sql"]
    )

    session = client.sessions.create_session(agent_id=agent.id)
    
    response = client.sessions.send_message(
        session_id=session.id,
        content="What is NeuronDB and what are its key features?"
    )
    
    print(f"\nAgent response: {response.content}")

    conn.close()
    print("\nRAG pipeline example complete!")

if __name__ == "__main__":
    main()







