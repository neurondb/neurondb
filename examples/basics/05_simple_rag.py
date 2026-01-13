#!/usr/bin/env python3
"""
Example 5: Simple RAG (Retrieval-Augmented Generation)
=======================================================
Learn how to:
- Store documents with embeddings
- Retrieve relevant context
- Use retrieved context (simulated LLM response)

This is a minimal RAG example. For production use, integrate with
OpenAI, Anthropic, or other LLM APIs.

Run: python 05_simple_rag.py

Requirements:
    pip install psycopg2-binary sentence-transformers
"""

import psycopg2
import os
from sentence_transformers import SentenceTransformer

# Database connection (defaults match Docker Compose setup)
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5433')),  # Docker Compose default port
    'database': os.getenv('DB_NAME', 'neurondb'),
    'user': os.getenv('DB_USER', 'neurondb'),  # Docker Compose default user
    'password': os.getenv('DB_PASSWORD', 'neurondb')  # Docker Compose default password
}

MODEL_NAME = "all-MiniLM-L6-v2"

print("=" * 60)
print("Example 5: Simple RAG (Retrieval-Augmented Generation)")
print("=" * 60)

# Load model and connect
model = SentenceTransformer(MODEL_NAME)
embedding_dim = model.get_sentence_embedding_dimension()

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# Create knowledge base
print("\n1. Creating knowledge base...")
cur.execute(f"""
    CREATE TABLE IF NOT EXISTS knowledge_base (
        id SERIAL PRIMARY KEY,
        title TEXT,
        content TEXT,
        embedding vector({embedding_dim})
    );
    
    CREATE INDEX IF NOT EXISTS knowledge_base_embedding_idx 
    ON knowledge_base 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
""")
conn.commit()
print("✓ Knowledge base created")

# Add some knowledge documents
print("\n2. Adding documents to knowledge base...")
documents = [
    ("Machine Learning Basics", 
     "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions."),
    
    ("Python Programming", 
     "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in data science, web development, automation, and artificial intelligence."),
    
    ("Database Systems", 
     "A database is an organized collection of data stored and accessed electronically. PostgreSQL is a powerful open-source relational database management system that supports advanced features like vector search."),
    
    ("Neural Networks", 
     "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information and can learn complex patterns from data."),
    
    ("Vector Embeddings", 
     "Vector embeddings are numerical representations of text, images, or other data in a high-dimensional space. Similar items have similar vectors, enabling semantic search and similarity matching."),
]

for title, content in documents:
    # Create embedding from title + content
    text = f"{title}: {content}"
    embedding = model.encode(text)
    embedding_str = '[' + ','.join(str(x) for x in embedding) + ']'
    
    cur.execute(
        "INSERT INTO knowledge_base (title, content, embedding) VALUES (%s, %s, %s::vector)",
        (title, content, embedding_str)
    )
    print(f"  ✓ Added: {title}")

conn.commit()
print(f"✓ Inserted {len(documents)} documents")

# RAG function: retrieve relevant context
def retrieve_context(query: str, top_k: int = 2):
    """Retrieve most relevant documents for a query"""
    query_embedding = model.encode(query)
    query_embedding_str = '[' + ','.join(str(x) for x in query_embedding) + ']'
    
    cur.execute("""
        SELECT 
            title,
            content,
            embedding <=> %s::vector AS distance
        FROM knowledge_base
        ORDER BY distance
        LIMIT %s
    """, (query_embedding_str, top_k))
    
    return cur.fetchall()

# Simulated LLM response (in production, call OpenAI/Anthropic/etc.)
def generate_response(query: str, context: list):
    """Simulate LLM response using retrieved context"""
    context_text = "\n\n".join([f"{title}:\n{content}" for title, content, _ in context])
    
    # In a real implementation, you would call an LLM API here:
    # response = openai.ChatCompletion.create(
    #     model="gpt-4",
    #     messages=[
    #         {"role": "system", "content": "Answer based on the provided context."},
    #         {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
    #     ]
    # )
    
    # For this example, we'll just return a simple response
    return f"Based on the retrieved context, here's information about: {query}"

# Example queries
print("\n3. Example RAG queries:")
queries = [
    "What is machine learning?",
    "Tell me about Python",
    "How do vector embeddings work?",
]

for query in queries:
    print(f"\n  Query: {query}")
    print("  " + "-" * 56)
    
    # Retrieve relevant context
    context = retrieve_context(query, top_k=2)
    
    print("  Retrieved context:")
    for i, (title, content, distance) in enumerate(context, 1):
        print(f"    {i}. {title} (similarity: {1-distance:.4f})")
        print(f"       {content[:80]}...")
    
    # Generate response (simulated)
    response = generate_response(query, context)
    print(f"\n  Response: {response}")

# Cleanup
print("\n4. Cleaning up...")
cur.execute("DROP TABLE IF EXISTS knowledge_base CASCADE")
conn.commit()
print("✓ Knowledge base dropped")

cur.close()
conn.close()

print("\n" + "=" * 60)
print("Example complete! ✓")
print("=" * 60)
print("\nTo use with real LLMs:")
print("  1. Install: pip install openai (or anthropic)")
print("  2. Set API key: export OPENAI_API_KEY=your_key")
print("  3. Replace generate_response() with actual LLM API calls")








