#!/usr/bin/env python3
"""
Complete RAG Pipeline Example in Python

This example demonstrates:
1. Document ingestion (chunking + embedding)
2. RAG query execution
3. Evaluation metrics
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor

def main():
    # Get database connection string from environment
    dsn = os.getenv("NEURONDB_DSN", "postgres://postgres:postgres@localhost:5432/neurondb?sslmode=disable")
    
    conn = psycopg2.connect(dsn)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Step 1: Document Ingestion
        print("=== Step 1: Document Ingestion ===")
        document_text = """Machine learning is a subset of artificial intelligence that focuses on 
        the development of algorithms and statistical models that enable computer systems to 
        improve their performance on a specific task through experience. Unlike traditional 
        programming, where explicit instructions are provided, machine learning systems learn 
        from data patterns and make predictions or decisions based on that learning."""
        
        table_name = "documents"
        
        # Create table if it doesn't exist
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                content TEXT,
                embedding vector(384),
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        
        # Ingest document using rag_ingest_document
        ingest_query = """
            SELECT * FROM neurondb.rag_ingest_document(
                %s, %s, 'content', 'embedding', 'default', 512, 128, '{}'::jsonb
            )
        """
        cur.execute(ingest_query, (document_text, table_name))
        chunks = cur.fetchall()
        
        chunk_ids = []
        for chunk in chunks:
            chunk_ids.append(chunk['chunk_id'])
            print(f"  Ingested chunk {chunk['chunk_id']}: {chunk['chunk_text'][:50]}...")
        print(f"Successfully ingested {len(chunk_ids)} chunks\n")
        
        # Step 2: RAG Query
        print("=== Step 2: RAG Query ===")
        query = "What is machine learning?"
        
        # Use rag_query function
        rag_query = """
            SELECT chunk_text, relevance_score 
            FROM neurondb.rag_query(%s, %s, 'embedding', 'content', 'default', 3)
        """
        cur.execute(rag_query, (query, table_name))
        results = cur.fetchall()
        
        contexts = []
        for result in results:
            contexts.append(result['chunk_text'])
            print(f"  Retrieved context (relevance: {result['relevance_score']:.3f}): {result['chunk_text'][:50]}...")
        print(f"Retrieved {len(contexts)} context chunks\n")
        
        # Step 3: Generate Answer (using LLM if available)
        print("=== Step 3: Generate Answer ===")
        if contexts:
            context_text = "\n\n".join(contexts)
            prompt = f"Context:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
            
            try:
                llm_query = "SELECT ndb_llm_complete(%s, '{\"temperature\": 0.7, \"max_tokens\": 500}') AS answer"
                cur.execute(llm_query, (prompt,))
                result = cur.fetchone()
                answer = result['answer'] if result else "LLM not available"
                print(f"  Generated answer: {answer}\n")
            except Exception as e:
                print(f"  LLM not available: {e}\n")
                answer = "Machine learning is a subset of AI that enables systems to learn from data."
            
            # Step 4: Evaluation
            print("=== Step 4: RAG Evaluation ===")
            eval_query = """
                SELECT neurondb.rag_evaluate(%s, %s, %s::text[], 'basic') AS evaluation
            """
            cur.execute(eval_query, (query, answer, contexts))
            result = cur.fetchone()
            
            if result and result['evaluation']:
                import json
                evaluation = json.loads(result['evaluation']) if isinstance(result['evaluation'], str) else result['evaluation']
                print(f"  Relevancy: {evaluation.get('relevancy', 0.0):.3f}")
                print(f"  Semantic Similarity: {evaluation.get('semantic_similarity', 0.0):.3f}")
                if 'similarity_stats' in evaluation:
                    stats = evaluation['similarity_stats']
                    print(f"  Average Similarity: {stats.get('avg', 0.0):.3f}")
        
        print("\n=== RAG Pipeline Complete ===")
        
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    main()
