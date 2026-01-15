#!/usr/bin/env python3
"""-------------------------------------------------------------------------
 *
 * neuronmcp-test-rag-sample-data.py
 *    Test RAG with Sample Data using NeuronMCP Tools
 *
 * Demonstrates RAG (Retrieval-Augmented Generation) workflow using NeuronMCP
 * tools. Works through ClaudeDesktop -> NeuronMCP -> NeuronDB pipeline.
 * Creates sample documents table, inserts sample data, generates embeddings
 * using batch_embedding tool, and tests RAG retrieval using retrieve_context tool.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/tests/neuronmcp-test-rag-sample-data.py
 *
 *-------------------------------------------------------------------------
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add client directory to path
sys.path.insert(0, str(Path(__file__).parent / "client"))

from mcp_client.client import MCPClient
from mcp_client.config import load_config


class RAGTester:
    """Test RAG functionality using NeuronMCP tools."""
    
    def __init__(self, config_path: str, server_name: str = "neurondb"):
        """Initialize RAG tester with MCP client."""
        self.config = load_config(config_path, server_name)
        self.client = MCPClient(self.config, verbose=True)
        self.table_name = "rag_test_documents"
        
    def connect(self):
        """Connect to MCP server."""
        print("Connecting to NeuronMCP server...")
        self.client.connect()
        print("✓ Connected\n")
    
    def disconnect(self):
        """Disconnect from MCP server."""
        self.client.disconnect()
        print("\n✓ Disconnected from NeuronMCP server")
    
    def execute_sql(self, query: str, read_only: bool = False) -> Dict[str, Any]:
        """Execute SQL query using postgresql_execute_query tool."""
        result = self.client.call_tool("postgresql_execute_query", {
            "query": query,
            "read_only": read_only,
            "max_rows": 1000,
            "timeout_seconds": 60
        })
        
        if result.get("isError", False):
            error_msg = self._extract_error(result)
            raise RuntimeError(f"SQL execution failed: {error_msg}")
        
        return result
    
    def _extract_error(self, result: Dict[str, Any]) -> str:
        """Extract error message from result."""
        if result.get("isError", False):
            content = result.get("content", [])
            if content and isinstance(content, list) and len(content) > 0:
                if isinstance(content[0], dict) and "text" in content[0]:
                    return content[0]["text"]
        
        if "error" in result:
            return str(result["error"])
        
        return "Unknown error"
    
    def _extract_results(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract rows from query result."""
        content = result.get("content", [])
        if content and isinstance(content, list) and len(content) > 0:
            if isinstance(content[0], dict) and "text" in content[0]:
                text = content[0]["text"]
                try:
                    # Try to parse JSON from text
                    if "{" in text:
                        json_start = text.find("{")
                        json_end = text.rfind("}") + 1
                        if json_end > json_start:
                            json_str = text[json_start:json_end]
                            data = json.loads(json_str)
                            if "rows" in data:
                                return data["rows"]
                except:
                    pass
        
        # Try to get rows directly from result
        if "rows" in result:
            return result["rows"]
        
        return []
    
    def setup_tables(self):
        """Create tables for RAG testing."""
        print("=" * 70)
        print("Step 1: Creating Tables")
        print("=" * 70)
        
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding VECTOR(384),
            metadata JSONB DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_embedding 
        ON {self.table_name} 
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """
        
        try:
            self.execute_sql(create_table_query, read_only=False)
            print("✓ Tables created successfully\n")
        except Exception as e:
            print(f"✗ Error creating tables: {e}\n")
            raise
    
    def insert_sample_data(self):
        """Insert sample documents."""
        print("=" * 70)
        print("Step 2: Inserting Sample Data")
        print("=" * 70)
        
        sample_documents = [
            {
                "title": "PostgreSQL Performance Tuning",
                "content": "PostgreSQL performance can be significantly improved through proper indexing strategies. B-tree indexes are the default and work well for most queries. GiST indexes are useful for full-text search and geometric data. Hash indexes can be faster for equality comparisons but are not WAL-logged. Partial indexes can reduce index size and improve performance for queries with common WHERE clauses.",
                "metadata": {"category": "database", "tags": ["postgresql", "performance", "indexing"]}
            },
            {
                "title": "Vector Databases Explained",
                "content": "Vector databases store high-dimensional vector embeddings generated from machine learning models. These embeddings capture semantic meaning of text, images, or other data. Vector similarity search using cosine similarity or Euclidean distance enables semantic search capabilities. HNSW and IVFFlat are popular indexing algorithms that make approximate nearest neighbor search fast even with millions of vectors.",
                "metadata": {"category": "machine_learning", "tags": ["vectors", "embeddings", "similarity_search"]}
            },
            {
                "title": "Retrieval-Augmented Generation Overview",
                "content": "RAG combines the power of large language models with external knowledge retrieval. The process involves: 1) Converting user queries to embeddings, 2) Retrieving relevant documents using vector similarity, 3) Providing retrieved context to the LLM, 4) Generating accurate responses grounded in factual data. This approach reduces hallucinations and enables LLMs to access up-to-date information.",
                "metadata": {"category": "ai", "tags": ["rag", "llm", "retrieval"]}
            },
            {
                "title": "Python Machine Learning Best Practices",
                "content": "When building ML models in Python, always split your data into training, validation, and test sets. Use cross-validation to get robust performance estimates. Feature scaling with StandardScaler or MinMaxScaler often improves model performance. Handle missing data appropriately using imputation or deletion strategies. Use pipelines to ensure consistent preprocessing.",
                "metadata": {"category": "machine_learning", "tags": ["python", "sklearn", "best_practices"]}
            },
            {
                "title": "Database Sharding Strategies",
                "content": "Sharding distributes data across multiple database instances to improve scalability. Common strategies include: Range-based sharding (e.g., by date), Hash-based sharding (distribute evenly), Directory-based sharding (lookup table), and Geographic sharding (by location). Each approach has trade-offs in terms of query complexity, data distribution, and rebalancing difficulty.",
                "metadata": {"category": "database", "tags": ["sharding", "scalability", "distributed"]}
            }
        ]
        
        # Insert documents
        for doc in sample_documents:
            insert_query = f"""
            INSERT INTO {self.table_name} (title, content, metadata)
            VALUES (
                {self._sql_escape(doc['title'])},
                {self._sql_escape(doc['content'])},
                '{json.dumps(doc['metadata'])}'::jsonb
            )
            """
            
            try:
                self.execute_sql(insert_query, read_only=False)
                print(f"✓ Inserted: {doc['title']}")
            except Exception as e:
                print(f"✗ Error inserting {doc['title']}: {e}")
                raise
        
        # Verify insertion
        count_query = f"SELECT COUNT(*) as count FROM {self.table_name}"
        result = self.execute_sql(count_query, read_only=True)
        rows = self._extract_results(result)
        if rows and len(rows) > 0:
            count = rows[0].get("count", 0)
            print(f"\n✓ Total documents: {count}\n")
        else:
            print("\n⚠ Could not verify document count\n")
    
    def _sql_escape(self, text: str) -> str:
        """Escape SQL string."""
        # Simple escaping - replace single quotes
        escaped = text.replace("'", "''")
        return f"'{escaped}'"
    
    def generate_embeddings(self):
        """Generate embeddings for all documents using SQL embedding function."""
        print("=" * 70)
        print("Step 3: Generating Embeddings")
        print("=" * 70)
        
        # Use SQL to generate embeddings directly (more reliable)
        # This uses embed_text SQL function
        update_query = f"""
        UPDATE {self.table_name}
        SET embedding = embed_text(content, 'sentence-transformers/all-MiniLM-L6-v2'::text)
        WHERE embedding IS NULL
        """
        
        print("Generating embeddings using embed_text function...")
        
        try:
            self.execute_sql(update_query, read_only=False)
            print("✓ Embeddings generated\n")
        except Exception as e:
            print(f"✗ Error generating embeddings: {e}\n")
            # Try alternative: generate one at a time
            print("Trying alternative approach (one at a time)...")
            self._generate_embeddings_one_by_one()
    
    def _generate_embeddings_one_by_one(self):
        """Generate embeddings one document at a time (fallback method)."""
        # Get all documents without embeddings
        select_query = f"SELECT id, content FROM {self.table_name} WHERE embedding IS NULL ORDER BY id"
        result = self.execute_sql(select_query, read_only=True)
        rows = self._extract_results(result)
        
        if not rows:
            print("✓ All documents already have embeddings\n")
            return
        
        print(f"Generating embeddings for {len(rows)} documents...")
        
        for i, row in enumerate(rows, 1):
            doc_id = row.get("id")
            content = row.get("content", "")
            
            if not content:
                continue
            
            # Escape content for SQL
            content_escaped = content.replace("'", "''")
            
            # Generate embedding using SQL function
            update_query = f"""
            UPDATE {self.table_name}
            SET embedding = embed_text('{content_escaped}'::text, 'sentence-transformers/all-MiniLM-L6-v2'::text)
            WHERE id = {doc_id}
            """
            
            try:
                self.execute_sql(update_query, read_only=False)
                if i % 5 == 0 or i == len(rows):
                    print(f"  Generated {i}/{len(rows)} embeddings...")
            except Exception as e:
                print(f"✗ Error generating embedding for document {doc_id}: {e}")
        
        print(f"✓ Completed embedding generation\n")
    
    def test_rag_retrieval(self):
        """Test RAG retrieval using retrieve_context tool."""
        print("=" * 70)
        print("Step 4: Testing RAG Retrieval")
        print("=" * 70)
        
        test_queries = [
            "How do database indexes work?",
            "What is retrieval augmented generation?",
            "How to improve machine learning model performance?",
            "What are vector similarity search methods?"
        ]
        
        for query in test_queries:
            print(f"\nQuery: \"{query}\"")
            print("-" * 70)
            
            try:
                result = self.client.call_tool("retrieve_context", {
                    "query": query,
                    "table": self.table_name,
                    "vector_column": "embedding",
                    "limit": 3
                })
                
                if result.get("isError", False):
                    error_msg = self._extract_error(result)
                    print(f"✗ Error: {error_msg}")
                    continue
                
                # Extract context from result
                context = self._extract_context(result)
                
                if context:
                    print(f"✓ Retrieved {len(context)} results:")
                    for i, ctx in enumerate(context[:3], 1):
                        if isinstance(ctx, dict):
                            title = ctx.get("title", "N/A")
                            content = ctx.get("content", ctx.get("text", str(ctx)))
                            similarity = ctx.get("similarity", ctx.get("distance", "N/A"))
                            print(f"\n{i}. {title} (similarity: {similarity})")
                            print(f"   {content[:150]}...")
                        elif isinstance(ctx, str):
                            print(f"\n{i}. {ctx[:150]}...")
                        else:
                            print(f"\n{i}. {str(ctx)[:150]}...")
                else:
                    print("⚠ No context retrieved")
                
            except Exception as e:
                print(f"✗ Error: {e}")
        
        print("\n" + "=" * 70 + "\n")
    
    def _extract_context(self, result: Dict[str, Any]) -> List[Any]:
        """Extract context from retrieve_context result."""
        content = result.get("content", [])
        if content and isinstance(content, list) and len(content) > 0:
            if isinstance(content[0], dict) and "text" in content[0]:
                text = content[0]["text"]
                try:
                    # Try to parse JSON from text
                    if "{" in text or "[" in text:
                        json_start = text.find("[")
                        if json_start == -1:
                            json_start = text.find("{")
                        json_end = text.rfind("]") + 1
                        if json_end == 0:
                            json_end = text.rfind("}") + 1
                        
                        if json_end > json_start:
                            json_str = text[json_start:json_end]
                            data = json.loads(json_str)
                            if isinstance(data, list):
                                return data
                            elif isinstance(data, dict) and "context" in data:
                                return data["context"]
                except:
                    pass
        
        # Try direct access
        if "context" in result:
            return result["context"]
        
        return []
    
    def verify_setup(self):
        """Verify the setup by checking document and embedding counts."""
        print("=" * 70)
        print("Verification")
        print("=" * 70)
        
        verify_query = f"""
        SELECT 
            COUNT(*) as total_docs,
            COUNT(embedding) as docs_with_embeddings,
            COUNT(*) - COUNT(embedding) as docs_without_embeddings
        FROM {self.table_name}
        """
        
        try:
            result = self.execute_sql(verify_query, read_only=True)
            rows = self._extract_results(result)
            
            if rows and len(rows) > 0:
                stats = rows[0]
                total = stats.get("total_docs", 0)
                with_emb = stats.get("docs_with_embeddings", 0)
                without_emb = stats.get("docs_without_embeddings", 0)
                
                print(f"Total documents: {total}")
                print(f"Documents with embeddings: {with_emb}")
                print(f"Documents without embeddings: {without_emb}")
                
                if with_emb > 0:
                    print("\n✓ Setup complete! Ready for RAG testing.\n")
                else:
                    print("\n⚠ No embeddings found. Please run generate_embeddings step.\n")
            else:
                print("⚠ Could not verify setup\n")
                
        except Exception as e:
            print(f"✗ Error during verification: {e}\n")
    
    def run_full_test(self):
        """Run the complete RAG test workflow."""
        try:
            self.connect()
            self.setup_tables()
            self.insert_sample_data()
            self.generate_embeddings()
            self.verify_setup()
            self.test_rag_retrieval()
        finally:
            self.disconnect()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test RAG with Sample Data using NeuronMCP Tools"
    )
    parser.add_argument(
        "-c", "--config",
        default=str(Path(__file__).parent.parent / "conf" / "neuronmcp-server.json"),
        help="Path to NeuronMCP server configuration file"
    )
    parser.add_argument(
        "--server-name",
        default="neurondb",
        help="Server name from config"
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip table creation and data insertion"
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation"
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    # Create tester and run
    tester = RAGTester(str(config_path), args.server_name)
    
    try:
        tester.connect()
        
        if not args.skip_setup:
            tester.setup_tables()
            tester.insert_sample_data()
        
        if not args.skip_embeddings:
            tester.generate_embeddings()
        
        tester.verify_setup()
        tester.test_rag_retrieval()
        
    finally:
        tester.disconnect()


if __name__ == "__main__":
    main()

