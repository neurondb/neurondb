#!/usr/bin/env python3
"""
Simple Hugging Face Dataset Loader for NeuronDB

This example demonstrates how to:
1. Load a simple dataset from Hugging Face
2. Create a table in PostgreSQL
3. Generate embeddings using NeuronDB's embed_text function
4. Store embeddings in the database

This script uses NeuronDB's built-in embedding functions, so it works
with the MCP server and doesn't require local model installation.
"""

import os
import sys

# Try to import required packages
try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: datasets library not installed")
    print("Install with: pip install datasets")
    sys.exit(1)

try:
    import psycopg2
    from psycopg2.extensions import quote_ident
except ImportError:
    print("ERROR: psycopg2 not installed")
    print("Install with: pip install psycopg2-binary")
    sys.exit(1)

# Database configuration (can be overridden by environment variables)
DB_CONFIG = {
    'host': os.getenv('PGHOST', 'localhost'),
    'port': int(os.getenv('PGPORT', '5432')),
    'database': os.getenv('PGDATABASE', 'neurondb'),
    'user': os.getenv('PGUSER', 'postgres'),
    'password': os.getenv('PGPASSWORD', '')
}

# Simple datasets that work well for testing
SIMPLE_DATASETS = {
    'squad': {
        'split': 'train',
        'text_column': 'question',
        'description': 'Stanford Question Answering Dataset - questions'
    },
    'imdb': {
        'split': 'train',
        'text_column': 'text',
        'description': 'IMDB movie reviews'
    },
    'ag_news': {
        'split': 'train',
        'text_column': 'text',
        'description': 'AG News articles'
    }
}


def load_hf_dataset(dataset_name='squad', split='train', limit=100):
    """
    Load a dataset from Hugging Face
    
    Args:
        dataset_name: Name of the Hugging Face dataset
        split: Dataset split (train, test, validation)
        limit: Maximum number of rows to load
    
    Returns:
        List of dictionaries containing the dataset rows
    """
    print(f"Loading dataset: {dataset_name} (split: {split})")
    
    try:
        # Try streaming first (more memory efficient)
        dataset = load_dataset(dataset_name, split=split, streaming=True)
        data = []
        count = 0
        for item in dataset:
            if isinstance(item, dict):
                data.append(item)
            else:
                data.append(dict(item))
            count += 1
            if count >= limit:
                break
        print(f"Loaded {len(data)} rows using streaming")
    except Exception as e:
        # Fallback to non-streaming
        print(f"Streaming failed, trying non-streaming mode: {e}")
        dataset = load_dataset(dataset_name, split=split, streaming=False)
        if limit > 0 and len(dataset) > limit:
            dataset = dataset.select(range(limit))
        data = [dict(row) for row in dataset]
        print(f"Loaded {len(data)} rows using non-streaming")
    
    return data


def create_table(conn, schema_name, table_name, text_column, embedding_dim=384):
    """
    Create a table for storing dataset with embeddings
    
    Args:
        conn: PostgreSQL connection
        schema_name: Schema name
        table_name: Table name
        text_column: Name of the text column
        embedding_dim: Embedding dimension (default: 384 for all-MiniLM-L6-v2)
    """
    cur = conn.cursor()
    
    # Create schema if not exists
    schema_quoted = quote_ident(schema_name, cur)
    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_quoted}")
    
    # Create table
    table_quoted = quote_ident(table_name, cur)
    cur.execute(f"""
        DROP TABLE IF EXISTS {schema_quoted}.{table_quoted};
        CREATE TABLE {schema_quoted}.{table_quoted} (
            id SERIAL PRIMARY KEY,
            {quote_ident(text_column, cur)} TEXT NOT NULL,
            embedding vector({embedding_dim}),
            metadata JSONB
        )
    """)
    
    conn.commit()
    cur.close()
    print(f"Created table: {schema_name}.{table_name}")


def insert_data_with_embeddings(conn, schema_name, table_name, data, text_column, 
                                 embedding_model='default', batch_size=10):
    """
    Insert data and generate embeddings using NeuronDB's embed_text function
    
    Args:
        conn: PostgreSQL connection
        schema_name: Schema name
        table_name: Table name
        data: List of dictionaries containing data
        text_column: Name of the text column
        embedding_model: Embedding model name (default: 'default')
        batch_size: Number of rows to process before committing
    """
    cur = conn.cursor()
    schema_quoted = quote_ident(schema_name, cur)
    table_quoted = quote_ident(table_name, cur)
    text_col_quoted = quote_ident(text_column, cur)
    
    inserted = 0
    errors = 0
    
    print(f"Inserting {len(data)} rows with embeddings...")
    
    for i, row in enumerate(data):
        try:
            # Get text value
            text = row.get(text_column, '')
            if not text:
                # Try common column names
                text = row.get('sentence', row.get('document', row.get('content', str(row))))
            
            if not text or len(str(text).strip()) == 0:
                continue
            
            # Prepare metadata (all other columns except text)
            metadata = {k: v for k, v in row.items() if k != text_column}
            
            # Generate embedding using NeuronDB's embed_text function
            # This uses the configured embedding model in PostgreSQL
            embed_sql = f"""
                INSERT INTO {schema_quoted}.{table_quoted} 
                    ({text_col_quoted}, embedding, metadata)
                VALUES 
                    (%s, embed_text(%s, %s), %s::jsonb)
                RETURNING id
            """
            
            cur.execute(embed_sql, (str(text), str(text), embedding_model, str(metadata)))
            row_id = cur.fetchone()[0]
            inserted += 1
            
            # Commit in batches
            if inserted % batch_size == 0:
                conn.commit()
                print(f"  Inserted {inserted}/{len(data)} rows...")
        
        except Exception as e:
            errors += 1
            print(f"  Error inserting row {i+1}: {e}")
            if errors > 10:
                print("  Too many errors, stopping")
                break
            continue
    
    # Final commit
    conn.commit()
    cur.close()
    
    print(f"✅ Inserted {inserted} rows (errors: {errors})")
    return inserted


def test_semantic_search(conn, schema_name, table_name, text_column, query_text, limit=5):
    """
    Test semantic search on the loaded data
    
    Args:
        conn: PostgreSQL connection
        schema_name: Schema name
        table_name: Table name
        text_column: Name of the text column
        query_text: Query text for semantic search
        limit: Number of results to return
    """
    cur = conn.cursor()
    schema_quoted = quote_ident(schema_name, cur)
    table_quoted = quote_ident(table_name, cur)
    text_col_quoted = quote_ident(text_column, cur)
    
    print(f"\n🔍 Testing semantic search with query: '{query_text}'")
    
    # Generate query embedding and search
    search_sql = f"""
        SELECT 
            id,
            {text_col_quoted},
            1 - (embedding <=> embed_text(%s)) as similarity
        FROM {schema_quoted}.{table_quoted}
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> embed_text(%s)
        LIMIT %s
    """
    
    cur.execute(search_sql, (query_text, query_text, limit))
    results = cur.fetchall()
    
    print(f"\nTop {len(results)} results:")
    for i, (row_id, text, similarity) in enumerate(results, 1):
        text_preview = str(text)[:150] + "..." if len(str(text)) > 150 else str(text)
        print(f"\n{i}. Similarity: {similarity:.4f} | ID: {row_id}")
        print(f"   Text: {text_preview}")
    
    cur.close()
    return results


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Load Hugging Face dataset into NeuronDB with embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Load SQuAD dataset (100 rows)
  {sys.argv[0]} --dataset squad --limit 100
  
  # Load IMDB reviews (50 rows)
  {sys.argv[0]} --dataset imdb --limit 50
  
  # Load AG News (200 rows) with custom model
  {sys.argv[0]} --dataset ag_news --limit 200 --model sentence-transformers/all-MiniLM-L6-v2

Simple datasets available:
{chr(10).join(f'  - {name}: {info["description"]}' for name, info in SIMPLE_DATASETS.items())}
        """
    )
    
    parser.add_argument('--dataset', default='squad',
                       help='Dataset name (default: squad)')
    parser.add_argument('--split', default='train',
                       help='Dataset split (default: train)')
    parser.add_argument('--text-column', default=None,
                       help='Text column name (auto-detected if not specified)')
    parser.add_argument('--limit', type=int, default=100,
                       help='Maximum rows to load (default: 100)')
    parser.add_argument('--schema', default='datasets',
                       help='Schema name (default: datasets)')
    parser.add_argument('--table', default=None,
                       help='Table name (auto-generated if not specified)')
    parser.add_argument('--model', default='default',
                       help='Embedding model (default: default, uses PostgreSQL config)')
    parser.add_argument('--test-query', default=None,
                       help='Test query for semantic search (default: auto-generated)')
    parser.add_argument('--skip-search', action='store_true',
                       help='Skip semantic search test')
    
    args = parser.parse_args()
    
    # Auto-detect text column if not specified
    if args.text_column is None:
        if args.dataset in SIMPLE_DATASETS:
            args.text_column = SIMPLE_DATASETS[args.dataset]['text_column']
            if args.split == 'train' and SIMPLE_DATASETS[args.dataset].get('split'):
                args.split = SIMPLE_DATASETS[args.dataset]['split']
        else:
            args.text_column = 'text'  # Default
    
    # Generate table name if not specified
    if args.table is None:
        args.table = args.dataset.replace('-', '_').replace('/', '_')
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║   Hugging Face Dataset Loader for NeuronDB                   ║
║   Using NeuronDB embed_text function for embeddings          ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    print(f"Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Split: {args.split}")
    print(f"  Text column: {args.text_column}")
    print(f"  Limit: {args.limit}")
    print(f"  Schema: {args.schema}")
    print(f"  Table: {args.table}")
    print(f"  Embedding model: {args.model}")
    print()
    
    try:
        # Load dataset
        data = load_hf_dataset(args.dataset, args.split, args.limit)
        
        if not data:
            print("❌ No data loaded")
            return 1
        
        # Connect to database
        print(f"\nConnecting to database: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
        conn = psycopg2.connect(**DB_CONFIG)
        
        # Create table (default embedding dimension: 384 for all-MiniLM-L6-v2)
        create_table(conn, args.schema, args.table, args.text_column, embedding_dim=384)
        
        # Insert data with embeddings
        inserted = insert_data_with_embeddings(
            conn, args.schema, args.table, data, args.text_column, args.model
        )
        
        if inserted == 0:
            print("❌ No rows inserted")
            conn.close()
            return 1
        
        # Test semantic search
        if not args.skip_search:
            if args.test_query:
                test_query = args.test_query
            else:
                # Generate a simple test query based on dataset
                if args.dataset == 'squad':
                    test_query = "What is the capital of France?"
                elif args.dataset == 'imdb':
                    test_query = "great movie with excellent acting"
                elif args.dataset == 'ag_news':
                    test_query = "technology and innovation"
                else:
                    test_query = "interesting topic"
            
            test_semantic_search(conn, args.schema, args.table, args.text_column, test_query)
        
        conn.close()
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║   ✅ SUCCESS!                                                 ║
║                                                               ║
║   Dataset loaded into: {args.schema}.{args.table:<30} ║
║   Rows inserted: {inserted:<43} ║
║                                                               ║
║   Query in SQL:                                               ║
║   SELECT * FROM {args.schema}.{args.table:<35} LIMIT 10; ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())



