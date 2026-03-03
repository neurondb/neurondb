"""
NeuronDB RAG Chatbot Example
Complete RAG (Retrieval-Augmented Generation) chatbot over PDF documents

Requirements:
    pip install psycopg2-binary sentence-transformers pypdf openai anthropic python-dotenv
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import psycopg2
from sentence_transformers import SentenceTransformer
import json

try:
    from pypdf import PdfReader
except ImportError:
    print("Warning: pypdf not installed. PDF ingestion will not work.")
    print("Install with: pip install pypdf")

# Configuration (defaults match Docker Compose setup)
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5433')),  # Docker Compose default port
    'database': os.getenv('DB_NAME', 'neurondb'),
    'user': os.getenv('DB_USER', 'neurondb'),  # Docker Compose default user
    'password': os.getenv('DB_PASSWORD', 'neurondb')  # Docker Compose default password
}

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


class PDFIngester:
    """Ingests PDF documents and generates embeddings"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.conn = psycopg2.connect(**db_config)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self._setup_schema()
    
    def _setup_schema(self):
        """Create necessary tables"""
        with self.conn.cursor() as cur:
            # Create documents table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pdf_documents (
                    id SERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    page_number INTEGER,
                    content TEXT NOT NULL,
                    embedding vector(384),
                    chunk_index INTEGER,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create HNSW index
            cur.execute("""
                CREATE INDEX IF NOT EXISTS pdf_documents_embedding_idx 
                ON pdf_documents 
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """)
            
            self.conn.commit()
            print("✓ Schema created")
    
    def extract_text_from_pdf(self, filepath: Path) -> List[Dict[str, Any]]:
        """Extract text from PDF with page numbers"""
        reader = PdfReader(str(filepath))
        pages = []
        
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text.strip():
                pages.append({
                    'page_number': page_num,
                    'text': text
                })
        
        return pages
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = ' '.join(words[i:i + CHUNK_SIZE])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def ingest_pdf(self, filepath: Path):
        """Ingest a single PDF"""
        print(f"Processing: {filepath.name}")
        
        # Extract text from PDF
        pages = self.extract_text_from_pdf(filepath)
        print(f"  Extracted {len(pages)} pages")
        
        total_chunks = 0
        
        # Process each page
        with self.conn.cursor() as cur:
            for page_data in pages:
                # Chunk page text
                chunks = self.chunk_text(page_data['text'])
                
                for idx, chunk in enumerate(chunks):
                    # Generate embedding
                    embedding = self.model.encode(chunk)
                    embedding_list = embedding.tolist()
                    
                    # Prepare metadata
                    metadata = {
                        'page_number': page_data['page_number'],
                        'total_pages': len(pages)
                    }
                    
                    # Store in database
                    cur.execute("""
                        INSERT INTO pdf_documents 
                        (filename, page_number, content, embedding, chunk_index, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        filepath.name,
                        page_data['page_number'],
                        chunk,
                        embedding_list,
                        idx,
                        json.dumps(metadata)
                    ))
                    
                    total_chunks += 1
        
        self.conn.commit()
        print(f"✓ Ingested {filepath.name} ({total_chunks} chunks)")
    
    def ingest_directory(self, dirpath: Path):
        """Ingest all PDF files in directory"""
        pdf_files = list(dirpath.glob('**/*.pdf'))
        
        print(f"\nFound {len(pdf_files)} PDF files to ingest\n")
        
        for filepath in pdf_files:
            try:
                self.ingest_pdf(filepath)
            except Exception as e:
                print(f"✗ Error processing {filepath.name}: {e}")
        
        print(f"\n✓ Ingestion complete!")
    
    def close(self):
        self.conn.close()


class RAGChatbot:
    """RAG Chatbot with retrieval and generation"""
    
    def __init__(self, db_config: Dict[str, Any], llm_provider: str = "openai"):
        self.conn = psycopg2.connect(**db_config)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.llm_provider = llm_provider
        self._setup_llm()
    
    def _setup_llm(self):
        """Setup LLM client"""
        if self.llm_provider == "openai":
            try:
                import openai
                self.llm_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                self.llm_model = "gpt-3.5-turbo"
            except ImportError:
                print("Warning: openai not installed. Install with: pip install openai")
                self.llm_client = None
        elif self.llm_provider == "anthropic":
            try:
                import anthropic
                self.llm_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
                self.llm_model = "claude-3-sonnet-20240229"
            except ImportError:
                print("Warning: anthropic not installed. Install with: pip install anthropic")
                self.llm_client = None
        else:
            self.llm_client = None
    
    def retrieve_context(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant context from documents"""
        # Generate query embedding
        query_embedding = self.model.encode(query)
        query_embedding_list = query_embedding.tolist()
        
        # Search database
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    id,
                    filename,
                    page_number,
                    content,
                    chunk_index,
                    metadata,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM pdf_documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding_list, query_embedding_list, k))
            
            results = []
            for row in cur.fetchall():
                results.append({
                    'id': row[0],
                    'filename': row[1],
                    'page_number': row[2],
                    'content': row[3],
                    'chunk_index': row[4],
                    'metadata': row[5],
                    'similarity': float(row[6])
                })
            
            return results
    
    def generate_answer(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Generate answer using LLM with retrieved context"""
        if not self.llm_client:
            return "LLM client not configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY."
        
        # Prepare context text
        context_text = "\n\n".join([
            f"[{ctx['filename']}, Page {ctx['page_number']}]\n{ctx['content']}"
            for ctx in context
        ])
        
        # Create prompt
        system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Always cite the source (filename and page number) when answering.
If the context doesn't contain enough information to answer the question, say so."""
        
        user_prompt = f"""Context:
{context_text}

Question: {query}

Please answer the question based on the context provided above."""
        
        # Generate response
        try:
            if self.llm_provider == "openai":
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                return response.choices[0].message.content
            
            elif self.llm_provider == "anthropic":
                response = self.llm_client.messages.create(
                    model=self.llm_model,
                    max_tokens=500,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return response.content[0].text
        
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def ask(self, query: str, k: int = 5, return_context: bool = False):
        """Ask a question and get an answer"""
        # Retrieve context
        context = self.retrieve_context(query, k=k)
        
        if not context:
            return "No relevant context found in the documents."
        
        # Generate answer
        answer = self.generate_answer(query, context)
        
        if return_context:
            return {
                'answer': answer,
                'context': context
            }
        
        return answer
    
    def interactive_chat(self):
        """Interactive chat interface"""
        print("\n" + "=" * 70)
        print("  RAG Chatbot - Ask questions about your documents")
        print("  Type 'exit' or 'quit' to end the conversation")
        print("=" * 70 + "\n")
        
        while True:
            try:
                query = input("\nYou: ").strip()
                
                if query.lower() in ['exit', 'quit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if not query:
                    continue
                
                print("\nSearching documents...", end='', flush=True)
                result = self.ask(query, k=5, return_context=True)
                print("\r" + " " * 30 + "\r", end='')  # Clear "Searching..." message
                
                print(f"Assistant: {result['answer']}\n")
                
                # Show sources
                print("Sources:")
                for i, ctx in enumerate(result['context'][:3], 1):
                    print(f"  {i}. {ctx['filename']}, Page {ctx['page_number']} "
                          f"(similarity: {ctx['similarity']:.3f})")
            
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
    
    def close(self):
        self.conn.close()


def create_sample_pdf_content():
    """Create sample text files (simulating PDF content)"""
    sample_dir = Path('sample_pdfs')
    sample_dir.mkdir(exist_ok=True)
    
    documents = {
        'ml_basics.txt': """
Machine Learning Fundamentals

Introduction to Machine Learning
Machine learning is a branch of artificial intelligence that enables systems to learn 
and improve from experience without being explicitly programmed. It focuses on 
developing computer programs that can access data and use it to learn for themselves.

Types of Machine Learning
1. Supervised Learning: The algorithm learns from labeled training data
2. Unsupervised Learning: The algorithm finds patterns in unlabeled data
3. Reinforcement Learning: The algorithm learns through trial and error

Applications
Machine learning is used in various domains including healthcare, finance, 
marketing, and autonomous systems.
""",
        'database_intro.txt': """
Database Systems Overview

What is a Database?
A database is an organized collection of structured information or data stored 
electronically. Modern databases use database management systems (DBMS) to 
manage and query data.

Types of Databases
- Relational Databases (PostgreSQL, MySQL)
- NoSQL Databases (MongoDB, Cassandra)
- Vector Databases (for similarity search)
- Time-series Databases (InfluxDB, TimescaleDB)

PostgreSQL Features
PostgreSQL is a powerful, open-source relational database system known for its 
reliability, robustness, and performance. It supports advanced features like 
JSON storage, full-text search, and custom extensions.
"""
    }
    
    for filename, content in documents.items():
        filepath = sample_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f"✓ Created {len(documents)} sample documents in {sample_dir}/")
    print(f"  Note: These are text files simulating PDF content")
    return sample_dir


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='NeuronDB RAG Chatbot Example')
    parser.add_argument('command', choices=['ingest', 'chat', 'query', 'demo'],
                       help='Command to run')
    parser.add_argument('--input-dir', type=str, default='sample_pdfs',
                       help='Input directory for PDF files')
    parser.add_argument('--query', type=str, help='Question to ask')
    parser.add_argument('--llm', type=str, default='openai',
                       choices=['openai', 'anthropic'],
                       help='LLM provider')
    
    args = parser.parse_args()
    
    if args.command == 'demo':
        print("=" * 70)
        print("  NeuronDB RAG Chatbot Demo")
        print("=" * 70)
        print()
        print("Note: This demo uses text files instead of PDFs for simplicity")
        print()
        
        # Create sample documents
        sample_dir = create_sample_pdf_content()
        print()
        
        # Ingest documents  
        print("=" * 70)
        print("  Step 1: Ingesting Documents")
        print("=" * 70)
        ingester = PDFIngester(DB_CONFIG)
        # For demo, we'll ingest text files
        for txt_file in sample_dir.glob('*.txt'):
            print(f"Processing: {txt_file.name}")
            with open(txt_file) as f:
                content = f.read()
            
            chunks = ingester.chunk_text(content)
            print(f"  Created {len(chunks)} chunks")
            
            with ingester.conn.cursor() as cur:
                for idx, chunk in enumerate(chunks):
                    embedding = ingester.model.encode(chunk)
                    embedding_list = embedding.tolist()
                    
                    cur.execute("""
                        INSERT INTO pdf_documents 
                        (filename, page_number, content, embedding, chunk_index, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (txt_file.name, 1, chunk, embedding_list, idx, json.dumps({})))
            
            ingester.conn.commit()
            print(f"✓ Ingested {txt_file.name}")
        
        ingester.close()
        print()
        
        # Query example
        print("=" * 70)
        print("  Step 2: Question Answering")
        print("=" * 70)
        print()
        
        chatbot = RAGChatbot(DB_CONFIG, llm_provider=args.llm)
        
        queries = [
            "What is machine learning?",
            "What are the types of databases mentioned?"
        ]
        
        for query in queries:
            print(f"\nQuery: \"{query}\"")
            print("-" * 70)
            
            result = chatbot.ask(query, k=3, return_context=True)
            print(f"Answer: {result['answer']}\n")
            
            print("Sources:")
            for i, ctx in enumerate(result['context'], 1):
                print(f"  {i}. {ctx['filename']} (similarity: {ctx['similarity']:.3f})")
        
        chatbot.close()
        print()
        print("=" * 70)
        print("  Demo Complete!")
        print("=" * 70)
    
    elif args.command == 'ingest':
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Directory not found: {input_dir}")
            sys.exit(1)
        
        ingester = PDFIngester(DB_CONFIG)
        ingester.ingest_directory(input_dir)
        ingester.close()
    
    elif args.command == 'chat':
        chatbot = RAGChatbot(DB_CONFIG, llm_provider=args.llm)
        chatbot.interactive_chat()
        chatbot.close()
    
    elif args.command == 'query':
        if not args.query:
            print("Error: --query is required for query command")
            sys.exit(1)
        
        chatbot = RAGChatbot(DB_CONFIG, llm_provider=args.llm)
        result = chatbot.ask(args.query, k=5, return_context=True)
        
        print(f"\nQuestion: {args.query}")
        print("=" * 70)
        print(f"\nAnswer: {result['answer']}\n")
        print("Sources:")
        for i, ctx in enumerate(result['context'], 1):
            print(f"\n{i}. {ctx['filename']}, Page {ctx['page_number']}")
            print(f"   Similarity: {ctx['similarity']:.3f}")
            print(f"   {ctx['content'][:200]}...")
        
        chatbot.close()


if __name__ == '__main__':
    main()



