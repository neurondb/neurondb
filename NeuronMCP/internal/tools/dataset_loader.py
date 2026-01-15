#!/usr/bin/env python3
"""
Comprehensive Dataset Loader for NeuronMCP
==========================================

Loads datasets from multiple sources (HuggingFace, URLs, GitHub, S3, local files)
into PostgreSQL with automatic schema detection, embedding generation, and index creation.

Usage:
    python3 dataset_loader.py --source-type huggingface --source-path "dataset/name" --auto-embed
    python3 dataset_loader.py --source-type url --source-path "https://example.com/data.csv"
    python3 dataset_loader.py --source-type local --source-path "/path/to/file.json"
"""

import os
import sys
import json
import argparse
import tempfile
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

try:
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extensions import quote_ident
    import psycopg2.extras
except ImportError as e:
    print(json.dumps({
        "error": f"Required Python package 'psycopg2' not installed: {e}",
        "status": "error",
        "hint": "Install with: pip install psycopg2-binary"
    }), file=sys.stderr)
    sys.exit(1)

try:
    import pandas as pd
except ImportError as e:
    print(json.dumps({
        "error": f"Required Python package 'pandas' not installed: {e}",
        "status": "error",
        "hint": "Install with: pip install pandas"
    }), file=sys.stderr)
    sys.exit(1)

# Optional imports
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False


class DatasetLoader:
    """Comprehensive dataset loader with schema detection and embedding generation"""
    
    def __init__(self, db_config: Dict[str, Any]):
        """Initialize loader with database configuration"""
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        self.schema_name = "datasets"
        self.table_name = None
        self.text_columns = []
        self.embedding_columns = []
        self.schema_info = {}
        
    def connect(self):
        """Connect to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(
                host=self.db_config.get('host', 'localhost'),
                port=int(self.db_config.get('port', 5432)),
                user=self.db_config.get('user', 'postgres'),
                password=self.db_config.get('password', ''),
                database=self.db_config.get('database', 'postgres')
            )
            self.cursor = self.conn.cursor()
        except Exception as e:
            raise Exception(f"Failed to connect to database: {e}")
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
    
    def _get_hf_token_from_db(self) -> Optional[str]:
        """Get HuggingFace API key from PostgreSQL GUC variable"""
        if not self.conn:
            return None
        
        try:
            # Query PostgreSQL for the neurondb.llm_api_key GUC variable
            # The 'true' parameter means return NULL if not set instead of raising an error
            self.cursor.execute("SELECT current_setting('neurondb.llm_api_key', true)")
            result = self.cursor.fetchone()
            if result and result[0]:
                token = result[0].strip()
                if token:
                    return token
        except Exception as e:
            # Log but don't fail - dataset might be public
            print(json.dumps({
                "warning": f"Could not read HuggingFace API key from PostgreSQL: {e}",
                "status": "info"
            }), file=sys.stderr, flush=True)
        
        return None
    
    def load_from_huggingface(self, dataset_name: str, split: str = "train", 
                              limit: int = 0, streaming: bool = True, config: Optional[str] = None) -> pd.DataFrame:
        """Load dataset from HuggingFace"""
        if not HAS_DATASETS:
            raise Exception("datasets library not available. Install with: pip install datasets")
        
        try:
            # Get HuggingFace API key from PostgreSQL if available
            hf_token = self._get_hf_token_from_db()
            if hf_token:
                # Set environment variables for HuggingFace authentication
                os.environ['HF_TOKEN'] = hf_token
                os.environ['HUGGINGFACE_HUB_TOKEN'] = hf_token
                print(json.dumps({
                    "status": "info",
                    "message": "Using HuggingFace API key from PostgreSQL configuration"
                }), flush=True)
            else:
                print(json.dumps({
                    "status": "info",
                    "message": "No HuggingFace API key found in PostgreSQL. Using public access (if dataset is public)"
                }), flush=True)
            
            # Set cache directory
            cache_dir = os.environ.get('HF_HOME', '/tmp/hf_cache')
            os.makedirs(cache_dir, exist_ok=True)
            os.makedirs(f"{cache_dir}/datasets", exist_ok=True)
            
            # Load dataset with optional config parameter
            load_kwargs = {}
            if config:
                load_kwargs['config_name'] = config
            
            if streaming:
                try:
                    dataset = load_dataset(dataset_name, split=split, streaming=True, **load_kwargs)
                    data = []
                    count = 0
                    for item in dataset:
                        if isinstance(item, dict):
                            data.append(item)
                        else:
                            # Convert to dict
                            data.append(dict(item))
                        count += 1
                        if limit > 0 and count >= limit:
                            break
                    df = pd.DataFrame(data)
                except Exception:
                    # Fallback to non-streaming
                    dataset = load_dataset(dataset_name, split=split, streaming=False, **load_kwargs)
                    if limit > 0:
                        dataset = dataset.select(range(min(limit, len(dataset))))
                    df = pd.DataFrame(dataset)
            else:
                dataset = load_dataset(dataset_name, split=split, streaming=False, **load_kwargs)
                if limit > 0:
                    dataset = dataset.select(range(min(limit, len(dataset))))
                df = pd.DataFrame(dataset)
            
            return df
        except Exception as e:
            raise Exception(f"Failed to load HuggingFace dataset '{dataset_name}': {e}")
    
    def load_from_url(self, url: str, format: str = "auto", limit: int = 0) -> pd.DataFrame:
        """Load dataset from URL with compression support"""
        if not HAS_REQUESTS:
            raise Exception("requests library not available. Install with: pip install requests")
        
        try:
            # Download file
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Detect compression from URL or Content-Encoding header
            compression = None
            if url.endswith('.gz') or url.endswith('.gzip'):
                compression = 'gzip'
            elif url.endswith('.bz2'):
                compression = 'bz2'
            elif url.endswith('.xz'):
                compression = 'xz'
            elif url.endswith('.zip'):
                compression = 'zip'
            elif 'gzip' in response.headers.get('content-encoding', ''):
                compression = 'gzip'
            
            # Remove compression extension from format detection
            base_url = url
            if compression:
                for ext in ['.gz', '.gzip', '.bz2', '.xz', '.zip']:
                    base_url = base_url.replace(ext, '')
            
            # Detect format from URL or content
            if format == "auto":
                if base_url.endswith('.csv'):
                    format = "csv"
                elif base_url.endswith('.json') or base_url.endswith('.jsonl'):
                    format = "json" if base_url.endswith('.json') else "jsonl"
                elif base_url.endswith('.parquet'):
                    format = "parquet"
                else:
                    # Try to detect from content-type
                    content_type = response.headers.get('content-type', '')
                    if 'csv' in content_type:
                        format = "csv"
                    elif 'json' in content_type:
                        format = "json"
                    else:
                        format = "csv"  # Default to CSV
            
            # Save to temporary file
            suffix = f".{format}"
            if compression:
                suffix += f".{compression}"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp.write(chunk)
                tmp_path = tmp.name
            
            try:
                # Load based on format with compression support
                if format == "csv":
                    df = pd.read_csv(tmp_path, compression=compression)
                elif format == "json":
                    df = pd.read_json(tmp_path, compression=compression)
                elif format == "jsonl":
                    df = pd.read_json(tmp_path, lines=True, compression=compression)
                elif format == "parquet":
                    if not HAS_PARQUET:
                        raise Exception("pyarrow not available for Parquet support")
                    df = pd.read_parquet(tmp_path)
                else:
                    raise Exception(f"Unsupported format: {format}")
                
                if limit > 0:
                    df = df.head(limit)
                
                return df
            finally:
                os.unlink(tmp_path)
        except Exception as e:
            raise Exception(f"Failed to load dataset from URL '{url}': {e}")
    
    def load_from_github(self, repo_path: str, file_path: str = None, 
                         limit: int = 0) -> pd.DataFrame:
        """Load dataset from GitHub repository"""
        if not HAS_REQUESTS:
            raise Exception("requests library not available")
        
        try:
            # Parse GitHub URL or repo path
            # Format: owner/repo or owner/repo/path/to/file
            if '/' in repo_path:
                parts = repo_path.split('/', 2)
                if len(parts) >= 2:
                    owner, repo = parts[0], parts[1]
                    file_path = parts[2] if len(parts) > 2 else file_path
                else:
                    raise Exception("Invalid GitHub path format. Use: owner/repo or owner/repo/path/to/file")
            else:
                raise Exception("Invalid GitHub path format")
            
            # Try to get file from GitHub API
            if file_path:
                api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
            else:
                # Try to find common data files
                common_files = ['data.csv', 'data.json', 'dataset.csv', 'dataset.json']
                for cf in common_files:
                    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{cf}"
                    try:
                        response = requests.get(api_url, timeout=10)
                        if response.status_code == 200:
                            file_path = cf
                            break
                    except:
                        continue
                else:
                    raise Exception(f"Could not find data file in repository {owner}/{repo}")
            
            # Get file content
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            file_data = response.json()
            
            if 'download_url' in file_data:
                # Use download URL
                return self.load_from_url(file_data['download_url'], limit=limit)
            elif 'content' in file_data:
                # Decode base64 content
                import base64
                content = base64.b64decode(file_data['content']).decode('utf-8')
                
                # Detect format
                from io import StringIO
                if file_path.endswith('.csv'):
                    df = pd.read_csv(StringIO(content))
                elif file_path.endswith('.json') or file_path.endswith('.jsonl'):
                    df = pd.read_json(StringIO(content), lines=file_path.endswith('.jsonl'))
                else:
                    raise Exception(f"Unsupported file format: {file_path}")
                
                if limit > 0:
                    df = df.head(limit)
                
                return df
            else:
                raise Exception("Could not retrieve file content from GitHub")
        except Exception as e:
            raise Exception(f"Failed to load dataset from GitHub '{repo_path}': {e}")
    
    def load_from_s3(self, s3_path: str, limit: int = 0) -> pd.DataFrame:
        """Load dataset from S3"""
        if not HAS_BOTO3:
            raise Exception("boto3 library not available. Install with: pip install boto3")
        
        try:
            # Parse S3 path: s3://bucket/key
            if not s3_path.startswith('s3://'):
                raise Exception("S3 path must start with 's3://'")
            
            path_parts = s3_path[5:].split('/', 1)
            bucket = path_parts[0]
            key = path_parts[1] if len(path_parts) > 1 else None
            
            if not key:
                raise Exception("S3 key is required")
            
            # Create S3 client
            s3_client = boto3.client('s3')
            
            # Download to temporary file
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                s3_client.download_fileobj(bucket, key, tmp)
                tmp_path = tmp.name
            
            try:
                # Detect format from key
                if key.endswith('.csv'):
                    df = pd.read_csv(tmp_path)
                elif key.endswith('.json'):
                    df = pd.read_json(tmp_path)
                elif key.endswith('.jsonl'):
                    df = pd.read_json(tmp_path, lines=True)
                elif key.endswith('.parquet'):
                    if not HAS_PARQUET:
                        raise Exception("pyarrow not available for Parquet support")
                    df = pd.read_parquet(tmp_path)
                else:
                    # Try CSV first
                    try:
                        df = pd.read_csv(tmp_path)
                    except:
                        df = pd.read_json(tmp_path)
                
                if limit > 0:
                    df = df.head(limit)
                
                return df
            finally:
                os.unlink(tmp_path)
        except Exception as e:
            raise Exception(f"Failed to load dataset from S3 '{s3_path}': {e}")
    
    def load_from_local(self, file_path: str, limit: int = 0) -> pd.DataFrame:
        """Load dataset from local file with compression support"""
        try:
            if not os.path.exists(file_path):
                raise Exception(f"File not found: {file_path}")
            
            # Detect compression
            compression = None
            if file_path.endswith('.gz') or file_path.endswith('.gzip'):
                compression = 'gzip'
            elif file_path.endswith('.bz2'):
                compression = 'bz2'
            elif file_path.endswith('.xz'):
                compression = 'xz'
            elif file_path.endswith('.zip'):
                compression = 'zip'
            
            # Remove compression extension for format detection
            base_path = file_path
            if compression:
                for ext in ['.gz', '.gzip', '.bz2', '.xz', '.zip']:
                    base_path = base_path.replace(ext, '')
            
            # Detect format from extension
            if base_path.endswith('.csv'):
                df = pd.read_csv(file_path, compression=compression)
            elif base_path.endswith('.json'):
                df = pd.read_json(file_path, compression=compression)
            elif base_path.endswith('.jsonl'):
                df = pd.read_json(file_path, lines=True, compression=compression)
            elif base_path.endswith('.parquet'):
                if not HAS_PARQUET:
                    raise Exception("pyarrow not available for Parquet support")
                df = pd.read_parquet(file_path)
            else:
                # Try CSV first, then JSON
                try:
                    df = pd.read_csv(file_path, compression=compression)
                except:
                    try:
                        df = pd.read_json(file_path, compression=compression)
                    except:
                        raise Exception(f"Could not determine file format for '{file_path}'")
            
            if limit > 0:
                df = df.head(limit)
            
            return df
        except Exception as e:
            raise Exception(f"Failed to load local file '{file_path}': {e}")
    
    def detect_schema(self, df: pd.DataFrame, sample_size: int = 1000) -> Dict[str, str]:
        """Detect PostgreSQL schema from DataFrame"""
        schema = {}
        
        # Sample data for better type detection
        sample_df = df.head(min(sample_size, len(df))) if len(df) > sample_size else df
        
        for col in df.columns:
            col_type = str(df[col].dtype)
            
            # Map pandas types to PostgreSQL types
            if col_type.startswith('int'):
                if df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                    schema[col] = "INTEGER"
                else:
                    schema[col] = "BIGINT"
            elif col_type.startswith('float'):
                schema[col] = "DOUBLE PRECISION"
            elif col_type == 'bool':
                schema[col] = "BOOLEAN"
            elif col_type.startswith('datetime'):
                schema[col] = "TIMESTAMP"
            elif col_type == 'object':
                # Check if it's actually text or JSON
                sample_values = sample_df[col].dropna().head(10)
                if len(sample_values) > 0:
                    first_val = sample_values.iloc[0]
                    if isinstance(first_val, (dict, list)):
                        schema[col] = "JSONB"
                    elif isinstance(first_val, str):
                        # Check if it's a long text (likely needs TEXT)
                        max_len = sample_df[col].astype(str).str.len().max()
                        if max_len > 255:
                            schema[col] = "TEXT"
                        else:
                            schema[col] = f"VARCHAR({int(max_len * 1.5)})"
                    else:
                        schema[col] = "TEXT"
                else:
                    schema[col] = "TEXT"
            else:
                schema[col] = "TEXT"
        
        self.schema_info = schema
        return schema
    
    def detect_text_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect text columns that should have embeddings"""
        text_cols = []
        
        for col in df.columns:
            col_type = str(df[col].dtype)
            
            # Check if column is text-like
            if col_type == 'object':
                sample_values = df[col].dropna().head(100)
                if len(sample_values) > 0:
                    # Check if values are strings (not dicts/lists)
                    first_val = sample_values.iloc[0]
                    if isinstance(first_val, str) and len(first_val) > 10:
                        # Check if it looks like text (not just IDs or codes)
                        avg_len = sample_values.astype(str).str.len().mean()
                        if avg_len > 20:  # Likely text content
                            text_cols.append(col)
        
        self.text_columns = text_cols
        return text_cols
    
    def create_table(self, df: pd.DataFrame, schema_name: str, table_name: str,
                     auto_embed: bool = True, create_indexes: bool = True):
        """Create optimized PostgreSQL table with schema detection"""
        self.schema_name = schema_name
        self.table_name = table_name
        
        # Detect schema
        pg_schema = self.detect_schema(df)
        
        # Detect text columns
        if auto_embed:
            text_cols = self.detect_text_columns(df)
            self.text_columns = text_cols
        
        # Create schema
        schema_quoted = quote_ident(schema_name, self.cursor)
        self.cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_quoted}")
        
        # Build CREATE TABLE statement
        columns = []
        
        # Check if DataFrame already has an 'id' column
        has_id_column = 'id' in df.columns
        
        if not has_id_column:
            # Only add SERIAL id if DataFrame doesn't have one
            columns.append("id SERIAL PRIMARY KEY")
        
        for col_name, col_type in pg_schema.items():
            col_quoted = quote_ident(col_name, self.cursor)
            # If this is the 'id' column and we didn't add SERIAL id, use it as PRIMARY KEY
            if col_name.lower() == 'id' and has_id_column:
                columns.append(f"{col_quoted} {col_type} PRIMARY KEY")
            else:
                columns.append(f"{col_quoted} {col_type}")
        
        # Add embedding columns for text columns
        if auto_embed and self.text_columns:
            for col in self.text_columns:
                embed_col = f"{col}_embedding"
                self.embedding_columns.append(embed_col)
                embed_col_quoted = quote_ident(embed_col, self.cursor)
                columns.append(f"{embed_col_quoted} vector(384)")  # Default dimension, can be adjusted
        
        table_quoted = quote_ident(table_name, self.cursor)
        create_sql = f"CREATE TABLE IF NOT EXISTS {schema_quoted}.{table_quoted} ({', '.join(columns)})"
        
        self.cursor.execute(create_sql)
        self.conn.commit()
        
        return pg_schema
    
    def load_data_batch(self, df: pd.DataFrame, schema_name: str, table_name: str,
                       batch_size: int = 1000):
        """Load data in batches using COPY for efficiency"""
        schema_quoted = quote_ident(schema_name, self.cursor)
        table_quoted = quote_ident(table_name, self.cursor)
        
        total_rows = len(df)
        inserted = 0
        
        # Check if table has SERIAL id column (not provided in data)
        # Query the table structure to see if id is SERIAL
        check_serial_sql = f"""
            SELECT column_default 
            FROM information_schema.columns 
            WHERE table_schema = %s 
            AND table_name = %s 
            AND column_name = 'id'
        """
        self.cursor.execute(check_serial_sql, (schema_name, table_name))
        serial_result = self.cursor.fetchone()
        has_serial_id = serial_result and serial_result[0] and 'nextval' in str(serial_result[0])
        
        # Prepare column names - exclude 'id' if it's SERIAL and not in DataFrame
        columns = list(df.columns)
        if has_serial_id and 'id' not in columns:
            # Table has SERIAL id, DataFrame doesn't have id - that's fine
            pass
        elif 'id' in columns and has_serial_id:
            # Both have id - exclude from insert if it's SERIAL
            columns = [c for c in columns if c != 'id']
        
        col_names_quoted = [quote_ident(col, self.cursor) for col in columns]
        
        # Process in batches
        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            batch_df = df.iloc[start_idx:end_idx]
            
            # Convert DataFrame to list of tuples
            values = []
            for _, row in batch_df.iterrows():
                row_values = []
                for col in columns:
                    val = row[col]
                    if pd.isna(val):
                        row_values.append(None)
                    elif isinstance(val, (dict, list)):
                        row_values.append(json.dumps(val))
                    else:
                        row_values.append(str(val))
                values.append(tuple(row_values))
            
            # Use batch INSERT for reliability (COPY can be tricky with mixed types)
            # Build INSERT statement with proper value placeholders
            placeholders = ','.join(['%s'] * len(col_names_quoted))
            insert_sql = f"INSERT INTO {schema_quoted}.{table_quoted} ({', '.join(col_names_quoted)}) VALUES ({placeholders})"
            
            try:
                # Execute batch insert
                self.cursor.executemany(insert_sql, values)
                inserted += len(batch_df)
                self.conn.commit()
                
                # Progress update
                progress = inserted / total_rows if total_rows > 0 else 1.0
                print(json.dumps({
                    "status": "loading",
                    "progress": progress,
                    "rows_loaded": inserted,
                    "total_rows": total_rows,
                    "current_batch": start_idx // batch_size + 1
                }), flush=True)
            except Exception as e:
                # Rollback and try individual inserts for this batch
                self.conn.rollback()
                for val_tuple in values:
                    try:
                        self.cursor.execute(insert_sql, val_tuple)
                        inserted += 1
                    except Exception as insert_err:
                        # Log but continue
                        print(json.dumps({
                            "warning": f"Failed to insert row: {insert_err}",
                            "status": "loading"
                        }), file=sys.stderr, flush=True)
                self.conn.commit()
        
        return inserted
    
    def generate_embeddings(self, schema_name: str, table_name: str,
                           embedding_model: str = "default", batch_size: int = 100):
        """Generate embeddings for text columns using NeuronDB"""
        if not self.text_columns:
            return 0
        
        schema_quoted = quote_ident(schema_name, self.cursor)
        table_quoted = quote_ident(table_name, self.cursor)
        
        total_embedded = 0
        
        for text_col, embed_col in zip(self.text_columns, self.embedding_columns):
            text_col_quoted = quote_ident(text_col, self.cursor)
            embed_col_quoted = quote_ident(embed_col, self.cursor)
            
            # Get count of rows needing embeddings
            count_sql = f"SELECT COUNT(*) FROM {schema_quoted}.{table_quoted} WHERE {embed_col_quoted} IS NULL AND {text_col_quoted} IS NOT NULL"
            self.cursor.execute(count_sql)
            total_count = self.cursor.fetchone()[0]
            
            if total_count == 0:
                continue
            
            # Generate embeddings in batches
            offset = 0
            while offset < total_count:
                # Fetch batch of text values
                fetch_sql = f"""
                    SELECT id, {text_col_quoted} 
                    FROM {schema_quoted}.{table_quoted} 
                    WHERE {embed_col_quoted} IS NULL AND {text_col_quoted} IS NOT NULL
                    LIMIT {batch_size} OFFSET {offset}
                """
                self.cursor.execute(fetch_sql)
                rows = self.cursor.fetchall()
                
                if not rows:
                    break
                
                # Generate embeddings
                for row_id, text_val in rows:
                    if text_val and len(str(text_val).strip()) > 0:
                        try:
                            # Use NeuronDB embed_text function
                            embed_sql = f"SELECT embed_text(%s, %s)::text"
                            self.cursor.execute(embed_sql, (str(text_val), embedding_model))
                            embedding = self.cursor.fetchone()[0]
                            
                            # Update row with embedding
                            update_sql = f"UPDATE {schema_quoted}.{table_quoted} SET {embed_col_quoted} = %s::vector WHERE id = %s"
                            self.cursor.execute(update_sql, (embedding, row_id))
                            total_embedded += 1
                        except Exception as e:
                            # Log error but continue
                            print(json.dumps({
                                "warning": f"Failed to generate embedding for row {row_id}: {e}",
                                "status": "embedding"
                            }), file=sys.stderr, flush=True)
                
                self.conn.commit()
                offset += batch_size
                
                # Progress update
                progress = total_embedded / total_count if total_count > 0 else 1.0
                print(json.dumps({
                    "status": "embedding",
                    "progress": progress,
                    "rows_embedded": total_embedded,
                    "total_rows": total_count,
                    "column": text_col
                }), flush=True)
        
        return total_embedded
    
    def create_indexes(self, schema_name: str, table_name: str):
        """Create indexes for optimal query performance"""
        schema_quoted = quote_ident(schema_name, self.cursor)
        table_quoted = quote_ident(table_name, self.cursor)
        
        indexes_created = []
        
        # Create HNSW indexes for vector columns
        for embed_col in self.embedding_columns:
            embed_col_quoted = quote_ident(embed_col, self.cursor)
            index_name = f"{table_name}_{embed_col}_hnsw_idx"
            index_name_quoted = quote_ident(index_name, self.cursor)
            
            try:
                create_idx_sql = f"""
                    CREATE INDEX IF NOT EXISTS {index_name_quoted}
                    ON {schema_quoted}.{table_quoted}
                    USING hnsw ({embed_col_quoted} vector_cosine_ops)
                """
                self.cursor.execute(create_idx_sql)
                indexes_created.append(f"HNSW index on {embed_col}")
            except Exception as e:
                # HNSW might not be available, try regular index
                try:
                    create_idx_sql = f"CREATE INDEX IF NOT EXISTS {index_name_quoted} ON {schema_quoted}.{table_quoted} USING ivfflat ({embed_col_quoted})"
                    self.cursor.execute(create_idx_sql)
                    indexes_created.append(f"IVFFlat index on {embed_col}")
                except:
                    pass
        
        # Create GIN indexes for text columns (full-text search)
        for text_col in self.text_columns:
            text_col_quoted = quote_ident(text_col, self.cursor)
            index_name = f"{table_name}_{text_col}_gin_idx"
            index_name_quoted = quote_ident(index_name, self.cursor)
            
            try:
                create_idx_sql = f"CREATE INDEX IF NOT EXISTS {index_name_quoted} ON {schema_quoted}.{table_quoted} USING gin (to_tsvector('english', {text_col_quoted}))"
                self.cursor.execute(create_idx_sql)
                indexes_created.append(f"GIN index on {text_col}")
            except Exception as e:
                pass
        
        self.conn.commit()
        return indexes_created


def main():
    """Main entry point for dataset loader"""
    parser = argparse.ArgumentParser(description="Load datasets into PostgreSQL with NeuronDB")
    
    # Source configuration
    parser.add_argument('--source-type', required=True, 
                       choices=['huggingface', 'url', 'github', 's3', 'local'],
                       help='Data source type')
    parser.add_argument('--source-path', required=True,
                       help='Dataset identifier (HF name, URL, file path, etc.)')
    
    # HuggingFace specific
    parser.add_argument('--split', default='train',
                       help='Dataset split for HuggingFace')
    parser.add_argument('--config', help='Dataset config for HuggingFace datasets')
    
    # Cache configuration
    parser.add_argument('--cache-dir', dest='cache_dir',
                       help='Cache directory path for downloads (optional, defaults to /tmp/hf_cache)')
    
    # Loading options
    parser.add_argument('--limit', type=int, default=0,
                       help='Maximum rows to load (0 for unlimited)')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for loading')
    parser.add_argument('--streaming', action='store_true', default=True,
                       help='Use streaming mode')
    parser.add_argument('--format', default='auto',
                       choices=['csv', 'json', 'jsonl', 'parquet', 'auto'],
                       help='File format hint')
    
    # Table configuration
    parser.add_argument('--schema-name', default='datasets',
                       help='PostgreSQL schema name')
    parser.add_argument('--table-name', help='Custom table name')
    
    # Embedding options
    parser.add_argument('--auto-embed', action='store_true', default=True,
                       help='Automatically generate embeddings for text columns')
    parser.add_argument('--no-auto-embed', dest='auto_embed', action='store_false',
                       help='Disable automatic embedding generation')
    parser.add_argument('--embedding-model', default='default',
                       help='Embedding model name')
    parser.add_argument('--text-columns', nargs='+',
                       help='Specific text columns to embed')
    
    # Index options
    parser.add_argument('--create-indexes', action='store_true', default=True,
                       help='Create indexes automatically')
    parser.add_argument('--no-create-indexes', dest='create_indexes', action='store_false',
                       help='Disable automatic index creation')
    
    # Database configuration (from environment)
    args = parser.parse_args()
    
    # Override cache directory if provided
    if args.cache_dir:
        os.environ['HF_HOME'] = args.cache_dir
        os.environ['HF_DATASETS_CACHE'] = f"{args.cache_dir}/datasets"
        os.makedirs(args.cache_dir, exist_ok=True)
        os.makedirs(f"{args.cache_dir}/datasets", exist_ok=True)
    
    # Get database config from environment
    db_config = {
        'host': os.getenv('PGHOST', 'localhost'),
        'port': int(os.getenv('PGPORT', '5432')),
        'user': os.getenv('PGUSER', 'postgres'),
        'password': os.getenv('PGPASSWORD', ''),
        'database': os.getenv('PGDATABASE', 'postgres')
    }
    
    try:
        loader = DatasetLoader(db_config)
        loader.connect()
        
        # Generate table name if not provided
        if not args.table_name:
            if args.source_type == 'huggingface':
                table_name = args.source_path.replace('/', '_').replace('-', '_')
            else:
                table_name = Path(args.source_path).stem.replace('.', '_')
            args.table_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name)
        
        # Load dataset based on source type
        print(json.dumps({"status": "loading_dataset", "source_type": args.source_type}), flush=True)
        
        if args.source_type == 'huggingface':
            df = loader.load_from_huggingface(
                args.source_path, args.split, args.limit, args.streaming, args.config
            )
        elif args.source_type == 'url':
            df = loader.load_from_url(args.source_path, args.format, args.limit)
        elif args.source_type == 'github':
            df = loader.load_from_github(args.source_path, limit=args.limit)
        elif args.source_type == 's3':
            df = loader.load_from_s3(args.source_path, args.limit)
        elif args.source_type == 'local':
            df = loader.load_from_local(args.source_path, args.limit)
        else:
            raise Exception(f"Unsupported source type: {args.source_type}")
        
        # Override text columns if specified
        if args.text_columns:
            loader.text_columns = args.text_columns
        
        # Create table
        print(json.dumps({"status": "creating_table"}), flush=True)
        loader.create_table(df, args.schema_name, args.table_name, 
                           args.auto_embed, args.create_indexes)
        
        # Load data
        print(json.dumps({"status": "loading_data", "total_rows": len(df)}), flush=True)
        rows_loaded = loader.load_data_batch(df, args.schema_name, args.table_name, 
                                            args.batch_size)
        
        # Generate embeddings
        if args.auto_embed and loader.text_columns:
            print(json.dumps({"status": "generating_embeddings"}), flush=True)
            rows_embedded = loader.generate_embeddings(
                args.schema_name, args.table_name, args.embedding_model
            )
        else:
            rows_embedded = 0
        
        # Create indexes
        if args.create_indexes:
            print(json.dumps({"status": "creating_indexes"}), flush=True)
            indexes_created = loader.create_indexes(args.schema_name, args.table_name)
        else:
            indexes_created = []
        
        # Final result
        result = {
            "status": "success",
            "rows_loaded": rows_loaded,
            "rows_embedded": rows_embedded,
            "table": f"{args.schema_name}.{args.table_name}",
            "text_columns": loader.text_columns,
            "embedding_columns": loader.embedding_columns,
            "indexes_created": len(indexes_created)
        }
        
        print(json.dumps(result), flush=True)
        loader.close()
        
    except Exception as e:
        error_result = {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }
        print(json.dumps(error_result), file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

