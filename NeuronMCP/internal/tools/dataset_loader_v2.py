#!/usr/bin/env python3
"""
Enhanced Comprehensive Dataset Loader for NeuronMCP
===================================================

Modular, extensible dataset loader supporting:
- All data sources (HuggingFace, URLs, GitHub, S3, Azure, GCS, FTP, databases, local)
- All formats (CSV, JSON, Parquet, Excel, HDF5, Avro, ORC, Feather, XML, HTML, etc.)
- Data transformations (filtering, mapping, validation, cleaning)
- Schema detection and inference
- Auto-embedding with NeuronDB
- Index creation
- Incremental loading with checkpoints
- Data quality validation

Usage:
    python3 dataset_loader_v2.py --source-type huggingface --source-path "dataset/name"
    python3 dataset_loader_v2.py --source-type s3 --source-path "s3://bucket/data.parquet"
    python3 dataset_loader_v2.py --source-type local --source-path "/path/to/data.csv" --format csv
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Import modular components
sys.path.insert(0, os.path.dirname(__file__))
from data_loaders.enhanced_loader import EnhancedDatasetLoader


def parse_json_arg(arg_str: str) -> dict:
    """Parse JSON string argument"""
    if not arg_str:
        return {}
    try:
        return json.loads(arg_str)
    except json.JSONDecodeError:
        # Try as a file path
        if os.path.exists(arg_str):
            with open(arg_str, 'r') as f:
                return json.load(f)
        raise ValueError(f"Invalid JSON: {arg_str}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Enhanced dataset loader for NeuronMCP with comprehensive capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Source configuration
    parser.add_argument('--source-type', required=True,
                       choices=['huggingface', 'url', 'github', 's3', 'azure', 'gcs', 'gs',
                                'ftp', 'sftp', 'local', 'database', 'db', 'postgresql', 'mysql', 'sqlite'],
                       help='Data source type')
    parser.add_argument('--source-path', required=True,
                       help='Dataset identifier/path (HF name, URL, file path, S3 path, etc.)')
    
    # HuggingFace specific
    parser.add_argument('--split', default='train',
                       help='Dataset split for HuggingFace')
    parser.add_argument('--config', help='Dataset config name for HuggingFace')
    
    # Format and compression
    parser.add_argument('--format', default='auto',
                       choices=['auto', 'csv', 'json', 'jsonl', 'parquet', 'excel', 'xlsx', 'xls',
                                'hdf5', 'h5', 'avro', 'orc', 'feather', 'xml', 'html', 'tsv'],
                       help='File format (auto-detect if not specified)')
    parser.add_argument('--compression', default='auto',
                       choices=['auto', 'gzip', 'bz2', 'xz', 'zip', 'none'],
                       help='Compression type')
    parser.add_argument('--encoding', default='auto',
                       help='File encoding (auto-detect if not specified)')
    
    # CSV specific options
    parser.add_argument('--csv-delimiter', dest='csv_delimiter',
                       help='CSV delimiter (auto-detect if not specified)')
    parser.add_argument('--csv-header', dest='csv_header', type=int, default=0,
                       help='Row to use as header (0 for first row, None for no header)')
    parser.add_argument('--csv-skip-rows', dest='csv_skip_rows', type=int, default=0,
                       help='Number of rows to skip at start')
    
    # Excel specific options
    parser.add_argument('--excel-sheet', dest='excel_sheet', default=0,
                       help='Excel sheet name or index')
    
    # Loading options
    parser.add_argument('--limit', type=int, default=0,
                       help='Maximum rows to load (0 for unlimited)')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for loading')
    parser.add_argument('--streaming', action='store_true',
                       help='Enable streaming mode')
    parser.add_argument('--no-streaming', dest='streaming', action='store_false',
                       help='Disable streaming mode')
    parser.set_defaults(streaming=None)
    
    # Table configuration
    parser.add_argument('--schema-name', default='datasets',
                       help='PostgreSQL schema name')
    parser.add_argument('--table-name', help='Custom table name')
    parser.add_argument('--if-exists', default='fail',
                       choices=['fail', 'replace', 'append'],
                       help='What to do if table exists')
    parser.add_argument('--load-mode', dest='load_mode', default='insert',
                       choices=['insert', 'append', 'upsert'],
                       help='Data loading mode')
    
    # Embedding options
    parser.add_argument('--auto-embed', action='store_true', default=True,
                       help='Automatically generate embeddings')
    parser.add_argument('--no-auto-embed', dest='auto_embed', action='store_false',
                       help='Disable automatic embedding generation')
    parser.add_argument('--embedding-model', default='default',
                       help='Embedding model name')
    parser.add_argument('--embedding-dimension', type=int, default=384,
                       help='Embedding vector dimension')
    parser.add_argument('--text-columns', nargs='+',
                       help='Specific text columns to embed')
    
    # Index options
    parser.add_argument('--create-indexes', action='store_true', default=True,
                       help='Create indexes automatically')
    parser.add_argument('--no-create-indexes', dest='create_indexes', action='store_false',
                       help='Disable automatic index creation')
    
    # Transformations (JSON config)
    parser.add_argument('--transformations', type=parse_json_arg,
                       help='JSON string or file path with transformation configuration')
    
    # Cloud credentials
    parser.add_argument('--aws-access-key', help='AWS access key ID')
    parser.add_argument('--aws-secret-key', help='AWS secret access key')
    parser.add_argument('--aws-region', help='AWS region')
    parser.add_argument('--azure-connection-string', help='Azure storage connection string')
    parser.add_argument('--gcs-credentials', help='GCS credentials file path')
    parser.add_argument('--github-token', help='GitHub personal access token')
    
    # Cache configuration
    parser.add_argument('--cache-dir', dest='cache_dir',
                       help='Cache directory path')
    
    # Database query (for database sources)
    parser.add_argument('--query', help='SQL query for database sources')
    
    # Incremental loading
    parser.add_argument('--checkpoint-key', help='Checkpoint key for incremental loading')
    parser.add_argument('--use-checkpoint', action='store_true',
                       help='Use checkpoint if available')
    
    args = parser.parse_args()
    
    # Build configuration dictionary
    config = {
        'source_type': args.source_type,
        'source_path': args.source_path,
        'limit': args.limit,
        'format': args.format,
        'compression': args.compression if args.compression != 'none' else None,
        'encoding': args.encoding,
        'streaming': args.streaming,
    }
    
    # Source-specific config
    if args.source_type == 'huggingface':
        config['split'] = args.split
        if args.config:
            config['config'] = args.config
    
    # Format-specific options
    format_options = {}
    
    csv_options = {}
    if args.csv_delimiter:
        csv_options['delimiter'] = args.csv_delimiter
    if args.csv_header is not None:
        csv_options['header'] = args.csv_header
    if args.csv_skip_rows:
        csv_options['skip_rows'] = args.csv_skip_rows
    
    if csv_options:
        format_options['csv_options'] = csv_options
    
    excel_options = {}
    if args.excel_sheet:
        excel_options['sheet_name'] = args.excel_sheet
    
    if excel_options:
        format_options['excel_options'] = excel_options
    
    config.update(format_options)
    
    # Cloud credentials
    if args.aws_access_key:
        config['aws_access_key'] = args.aws_access_key
    if args.aws_secret_key:
        config['aws_secret_key'] = args.aws_secret_key
    if args.aws_region:
        config['aws_region'] = args.aws_region
    if args.azure_connection_string:
        config['azure_connection_string'] = args.azure_connection_string
    if args.gcs_credentials:
        config['gcs_credentials'] = args.gcs_credentials
    if args.github_token:
        config['github_token'] = args.github_token
    
    # Transformations
    if args.transformations:
        config['transformations'] = args.transformations
    
    # Get database config from environment
    db_config = {
        'host': os.getenv('PGHOST', 'localhost'),
        'port': int(os.getenv('PGPORT', '5432')),
        'user': os.getenv('PGUSER', 'postgres'),
        'password': os.getenv('PGPASSWORD', ''),
        'database': os.getenv('PGDATABASE', 'postgres')
    }
    
    # Override cache directory if provided
    if args.cache_dir:
        os.environ['HF_HOME'] = args.cache_dir
        os.environ['HF_DATASETS_CACHE'] = f"{args.cache_dir}/datasets"
        os.makedirs(args.cache_dir, exist_ok=True)
        os.makedirs(f"{args.cache_dir}/datasets", exist_ok=True)
    
    try:
        loader = EnhancedDatasetLoader(db_config)
        loader.connect()
        
        # Generate table name if not provided
        if not args.table_name:
            if args.source_type == 'huggingface':
                table_name = args.source_path.replace('/', '_').replace('-', '_')
            else:
                table_name = Path(args.source_path).stem.replace('.', '_')
            table_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name)
        else:
            table_name = args.table_name
        
        # Check for checkpoint
        checkpoint = None
        if args.use_checkpoint and args.checkpoint_key:
            checkpoint = loader.get_checkpoint(args.checkpoint_key)
            if checkpoint:
                print(json.dumps({
                    "status": "checkpoint_found",
                    "checkpoint": checkpoint
                }), flush=True)
        
        # Load dataset
        print(json.dumps({"status": "loading_dataset", "source_type": args.source_type}), flush=True)
        df = loader.load_data(config)
        
        # Override text columns if specified
        if args.text_columns:
            loader.text_columns = args.text_columns
        
        # Create table
        print(json.dumps({"status": "creating_table"}), flush=True)
        loader.create_table(
            df, args.schema_name, table_name,
            args.auto_embed, args.create_indexes, args.if_exists
        )
        
        # Load data
        print(json.dumps({"status": "loading_data", "total_rows": len(df)}), flush=True)
        rows_loaded = loader.load_data_batch(
            df, args.schema_name, table_name,
            args.batch_size, args.load_mode
        )
        
        # Generate embeddings
        rows_embedded = 0
        if args.auto_embed and loader.text_columns:
            print(json.dumps({"status": "generating_embeddings"}), flush=True)
            rows_embedded = loader.generate_embeddings(
                args.schema_name, table_name, args.embedding_model
            )
        
        # Create indexes
        indexes_created = []
        if args.create_indexes:
            print(json.dumps({"status": "creating_indexes"}), flush=True)
            indexes_created = loader.create_indexes(args.schema_name, table_name)
        
        # Create checkpoint if requested
        if args.checkpoint_key:
            checkpoint_metadata = {
                'rows_loaded': rows_loaded,
                'rows_embedded': rows_embedded,
                'table': f"{args.schema_name}.{table_name}",
                'timestamp': str(pd.Timestamp.now())
            }
            loader.create_checkpoint(args.checkpoint_key, checkpoint_metadata)
        
        # Final result
        result = {
            "status": "success",
            "rows_loaded": rows_loaded,
            "rows_embedded": rows_embedded,
            "table": f"{args.schema_name}.{table_name}",
            "text_columns": loader.text_columns,
            "embedding_columns": loader.embedding_columns,
            "indexes_created": len(indexes_created),
            "indexes": indexes_created
        }
        
        print(json.dumps(result), flush=True)
        loader.close()
        
    except Exception as e:
        error_result = {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }
        import traceback
        error_result["traceback"] = traceback.format_exc()
        print(json.dumps(error_result), file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

