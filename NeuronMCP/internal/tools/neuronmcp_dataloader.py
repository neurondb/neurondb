#!/usr/bin/env python3
"""
NeuronMCP Comprehensive Data Loader
====================================

Unified, modular dataset loading system with comprehensive capabilities:
- Multiple data sources: HuggingFace, URLs, GitHub, S3, Azure, GCS, FTP, databases, local files
- Multiple formats: CSV, JSON, Parquet, Excel, HDF5, Avro, ORC, Feather, XML, HTML, etc.
- Data transformations: filtering, mapping, validation, cleaning
- Schema detection and inference
- Auto-embedding with NeuronDB
- Index creation
- Incremental loading with checkpoints
- Data quality validation

Usage:
    python3 neuronmcp_dataloader.py --source-type huggingface --source-path "dataset/name"
    python3 neuronmcp_dataloader.py --source-type s3 --source-path "s3://bucket/data.parquet"
    python3 neuronmcp_dataloader.py --source-type local --source-path "/path/to/data.csv"
"""

import os
import sys
import json
import argparse
import tempfile
import re
import gzip
import bz2
import lzma
import zipfile
import csv
from typing import Dict, List, Any, Optional, Iterator, IO, Callable, Tuple
from pathlib import Path
from abc import ABC, abstractmethod
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
    import numpy as np
except ImportError as e:
    print(json.dumps({
        "error": f"Required Python package 'pandas' not installed: {e}",
        "status": "error",
        "hint": "Install with: pip install pandas"
    }), file=sys.stderr)
    sys.exit(1)

# Optional imports for various data sources and formats
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    import boto3
    from botocore.exceptions import ClientError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

try:
    from azure.storage.blob import BlobServiceClient
    HAS_AZURE = True
except ImportError:
    HAS_AZURE = False

try:
    from google.cloud import storage
    HAS_GCS = True
except ImportError:
    HAS_GCS = False

try:
    from ftplib import FTP
    import paramiko
    HAS_FTP = True
except ImportError:
    HAS_FTP = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import openpyxl
    import xlrd
    HAS_EXCEL = True
except ImportError:
    HAS_EXCEL = False

try:
    import h5py
    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False

try:
    import fastavro
    HAS_AVRO = True
except ImportError:
    HAS_AVRO = False

try:
    import pyarrow.parquet as pq
    import pyarrow.feather as pf
    HAS_ARROW = True
except ImportError:
    HAS_ARROW = False

try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================================================
# BASE LOADER INTERFACE
# ============================================================================

class BaseDataLoader(ABC):
    """Abstract base class for data source loaders"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize loader with configuration"""
        self.config = config
        self.source_path = config.get('source_path', '')
        self.limit = config.get('limit', 0)
        self.format_hint = config.get('format', 'auto')
        
    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load data and return as pandas DataFrame"""
        pass
    
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Check if this loader supports streaming mode"""
        pass
    
    def load_streaming(self) -> Iterator[pd.DataFrame]:
        """Load data in streaming mode"""
        if not self.supports_streaming():
            raise NotImplementedError(f"{self.__class__.__name__} does not support streaming mode")
        
        df = self.load()
        chunk_size = self.config.get('chunk_size', 1000)
        
        for i in range(0, len(df), chunk_size):
            yield df.iloc[i:i + chunk_size]
    
    def validate_config(self) -> bool:
        """Validate loader configuration"""
        if not self.source_path:
            raise ValueError("source_path is required")
        return True


# ============================================================================
# FORMAT HANDLERS
# ============================================================================

class FormatHandler:
    """Handles reading various file formats with compression support"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize format handler with options"""
        self.config = config or {}
        self.format = self.config.get('format', 'auto')
        self.compression = self.config.get('compression', 'auto')
        self.encoding = self.config.get('encoding', 'auto')
        
    def detect_compression(self, file_path: str) -> Optional[str]:
        """Detect compression type from file extension"""
        path_lower = file_path.lower()
        if path_lower.endswith(('.gz', '.gzip')):
            return 'gzip'
        elif path_lower.endswith('.bz2'):
            return 'bz2'
        elif path_lower.endswith('.xz'):
            return 'xz'
        elif path_lower.endswith('.zip'):
            return 'zip'
        return None
    
    def detect_format(self, file_path: str, content: bytes = None) -> str:
        """Detect file format from extension or content"""
        base_path = file_path.lower()
        for ext in ['.gz', '.gzip', '.bz2', '.xz', '.zip']:
            base_path = base_path.replace(ext, '')
        
        if base_path.endswith('.csv'):
            return 'csv'
        elif base_path.endswith('.tsv'):
            return 'tsv'
        elif base_path.endswith('.json'):
            return 'json'
        elif base_path.endswith('.jsonl') or base_path.endswith('.ndjson'):
            return 'jsonl'
        elif base_path.endswith('.parquet'):
            return 'parquet'
        elif base_path.endswith(('.xlsx', '.xls')):
            return 'excel'
        elif base_path.endswith(('.hdf5', '.h5')):
            return 'hdf5'
        elif base_path.endswith('.avro'):
            return 'avro'
        elif base_path.endswith('.orc'):
            return 'orc'
        elif base_path.endswith('.feather'):
            return 'feather'
        elif base_path.endswith(('.pkl', '.pickle')):
            return 'pickle'
        elif base_path.endswith('.xml'):
            return 'xml'
        elif base_path.endswith(('.html', '.htm')):
            return 'html'
        elif base_path.endswith('.txt'):
            return 'txt'
        
        if content and len(content) > 0:
            try:
                json.loads(content[:1024].decode('utf-8', errors='ignore'))
                return 'json'
            except:
                pass
            if content[:5] == b'<?xml':
                return 'xml'
        
        return 'csv'
    
    def detect_encoding(self, file_path: str, sample_size: int = 10000) -> str:
        """Detect file encoding"""
        if not HAS_CHARDET:
            return 'utf-8'
        try:
            with open(file_path, 'rb') as f:
                sample = f.read(sample_size)
                result = chardet.detect(sample)
                return result.get('encoding', 'utf-8')
        except:
            return 'utf-8'
    
    def detect_delimiter(self, file_path: str, sample_lines: int = 5) -> str:
        """Detect CSV delimiter"""
        try:
            encoding = self.encoding if self.encoding != 'auto' else 'utf-8'
            if encoding == 'auto':
                encoding = self.detect_encoding(file_path)
            
            with open(file_path, 'r', encoding=encoding) as f:
                sample = ''.join(f.readline() for _ in range(sample_lines))
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                return delimiter
        except:
            return ','
    
    def read_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read CSV file with advanced options"""
        csv_config = self.config.get('csv_options', {})
        
        delimiter = csv_config.get('delimiter')
        if delimiter is None or delimiter == 'auto':
            delimiter = self.detect_delimiter(file_path)
            csv_config['delimiter'] = delimiter
        
        encoding = self.encoding
        if encoding == 'auto':
            encoding = self.detect_encoding(file_path)
        
        read_kwargs = {
            'encoding': encoding,
            'delimiter': delimiter,
            'skiprows': csv_config.get('skip_rows', 0),
            'header': csv_config.get('header', 'infer'),
            'names': csv_config.get('column_names'),
            'dtype': csv_config.get('dtype'),
            'na_values': csv_config.get('na_values'),
            'keep_default_na': csv_config.get('keep_default_na', True),
            'nrows': self.config.get('limit', None),
            **kwargs
        }
        
        read_kwargs = {k: v for k, v in read_kwargs.items() if v is not None}
        
        compression = self.compression
        if compression == 'auto':
            compression = self.detect_compression(file_path)
        
        if compression:
            read_kwargs['compression'] = compression
        
        return pd.read_csv(file_path, **read_kwargs)
    
    def read_json(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read JSON file"""
        json_config = self.config.get('json_options', {})
        
        encoding = self.encoding
        if encoding == 'auto':
            encoding = self.detect_encoding(file_path)
        
        read_kwargs = {
            'encoding': encoding,
            'orient': json_config.get('orient', 'records'),
            'lines': json_config.get('lines', False),
            'nrows': self.config.get('limit', None),
            **kwargs
        }
        
        read_kwargs = {k: v for k, v in read_kwargs.items() if v is not None}
        
        compression = self.compression
        if compression == 'auto':
            compression = self.detect_compression(file_path)
        
        if compression:
            read_kwargs['compression'] = compression
        
        return pd.read_json(file_path, **read_kwargs)
    
    def read_parquet(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read Parquet file"""
        if not HAS_ARROW:
            raise ImportError("pyarrow required. Install with: pip install pyarrow")
        
        parquet_config = self.config.get('parquet_options', {})
        read_kwargs = {
            'columns': parquet_config.get('columns'),
            'filters': parquet_config.get('filters'),
            **kwargs
        }
        read_kwargs = {k: v for k, v in read_kwargs.items() if v is not None}
        return pd.read_parquet(file_path, **read_kwargs)
    
    def read_excel(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read Excel file"""
        if not HAS_EXCEL:
            raise ImportError("openpyxl/xlrd required. Install with: pip install openpyxl xlrd")
        
        excel_config = self.config.get('excel_options', {})
        read_kwargs = {
            'sheet_name': excel_config.get('sheet_name', 0),
            'header': excel_config.get('header', 0),
            'skiprows': excel_config.get('skip_rows', 0),
            'nrows': self.config.get('limit', None),
            'usecols': excel_config.get('usecols'),
            **kwargs
        }
        read_kwargs = {k: v for k, v in read_kwargs.items() if v is not None}
        return pd.read_excel(file_path, **read_kwargs)
    
    def read_hdf5(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read HDF5 file"""
        if not HAS_HDF5:
            raise ImportError("h5py required. Install with: pip install h5py")
        
        hdf5_config = self.config.get('hdf5_options', {})
        key = hdf5_config.get('key', 'default')
        mode = hdf5_config.get('mode', 'r')
        return pd.read_hdf(file_path, key=key, mode=mode, **kwargs)
    
    def read_avro(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read Avro file"""
        if not HAS_AVRO:
            raise ImportError("fastavro required. Install with: pip install fastavro")
        
        records = []
        with open(file_path, 'rb') as f:
            reader = fastavro.reader(f)
            for record in reader:
                records.append(record)
                if self.config.get('limit', 0) > 0 and len(records) >= self.config.get('limit'):
                    break
        return pd.DataFrame(records)
    
    def read_orc(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read ORC file"""
        if not HAS_ARROW:
            raise ImportError("pyarrow required. Install with: pip install pyarrow")
        
        orc_config = self.config.get('orc_options', {})
        read_kwargs = {'columns': orc_config.get('columns'), **kwargs}
        read_kwargs = {k: v for k, v in read_kwargs.items() if v is not None}
        return pd.read_orc(file_path, **read_kwargs)
    
    def read_feather(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read Feather file"""
        if not HAS_ARROW:
            raise ImportError("pyarrow required. Install with: pip install pyarrow")
        return pd.read_feather(file_path, **kwargs)
    
    def read_xml(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read XML file"""
        xml_config = self.config.get('xml_options', {})
        read_kwargs = {
            'xpath': xml_config.get('xpath', './/row'),
            'namespaces': xml_config.get('namespaces', {}),
            **kwargs
        }
        read_kwargs = {k: v for k, v in read_kwargs.items() if v is not None}
        return pd.read_xml(file_path, **read_kwargs)
    
    def read_html(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read HTML file (extract tables)"""
        html_config = self.config.get('html_options', {})
        read_kwargs = {
            'match': html_config.get('table_match'),
            'header': html_config.get('header', 0),
            'index_col': html_config.get('index_col'),
            **kwargs
        }
        read_kwargs = {k: v for k, v in read_kwargs.items() if v is not None}
        tables = pd.read_html(file_path, **read_kwargs)
        return tables[html_config.get('table_index', 0)]
    
    def read(self, file_path: str, format_override: str = None) -> pd.DataFrame:
        """Read file with automatic format detection"""
        file_format = format_override or self.format
        if file_format == 'auto':
            file_format = self.detect_format(file_path)
        
        if file_format == 'csv' or file_format == 'tsv':
            return self.read_csv(file_path)
        elif file_format == 'json':
            return self.read_json(file_path, lines=False)
        elif file_format == 'jsonl':
            return self.read_json(file_path, lines=True)
        elif file_format == 'parquet':
            return self.read_parquet(file_path)
        elif file_format in ('excel', 'xlsx', 'xls'):
            return self.read_excel(file_path)
        elif file_format in ('hdf5', 'h5'):
            return self.read_hdf5(file_path)
        elif file_format == 'avro':
            return self.read_avro(file_path)
        elif file_format == 'orc':
            return self.read_orc(file_path)
        elif file_format == 'feather':
            return self.read_feather(file_path)
        elif file_format == 'xml':
            return self.read_xml(file_path)
        elif file_format in ('html', 'htm'):
            return self.read_html(file_path)
        else:
            raise ValueError(f"Unsupported format: {file_format}")


# ============================================================================
# DATA SOURCE LOADERS
# ============================================================================

class HuggingFaceLoader(BaseDataLoader):
    """Load datasets from HuggingFace Hub"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.split = config.get('split', 'train')
        self.config_name = config.get('config')
        self.streaming = config.get('streaming', True)
        self._hf_token = None
        
    def supports_streaming(self) -> bool:
        return True
    
    def _get_hf_token(self) -> Optional[str]:
        """Get HuggingFace token from environment or config"""
        if self._hf_token is None:
            self._hf_token = (
                os.getenv('HF_TOKEN') or 
                os.getenv('HUGGINGFACE_HUB_TOKEN') or
                self.config.get('hf_token')
            )
        return self._hf_token
    
    def load(self) -> pd.DataFrame:
        if not HAS_DATASETS:
            raise ImportError("datasets library required. Install with: pip install datasets")
        
        dataset_name = self.source_path
        load_kwargs = {'split': self.split}
        
        if self.config_name:
            load_kwargs['config_name'] = self.config_name
        
        hf_token = self._get_hf_token()
        if hf_token:
            load_kwargs['token'] = hf_token
        
        if self.streaming:
            try:
                dataset = load_dataset(dataset_name, streaming=True, **load_kwargs)
                data = []
                count = 0
                for item in dataset:
                    data.append(dict(item))
                    count += 1
                    if self.limit > 0 and count >= self.limit:
                        break
                return pd.DataFrame(data)
            except Exception:
                dataset = load_dataset(dataset_name, streaming=False, **load_kwargs)
                if self.limit > 0:
                    dataset = dataset.select(range(min(self.limit, len(dataset))))
                return pd.DataFrame(dataset)
        else:
            dataset = load_dataset(dataset_name, streaming=False, **load_kwargs)
            if self.limit > 0:
                dataset = dataset.select(range(min(self.limit, len(dataset))))
            return pd.DataFrame(dataset)


class URLLoader(BaseDataLoader):
    """Load datasets from HTTP/HTTPS URLs"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.format_handler = FormatHandler(config)
        self.timeout = config.get('timeout', 30)
        self.headers = config.get('headers', {})
        self.auth = config.get('auth')
        
    def supports_streaming(self) -> bool:
        return True
    
    def load(self) -> pd.DataFrame:
        if not HAS_REQUESTS:
            raise ImportError("requests library required. Install with: pip install requests")
        
        response = requests.get(
            self.source_path,
            stream=True,
            timeout=self.timeout,
            headers=self.headers,
            auth=self.auth
        )
        response.raise_for_status()
        
        if self.format_hint == 'auto':
            content_type = response.headers.get('content-type', '')
            if 'json' in content_type:
                detected_format = 'json'
            elif 'xml' in content_type:
                detected_format = 'xml'
            elif 'html' in content_type:
                detected_format = 'html'
            else:
                detected_format = 'csv'
        else:
            detected_format = self.format_hint
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{detected_format}") as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp_path = tmp.name
        
        try:
            return self.format_handler.read(tmp_path, format_override=detected_format)
        finally:
            os.unlink(tmp_path)


class GitHubLoader(BaseDataLoader):
    """Load datasets from GitHub repositories"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.format_handler = FormatHandler(config)
        self.token = config.get('github_token') or os.getenv('GITHUB_TOKEN')
        
    def supports_streaming(self) -> bool:
        return False
    
    def load(self) -> pd.DataFrame:
        if not HAS_REQUESTS:
            raise ImportError("requests library required. Install with: pip install requests")
        
        parts = self.source_path.split('/', 2)
        if len(parts) < 2:
            raise ValueError("Invalid GitHub path. Use format: owner/repo/path/to/file")
        
        owner, repo = parts[0], parts[1]
        file_path = parts[2] if len(parts) > 2 else None
        
        if not file_path:
            common_files = ['data.csv', 'data.json', 'dataset.csv', 'dataset.json']
            for cf in common_files:
                api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{cf}"
                headers = {}
                if self.token:
                    headers['Authorization'] = f"token {self.token}"
                
                response = requests.get(api_url, headers=headers, timeout=10)
                if response.status_code == 200:
                    file_path = cf
                    break
            
            if not file_path:
                raise ValueError(f"Could not find data file in repository {owner}/{repo}")
        
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
        headers = {}
        if self.token:
            headers['Authorization'] = f"token {self.token}"
        
        response = requests.get(api_url, headers=headers, timeout=30)
        response.raise_for_status()
        file_data = response.json()
        
        if 'download_url' in file_data:
            url_loader = URLLoader({'source_path': file_data['download_url'], **self.config})
            return url_loader.load()
        elif 'content' in file_data:
            import base64
            content = base64.b64decode(file_data['content']).decode('utf-8')
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=Path(file_path).suffix) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            try:
                return self.format_handler.read(tmp_path)
            finally:
                os.unlink(tmp_path)
        else:
            raise ValueError("Could not retrieve file content from GitHub")


class S3Loader(BaseDataLoader):
    """Load datasets from Amazon S3"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.format_handler = FormatHandler(config)
        self.aws_access_key = config.get('aws_access_key') or os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = config.get('aws_secret_key') or os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region = config.get('aws_region') or os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        
    def supports_streaming(self) -> bool:
        return True
    
    def load(self) -> pd.DataFrame:
        if not HAS_BOTO3:
            raise ImportError("boto3 required. Install with: pip install boto3")
        
        if not self.source_path.startswith('s3://'):
            raise ValueError("S3 path must start with 's3://'")
        
        path_parts = self.source_path[5:].split('/', 1)
        bucket = path_parts[0]
        key = path_parts[1] if len(path_parts) > 1 else None
        
        if not key:
            raise ValueError("S3 key is required")
        
        s3_kwargs = {'region_name': self.aws_region}
        if self.aws_access_key and self.aws_secret_key:
            s3_kwargs.update({
                'aws_access_key_id': self.aws_access_key,
                'aws_secret_access_key': self.aws_secret_key
            })
        
        s3_client = boto3.client('s3', **s3_kwargs)
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            s3_client.download_fileobj(bucket, key, tmp)
            tmp_path = tmp.name
        
        try:
            return self.format_handler.read(tmp_path)
        finally:
            os.unlink(tmp_path)


class AzureBlobLoader(BaseDataLoader):
    """Load datasets from Azure Blob Storage"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.format_handler = FormatHandler(config)
        self.connection_string = (
            config.get('azure_connection_string') or 
            os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        )
        self.account_name = config.get('azure_account_name')
        self.account_key = config.get('azure_account_key')
        
    def supports_streaming(self) -> bool:
        return True
    
    def load(self) -> pd.DataFrame:
        if not HAS_AZURE:
            raise ImportError("azure-storage-blob required. Install with: pip install azure-storage-blob")
        
        if not self.source_path.startswith('azure://'):
            raise ValueError("Azure path must start with 'azure://'")
        
        path_parts = self.source_path[8:].split('/', 1)
        container = path_parts[0]
        blob_name = path_parts[1] if len(path_parts) > 1 else None
        
        if not blob_name:
            raise ValueError("Blob name is required")
        
        if self.connection_string:
            blob_service = BlobServiceClient.from_connection_string(self.connection_string)
        elif self.account_name and self.account_key:
            account_url = f"https://{self.account_name}.blob.core.windows.net"
            blob_service = BlobServiceClient(account_url, credential=self.account_key)
        else:
            raise ValueError("Azure credentials required (connection_string or account_name+account_key)")
        
        blob_client = blob_service.get_blob_client(container=container, blob=blob_name)
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            download_stream = blob_client.download_blob()
            tmp.write(download_stream.readall())
            tmp_path = tmp.name
        
        try:
            return self.format_handler.read(tmp_path)
        finally:
            os.unlink(tmp_path)


class GCSLoader(BaseDataLoader):
    """Load datasets from Google Cloud Storage"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.format_handler = FormatHandler(config)
        self.credentials_path = config.get('gcs_credentials') or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
    def supports_streaming(self) -> bool:
        return True
    
    def load(self) -> pd.DataFrame:
        if not HAS_GCS:
            raise ImportError("google-cloud-storage required. Install with: pip install google-cloud-storage")
        
        if not self.source_path.startswith('gs://'):
            raise ValueError("GCS path must start with 'gs://'")
        
        path_parts = self.source_path[5:].split('/', 1)
        bucket_name = path_parts[0]
        blob_name = path_parts[1] if len(path_parts) > 1 else None
        
        if not blob_name:
            raise ValueError("Blob name is required")
        
        if self.credentials_path:
            storage_client = storage.Client.from_service_account_json(self.credentials_path)
        else:
            storage_client = storage.Client()
        
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            blob.download_to_file(tmp)
            tmp_path = tmp.name
        
        try:
            return self.format_handler.read(tmp_path)
        finally:
            os.unlink(tmp_path)


class FTPLoader(BaseDataLoader):
    """Load datasets from FTP/SFTP servers"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.format_handler = FormatHandler(config)
        self.username = config.get('ftp_username') or os.getenv('FTP_USERNAME')
        self.password = config.get('ftp_password') or os.getenv('FTP_PASSWORD')
        self.protocol = config.get('protocol', 'ftp')
        
    def supports_streaming(self) -> bool:
        return False
    
    def load(self) -> pd.DataFrame:
        if not HAS_FTP:
            raise ImportError("paramiko required for SFTP. Install with: pip install paramiko")
        
        if not self.source_path.startswith(('ftp://', 'sftp://')):
            raise ValueError("FTP path must start with 'ftp://' or 'sftp://'")
        
        is_sftp = self.source_path.startswith('sftp://')
        url_path = self.source_path[6:] if not is_sftp else self.source_path[7:]
        
        parts = url_path.split('/', 1)
        host = parts[0]
        file_path = parts[1] if len(parts) > 1 else None
        
        if not file_path:
            raise ValueError("File path is required")
        
        if is_sftp:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(host, username=self.username, password=self.password)
            sftp = ssh.open_sftp()
            
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                sftp.get(file_path, tmp.name)
                tmp_path = tmp.name
            
            sftp.close()
            ssh.close()
        else:
            ftp = FTP(host)
            if self.username and self.password:
                ftp.login(self.username, self.password)
            else:
                ftp.login()
            
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                ftp.retrbinary(f'RETR {file_path}', tmp.write)
                tmp_path = tmp.name
            
            ftp.quit()
        
        try:
            return self.format_handler.read(tmp_path)
        finally:
            os.unlink(tmp_path)


class LocalFileLoader(BaseDataLoader):
    """Load datasets from local filesystem"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.format_handler = FormatHandler(config)
        
    def supports_streaming(self) -> bool:
        return True
    
    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.source_path):
            raise FileNotFoundError(f"File not found: {self.source_path}")
        
        return self.format_handler.read(self.source_path)


class DatabaseLoader(BaseDataLoader):
    """Load datasets from database sources"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.query = config.get('query')
        self.db_type = config.get('db_type', 'postgresql')
        self.connection_string = config.get('connection_string')
        
    def supports_streaming(self) -> bool:
        return True
    
    def load(self) -> pd.DataFrame:
        if not self.query:
            raise ValueError("Query is required for database sources")
        
        if self.db_type == 'postgresql':
            import psycopg2
            conn = psycopg2.connect(self.connection_string or os.getenv('DATABASE_URL'))
            return pd.read_sql_query(self.query, conn)
        elif self.db_type == 'mysql':
            import pymysql
            conn = pymysql.connect(
                host=self.config.get('host', 'localhost'),
                user=self.config.get('user'),
                password=self.config.get('password'),
                database=self.config.get('database')
            )
            return pd.read_sql_query(self.query, conn)
        elif self.db_type == 'sqlite':
            import sqlite3
            conn = sqlite3.connect(self.source_path)
            return pd.read_sql_query(self.query, conn)
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")


def create_loader(source_type: str, config: Dict[str, Any]) -> BaseDataLoader:
    """Factory function to create appropriate loader"""
    loaders = {
        'huggingface': HuggingFaceLoader,
        'url': URLLoader,
        'github': GitHubLoader,
        's3': S3Loader,
        'azure': AzureBlobLoader,
        'gcs': GCSLoader,
        'gs': GCSLoader,
        'ftp': FTPLoader,
        'sftp': FTPLoader,
        'local': LocalFileLoader,
        'database': DatabaseLoader,
        'db': DatabaseLoader,
        'postgresql': DatabaseLoader,
        'mysql': DatabaseLoader,
        'sqlite': DatabaseLoader,
    }
    
    loader_class = loaders.get(source_type.lower())
    if not loader_class:
        raise ValueError(f"Unsupported source type: {source_type}. Supported: {list(loaders.keys())}")
    
    return loader_class(config)


# ============================================================================
# DATA TRANSFORMERS
# ============================================================================

class DataTransformer:
    """Pipeline for transforming data"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.transformations = []
        
    def add_transformation(self, transform_func: Callable, *args, **kwargs):
        self.transformations.append((transform_func, args, kwargs))
        return self
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for transform_func, args, kwargs in self.transformations:
            result = transform_func(result, *args, **kwargs)
        return result


def rename_columns(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    return df.rename(columns=column_mapping)


def select_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    existing_columns = [col for col in columns if col in df.columns]
    if len(existing_columns) < len(columns):
        missing = set(columns) - set(existing_columns)
        print(json.dumps({"warning": f"Columns not found: {missing}"}), flush=True)
    return df[existing_columns]


def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    return df.drop(columns=[col for col in columns if col in df.columns], errors='ignore')


def filter_rows(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    return df.query(condition)


def cast_types(df: pd.DataFrame, type_mapping: Dict[str, str]) -> pd.DataFrame:
    result = df.copy()
    for col, dtype in type_mapping.items():
        if col in result.columns:
            try:
                result[col] = result[col].astype(dtype)
            except Exception as e:
                print(json.dumps({"warning": f"Could not cast {col} to {dtype}: {e}"}), flush=True)
    return result


def fill_null_values(df: pd.DataFrame, fill_config: Dict[str, Any]) -> pd.DataFrame:
    result = df.copy()
    for col, strategy in fill_config.items():
        if col not in result.columns:
            continue
        
        if isinstance(strategy, (int, float, str)):
            result[col] = result[col].fillna(strategy)
        elif strategy == 'mean':
            result[col] = result[col].fillna(result[col].mean())
        elif strategy == 'median':
            result[col] = result[col].fillna(result[col].median())
        elif strategy == 'mode':
            mode_val = result[col].mode()
            result[col] = result[col].fillna(mode_val[0] if len(mode_val) > 0 else None)
        elif strategy == 'forward':
            result[col] = result[col].ffill()
        elif strategy == 'backward':
            result[col] = result[col].bfill()
    
    return result


def remove_duplicates(df: pd.DataFrame, subset: List[str] = None, keep: str = 'first') -> pd.DataFrame:
    return df.drop_duplicates(subset=subset, keep=keep)


def normalize_text(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    result = df.copy()
    for col in columns:
        if col in result.columns:
            result[col] = result[col].astype(str).str.lower().str.strip()
    return result


def create_transformer_from_config(config: Dict[str, Any]) -> DataTransformer:
    """Create transformer from configuration"""
    transformer = DataTransformer(config)
    
    if 'rename_columns' in config:
        transformer.add_transformation(rename_columns, config['rename_columns'])
    if 'select_columns' in config:
        transformer.add_transformation(select_columns, config['select_columns'])
    if 'drop_columns' in config:
        transformer.add_transformation(drop_columns, config['drop_columns'])
    if 'filter' in config:
        transformer.add_transformation(filter_rows, config['filter'])
    if 'cast_types' in config:
        transformer.add_transformation(cast_types, config['cast_types'])
    if 'fill_nulls' in config:
        transformer.add_transformation(fill_null_values, config['fill_nulls'])
    if config.get('remove_duplicates', False):
        subset = config.get('duplicate_subset')
        keep = config.get('duplicate_keep', 'first')
        transformer.add_transformation(remove_duplicates, subset, keep)
    if 'normalize_text' in config:
        transformer.add_transformation(normalize_text, config['normalize_text'])
    
    return transformer


# ============================================================================
# ENHANCED DATASET LOADER
# ============================================================================

class EnhancedDatasetLoader:
    """Enhanced dataset loader with full feature support"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        self.config = {}
        self.loader = None
        self.transformer = None
        self.schema_info = {}
        self.text_columns = []
        self.embedding_columns = []
        
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
            self.cursor.execute("SELECT current_setting('neurondb.llm_api_key', true)")
            result = self.cursor.fetchone()
            if result and result[0]:
                token = result[0].strip()
                if token:
                    return token
        except Exception:
            pass
        return None
    
    def load_data(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Load data from configured source"""
        self.config = config
        
        source_type = config.get('source_type', 'local')
        self.loader = create_loader(source_type, config)
        self.loader.validate_config()
        
        if config.get('streaming', False) and self.loader.supports_streaming():
            chunks = []
            for chunk in self.loader.load_streaming():
                chunks.append(chunk)
                if config.get('limit', 0) > 0:
                    total_rows = sum(len(c) for c in chunks)
                    if total_rows >= config.get('limit'):
                        break
            df = pd.concat(chunks, ignore_index=True)
            if config.get('limit', 0) > 0:
                df = df.head(config.get('limit'))
        else:
            df = self.loader.load()
        
        if 'transformations' in config:
            self.transformer = create_transformer_from_config(config['transformations'])
            df = self.transformer.apply(df)
        
        return df
    
    def detect_schema(self, df: pd.DataFrame, sample_size: int = 1000) -> Dict[str, str]:
        """Enhanced schema detection"""
        schema = {}
        sample_df = df.head(min(sample_size, len(df))) if len(df) > sample_size else df
        
        for col in df.columns:
            col_type = str(df[col].dtype)
            
            if col_type.startswith('int'):
                min_val = df[col].min()
                max_val = df[col].max()
                if pd.isna(min_val) or pd.isna(max_val):
                    schema[col] = "INTEGER"
                elif min_val >= -2147483648 and max_val <= 2147483647:
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
                sample_values = sample_df[col].dropna().head(10)
                if len(sample_values) > 0:
                    first_val = sample_values.iloc[0]
                    if isinstance(first_val, (dict, list)):
                        schema[col] = "JSONB"
                    elif isinstance(first_val, str):
                        max_len = sample_df[col].astype(str).str.len().max()
                        avg_len = sample_df[col].astype(str).str.len().mean()
                        
                        if max_len > 255 or avg_len > 50:
                            schema[col] = "TEXT"
                        else:
                            unique_ratio = df[col].nunique() / len(df)
                            if unique_ratio < 0.1 and len(df) > 100:
                                max_str_len = int(max_len * 1.5)
                                schema[col] = f"VARCHAR({max_str_len})"
                            else:
                                max_str_len = int(max_len * 1.2)
                                schema[col] = f"VARCHAR({max(1, max_str_len)})"
                    else:
                        schema[col] = "TEXT"
                else:
                    schema[col] = "TEXT"
            else:
                schema[col] = "TEXT"
        
        self.schema_info = schema
        return schema
    
    def detect_text_columns(self, df: pd.DataFrame, min_avg_length: int = 20) -> List[str]:
        """Detect text columns for embedding"""
        text_cols = []
        
        for col in df.columns:
            col_type = str(df[col].dtype)
            
            if col_type == 'object':
                sample_values = df[col].dropna().head(100)
                if len(sample_values) > 0:
                    first_val = sample_values.iloc[0]
                    if isinstance(first_val, str):
                        avg_len = sample_values.astype(str).str.len().mean()
                        if avg_len >= min_avg_length:
                            unique_ratio = df[col].nunique() / len(df)
                            if unique_ratio > 0.1 or len(df) < 1000:
                                text_cols.append(col)
        
        self.text_columns = text_cols
        return text_cols
    
    def create_table(self, df: pd.DataFrame, schema_name: str, table_name: str,
                     auto_embed: bool = True, create_indexes: bool = True,
                     if_exists: str = 'fail'):
        """Create optimized PostgreSQL table"""
        self.schema_name = schema_name
        self.table_name = table_name
        
        pg_schema = self.detect_schema(df)
        
        if auto_embed:
            text_cols = self.detect_text_columns(df)
            self.text_columns = text_cols
        
        schema_quoted = quote_ident(schema_name, self.cursor)
        self.cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_quoted}")
        
        columns = ["id SERIAL PRIMARY KEY"]
        
        for col_name, col_type in pg_schema.items():
            col_quoted = quote_ident(col_name, self.cursor)
            columns.append(f"{col_quoted} {col_type}")
        
        if auto_embed and self.text_columns:
            for col in self.text_columns:
                embed_col = f"{col}_embedding"
                self.embedding_columns.append(embed_col)
                embed_col_quoted = quote_ident(embed_col, self.cursor)
                embed_dim = self.config.get('embedding_dimension', 384)
                columns.append(f"{embed_col_quoted} vector({embed_dim})")
        
        table_quoted = quote_ident(table_name, self.cursor)
        
        if if_exists == 'replace':
            self.cursor.execute(f"DROP TABLE IF EXISTS {schema_quoted}.{table_quoted}")
        elif if_exists == 'append':
            self.cursor.execute(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = %s AND table_name = %s
                )
            """, (schema_name, table_name))
            exists = self.cursor.fetchone()[0]
            if exists:
                return pg_schema
        
        create_sql = f"CREATE TABLE IF NOT EXISTS {schema_quoted}.{table_quoted} ({', '.join(columns)})"
        self.cursor.execute(create_sql)
        self.conn.commit()
        
        return pg_schema
    
    def load_data_batch(self, df: pd.DataFrame, schema_name: str, table_name: str,
                       batch_size: int = 1000, mode: str = 'insert'):
        """Load data in batches"""
        schema_quoted = quote_ident(schema_name, self.cursor)
        table_quoted = quote_ident(table_name, self.cursor)
        
        total_rows = len(df)
        inserted = 0
        
        columns = [col for col in df.columns if col != 'id']
        col_names_quoted = [quote_ident(col, self.cursor) for col in columns]
        
        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            batch_df = df.iloc[start_idx:end_idx]
            
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
            
            if mode == 'insert' or mode == 'append':
                placeholders = ','.join(['%s'] * len(col_names_quoted))
                insert_sql = f"INSERT INTO {schema_quoted}.{table_quoted} ({', '.join(col_names_quoted)}) VALUES ({placeholders})"
                self.cursor.executemany(insert_sql, values)
            elif mode == 'upsert':
                placeholders = ','.join(['%s'] * len(col_names_quoted))
                update_clause = ', '.join([f"{col} = EXCLUDED.{col}" for col in col_names_quoted])
                insert_sql = f"""
                    INSERT INTO {schema_quoted}.{table_quoted} ({', '.join(col_names_quoted)}) 
                    VALUES ({placeholders})
                    ON CONFLICT (id) DO UPDATE SET {update_clause}
                """
                self.cursor.executemany(insert_sql, values)
            
            inserted += len(batch_df)
            self.conn.commit()
            
            progress = inserted / total_rows if total_rows > 0 else 1.0
            print(json.dumps({
                "status": "loading",
                "progress": progress,
                "rows_loaded": inserted,
                "total_rows": total_rows,
                "current_batch": start_idx // batch_size + 1
            }), flush=True)
        
        return inserted
    
    def generate_embeddings(self, schema_name: str, table_name: str,
                           embedding_model: str = "default", batch_size: int = 100):
        """Generate embeddings for text columns"""
        if not self.text_columns:
            return 0
        
        schema_quoted = quote_ident(schema_name, self.cursor)
        table_quoted = quote_ident(table_name, self.cursor)
        
        total_embedded = 0
        
        for text_col, embed_col in zip(self.text_columns, self.embedding_columns):
            text_col_quoted = quote_ident(text_col, self.cursor)
            embed_col_quoted = quote_ident(embed_col, self.cursor)
            
            count_sql = f"""
                SELECT COUNT(*) FROM {schema_quoted}.{table_quoted} 
                WHERE {embed_col_quoted} IS NULL AND {text_col_quoted} IS NOT NULL
            """
            self.cursor.execute(count_sql)
            total_count = self.cursor.fetchone()[0]
            
            if total_count == 0:
                continue
            
            offset = 0
            while offset < total_count:
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
                
                for row_id, text_val in rows:
                    if text_val and len(str(text_val).strip()) > 0:
                        try:
                            embed_sql = f"SELECT embed_text(%s, %s)::text"
                            self.cursor.execute(embed_sql, (str(text_val), embedding_model))
                            embedding = self.cursor.fetchone()[0]
                            
                            update_sql = f"""
                                UPDATE {schema_quoted}.{table_quoted} 
                                SET {embed_col_quoted} = %s::vector 
                                WHERE id = %s
                            """
                            self.cursor.execute(update_sql, (embedding, row_id))
                            total_embedded += 1
                        except Exception as e:
                            print(json.dumps({
                                "warning": f"Failed to generate embedding for row {row_id}: {e}",
                                "status": "embedding"
                            }), file=sys.stderr, flush=True)
                
                self.conn.commit()
                offset += batch_size
                
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
            except Exception:
                try:
                    create_idx_sql = f"""
                        CREATE INDEX IF NOT EXISTS {index_name_quoted} 
                        ON {schema_quoted}.{table_quoted} 
                        USING ivfflat ({embed_col_quoted})
                    """
                    self.cursor.execute(create_idx_sql)
                    indexes_created.append(f"IVFFlat index on {embed_col}")
                except:
                    pass
        
        for text_col in self.text_columns:
            text_col_quoted = quote_ident(text_col, self.cursor)
            index_name = f"{table_name}_{text_col}_gin_idx"
            index_name_quoted = quote_ident(index_name, self.cursor)
            
            try:
                create_idx_sql = f"""
                    CREATE INDEX IF NOT EXISTS {index_name_quoted} 
                    ON {schema_quoted}.{table_quoted} 
                    USING gin (to_tsvector('english', {text_col_quoted}))
                """
                self.cursor.execute(create_idx_sql)
                indexes_created.append(f"GIN index on {text_col}")
            except Exception:
                pass
        
        self.conn.commit()
        return indexes_created
    
    def create_checkpoint(self, checkpoint_key: str, metadata: Dict[str, Any]):
        """Create checkpoint for incremental loading"""
        checkpoint_table = "data_load_checkpoints"
        
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {checkpoint_table} (
                checkpoint_key VARCHAR(255) PRIMARY KEY,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        self.cursor.execute(f"""
            INSERT INTO {checkpoint_table} (checkpoint_key, metadata)
            VALUES (%s, %s)
            ON CONFLICT (checkpoint_key) DO UPDATE SET metadata = %s, created_at = NOW()
        """, (checkpoint_key, json.dumps(metadata), json.dumps(metadata)))
        
        self.conn.commit()
    
    def get_checkpoint(self, checkpoint_key: str) -> Optional[Dict[str, Any]]:
        """Get checkpoint metadata"""
        checkpoint_table = "data_load_checkpoints"
        
        self.cursor.execute(f"""
            SELECT metadata FROM {checkpoint_table}
            WHERE checkpoint_key = %s
        """, (checkpoint_key,))
        
        result = self.cursor.fetchone()
        if result:
            return json.loads(result[0])
        return None


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def parse_json_arg(arg_str: str) -> dict:
    """Parse JSON string argument"""
    if not arg_str:
        return {}
    try:
        return json.loads(arg_str)
    except json.JSONDecodeError:
        if os.path.exists(arg_str):
            with open(arg_str, 'r') as f:
                return json.load(f)
        raise ValueError(f"Invalid JSON: {arg_str}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="NeuronMCP Comprehensive Dataset Loader",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Source configuration
    parser.add_argument('--source-type', required=True,
                       choices=['huggingface', 'url', 'github', 's3', 'azure', 'gcs', 'gs',
                                'ftp', 'sftp', 'local', 'database', 'db', 'postgresql', 'mysql', 'sqlite'],
                       help='Data source type')
    parser.add_argument('--source-path', required=True,
                       help='Dataset identifier/path')
    
    # HuggingFace specific
    parser.add_argument('--split', default='train', help='Dataset split for HuggingFace')
    parser.add_argument('--config', help='Dataset config name for HuggingFace')
    
    # Format and compression
    parser.add_argument('--format', default='auto',
                       choices=['auto', 'csv', 'json', 'jsonl', 'parquet', 'excel', 'xlsx', 'xls',
                                'hdf5', 'h5', 'avro', 'orc', 'feather', 'xml', 'html', 'tsv'],
                       help='File format')
    parser.add_argument('--compression', default='auto',
                       choices=['auto', 'gzip', 'bz2', 'xz', 'zip', 'none'],
                       help='Compression type')
    parser.add_argument('--encoding', default='auto', help='File encoding')
    
    # CSV specific options
    parser.add_argument('--csv-delimiter', dest='csv_delimiter', help='CSV delimiter')
    parser.add_argument('--csv-header', dest='csv_header', type=int, default=0, help='CSV header row')
    parser.add_argument('--csv-skip-rows', dest='csv_skip_rows', type=int, default=0, help='CSV rows to skip')
    
    # Excel specific options
    parser.add_argument('--excel-sheet', dest='excel_sheet', default=0, help='Excel sheet name/index')
    
    # Loading options
    parser.add_argument('--limit', type=int, default=0, help='Maximum rows to load (0 for unlimited)')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for loading')
    parser.add_argument('--streaming', action='store_true', help='Enable streaming mode')
    parser.add_argument('--no-streaming', dest='streaming', action='store_false', help='Disable streaming')
    parser.set_defaults(streaming=None)
    
    # Table configuration
    parser.add_argument('--schema-name', default='datasets', help='PostgreSQL schema name')
    parser.add_argument('--table-name', help='Custom table name')
    parser.add_argument('--if-exists', default='fail', choices=['fail', 'replace', 'append'],
                       help='What to do if table exists')
    parser.add_argument('--load-mode', dest='load_mode', default='insert',
                       choices=['insert', 'append', 'upsert'], help='Data loading mode')
    
    # Embedding options
    parser.add_argument('--auto-embed', action='store_true', default=True, help='Auto-generate embeddings')
    parser.add_argument('--no-auto-embed', dest='auto_embed', action='store_false', help='Disable auto-embedding')
    parser.add_argument('--embedding-model', default='default', help='Embedding model name')
    parser.add_argument('--embedding-dimension', type=int, default=384, help='Embedding vector dimension')
    parser.add_argument('--text-columns', nargs='+', help='Specific text columns to embed')
    
    # Index options
    parser.add_argument('--create-indexes', action='store_true', default=True, help='Create indexes')
    parser.add_argument('--no-create-indexes', dest='create_indexes', action='store_false', help='Disable indexes')
    
    # Transformations
    parser.add_argument('--transformations', type=parse_json_arg, help='JSON transformations configuration')
    
    # Cloud credentials
    parser.add_argument('--aws-access-key', help='AWS access key ID')
    parser.add_argument('--aws-secret-key', help='AWS secret access key')
    parser.add_argument('--aws-region', help='AWS region')
    parser.add_argument('--azure-connection-string', help='Azure storage connection string')
    parser.add_argument('--gcs-credentials', help='GCS credentials file path')
    parser.add_argument('--github-token', help='GitHub personal access token')
    
    # Cache configuration
    parser.add_argument('--cache-dir', dest='cache_dir', help='Cache directory path')
    
    # Database query
    parser.add_argument('--query', help='SQL query for database sources')
    
    # Incremental loading
    parser.add_argument('--checkpoint-key', help='Checkpoint key for incremental loading')
    parser.add_argument('--use-checkpoint', action='store_true', help='Use checkpoint if available')
    
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
    
    if args.transformations:
        config['transformations'] = args.transformations
    
    if args.query:
        config['query'] = args.query
    
    if args.embedding_dimension:
        config['embedding_dimension'] = args.embedding_dimension
    
    # Get database config from environment
    db_config = {
        'host': os.getenv('PGHOST', 'localhost'),
        'port': int(os.getenv('PGPORT', '5432')),
        'user': os.getenv('PGUSER', 'postgres'),
        'password': os.getenv('PGPASSWORD', ''),
        'database': os.getenv('PGDATABASE', 'postgres')
    }
    
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
        
    except ImportError as e:
        error_result = {
            "status": "error",
            "error": f"Required Python package missing: {str(e)}",
            "error_type": "ImportError",
            "hint": "Install required dependencies: pip install psycopg2-binary pandas datasets requests boto3 azure-storage-blob google-cloud-storage paramiko openpyxl xlrd h5py fastavro pyarrow chardet"
        }
        print(json.dumps(error_result), file=sys.stderr, flush=True)
        sys.exit(1)
    except FileNotFoundError as e:
        error_result = {
            "status": "error",
            "error": f"File or dataset not found: {str(e)}",
            "error_type": "FileNotFoundError",
            "hint": "Please verify that the source path is correct and accessible. For remote sources, check your network connection."
        }
        print(json.dumps(error_result), file=sys.stderr, flush=True)
        sys.exit(1)
    except ValueError as e:
        error_result = {
            "status": "error",
            "error": f"Invalid configuration: {str(e)}",
            "error_type": "ValueError",
            "hint": "Please check your parameters. Ensure source_type and source_path are correct."
        }
        print(json.dumps(error_result), file=sys.stderr, flush=True)
        sys.exit(1)
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        hint = ""
        
        # Provide specific hints based on error message patterns
        if "connection" in error_msg.lower() or "connect" in error_msg.lower():
            hint = "Connection failed. Please check your database connection settings, network connectivity, or cloud service credentials."
        elif "permission" in error_msg.lower() or "access" in error_msg.lower():
            hint = "Permission denied. Please check file permissions, database user permissions, or cloud service access credentials."
        elif "authentication" in error_msg.lower() or "credential" in error_msg.lower():
            hint = "Authentication failed. Please verify your credentials (API keys, tokens, passwords) are correct."
        elif "not found" in error_msg.lower():
            hint = "Resource not found. Please verify the source path, table name, or dataset identifier is correct."
        elif "timeout" in error_msg.lower():
            hint = "Operation timed out. The data source may be slow or unavailable. Try increasing timeout settings."
        elif "memory" in error_msg.lower() or "out of memory" in error_msg.lower():
            hint = "Out of memory. Try reducing the limit parameter or processing data in smaller batches."
        else:
            hint = "An unexpected error occurred. Check the error message for details."
        
        error_result = {
            "status": "error",
            "error": error_msg,
            "error_type": error_type,
            "hint": hint
        }
        
        import traceback
        error_result["traceback"] = traceback.format_exc()
        print(json.dumps(error_result), file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

