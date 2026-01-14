#!/usr/bin/env python3
"""-------------------------------------------------------------------------
 *
 * neuronmcp_cli.py
 *    NeuronMCP Common CLI Library for Python
 *
 * Provides standard CLI functions for all NeuronMCP Python scripts including
 * argument parsing, help message display, version information, and logging
 * configuration. Ensures consistent command-line interface across all scripts.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/tests/lib/neuronmcp_cli.py
 *
 *-------------------------------------------------------------------------
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Version
VERSION = "1.0.0"

# Exit codes
EXIT_SUCCESS = 0
EXIT_GENERAL_ERROR = 1
EXIT_MISUSE = 2


def setup_cli(description: str, version: str = VERSION, prog: Optional[str] = None) -> argparse.ArgumentParser:
    """
    Set up standard CLI argument parser with common options.
    
    Args:
        description: Script description
        version: Script version
        prog: Program name (defaults to script name)
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description=description,
        prog=prog,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Standard options
    parser.add_argument(
        '-V', '--version',
        action='version',
        version=f'%(prog)s {version}'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser


def get_logger(verbose: bool = False, name: Optional[str] = None) -> logging.Logger:
    """
    Get configured logger instance.
    
    Args:
        verbose: Enable verbose (DEBUG) logging
        name: Logger name (defaults to script name)
    
    Returns:
        Configured Logger instance
    """
    if name is None:
        name = Path(sys.argv[0]).stem
    
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            '[%(levelname)s] %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    return logger


def print_info(message: str):
    """Print info message."""
    print(f"[INFO] {message}", file=sys.stderr)


def print_success(message: str):
    """Print success message."""
    print(f"[SUCCESS] {message}", file=sys.stderr)


def print_warning(message: str):
    """Print warning message."""
    print(f"[WARNING] {message}", file=sys.stderr)


def print_error(message: str):
    """Print error message."""
    print(f"[ERROR] {message}", file=sys.stderr)


def print_debug(message: str, verbose: bool = False):
    """Print debug message if verbose."""
    if verbose:
        print(f"[DEBUG] {message}", file=sys.stderr)


def error(message: str, exit_code: int = EXIT_GENERAL_ERROR):
    """Print error and exit."""
    print_error(message)
    sys.exit(exit_code)


def require_file(file_path: Path, description: str = "file"):
    """Require file to exist."""
    if not file_path.exists():
        error(f"Required {description} not found: {file_path}")


def require_dir(dir_path: Path, description: str = "directory"):
    """Require directory to exist."""
    if not dir_path.is_dir():
        error(f"Required {description} not found: {dir_path}")


def print_section(title: str):
    """Print section header."""
    print("", file=sys.stderr)
    print("=" * 40, file=sys.stderr)
    print(title, file=sys.stderr)
    print("=" * 40, file=sys.stderr)
    print("", file=sys.stderr)
