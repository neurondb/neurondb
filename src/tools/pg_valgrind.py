#!/usr/bin/env python3
"""
PostgreSQL Valgrind Memory Debugging Tool
==========================================

A comprehensive, modular tool for running PostgreSQL under Valgrind and generating
detailed memory debugging reports. Supports multiple report formats including
HTML, JSON, and text summaries.

Usage:
    python pg_valgrind.py [OPTIONS] [TEST_SQL_FILE]

Examples:
    # Run PostgreSQL under Valgrind and execute test SQL
    python pg_valgrind.py --pgdata ./pgdata tests/sql/basic/029_embeddings_basic.sql

    # Generate HTML report with detailed stack traces
    python pg_valgrind.py --pgdata ./pgdata --html-report report.html --verbose

    # Quick leak check only
    python pg_valgrind.py --pgdata ./pgdata --leak-check=yes --leak-resolution=high
"""

import os
import sys
import re
import json
import subprocess
import argparse
import tempfile
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
from enum import Enum


# ============================================================================
# Configuration and Constants
# ============================================================================

class ErrorType(Enum):
	"""Types of Valgrind errors"""
	INVALID_READ = "Invalid read"
	INVALID_WRITE = "Invalid write"
	USE_AFTER_FREE = "Use of uninitialised value"
	INVALID_FREE = "Invalid free()"
	DOUBLE_FREE = "Double free()"
	MEMORY_LEAK = "Memory leak"
	UNINITIALIZED = "Uninitialised value"
	CONDITIONAL_JUMP = "Conditional jump"
	SYSCALL_PARAM = "Syscall param"


@dataclass
class ValgrindError:
	"""Represents a single Valgrind error"""
	error_type: str
	severity: str = "unknown"
	pid: Optional[int] = None
	tid: Optional[int] = None
	what: str = ""
	stack_trace: List[Dict[str, str]] = field(default_factory=list)
	heap_block: Optional[str] = None
	source_file: Optional[str] = None
	source_line: Optional[int] = None
	function: Optional[str] = None
	bytes_lost: Optional[int] = None
	leak_summary: Optional[str] = None
	first_occurrence: bool = True
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary for JSON serialization"""
		return asdict(self)


@dataclass
class ValgrindReport:
	"""Complete Valgrind report with all errors and summary"""
	timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
	command: str = ""
	pid: Optional[int] = None
	errors: List[ValgrindError] = field(default_factory=list)
	total_errors: int = 0
	total_leaks: int = 0
	bytes_leaked: int = 0
	bytes_definitely_lost: int = 0
	bytes_indirectly_lost: int = 0
	bytes_possibly_lost: int = 0
	execution_time: float = 0.0
	valgrind_version: str = ""
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary for JSON serialization"""
		result = asdict(self)
		result['errors'] = [e.to_dict() for e in self.errors]
		return result


# ============================================================================
# PostgreSQL Detection Module
# ============================================================================

class PostgreSQLDetector:
	"""Detects and validates PostgreSQL installation"""
	
	@staticmethod
	def find_pg_config() -> Optional[str]:
		"""Find pg_config binary in common locations"""
		# Check environment variable first
		if 'PG_CONFIG' in os.environ:
			pg_config = os.environ['PG_CONFIG']
			if os.path.isfile(pg_config) and os.access(pg_config, os.X_OK):
				return pg_config
		
		# Check PATH
		pg_config = shutil.which('pg_config')
		if pg_config:
			return pg_config
		
		# Check common installation paths
		common_paths = [
			'/usr/local/pgsql.18-pge/bin/pg_config',
			'/usr/local/pgsql.17/bin/pg_config',
			'/usr/pgsql-18/bin/pg_config',
			'/usr/pgsql-17/bin/pg_config',
			'/usr/lib/postgresql/18/bin/pg_config',
			'/usr/lib/postgresql/17/bin/pg_config',
			'/opt/homebrew/opt/postgresql@18/bin/pg_config',
			'/opt/homebrew/opt/postgresql@17/bin/pg_config',
		]
		
		for path in common_paths:
			if os.path.isfile(path) and os.access(path, os.X_OK):
				return path
		
		return None
	
	@staticmethod
	def get_pg_vars(pg_config: str) -> Dict[str, str]:
		"""Get PostgreSQL configuration variables"""
		variables = ['BINDIR', 'SBINDIR', 'LIBDIR', 'INCLUDEDIR', 'PKGLIBDIR']
		result = {}
		
		for var in variables:
			try:
				output = subprocess.check_output(
					[pg_config, f'--{var.lower()}'],
					stderr=subprocess.DEVNULL,
					text=True
				).strip()
				result[var] = output
			except subprocess.CalledProcessError:
				pass
		
		return result
	
	@staticmethod
	def find_postgres(pg_vars: Dict[str, str]) -> Optional[str]:
		"""Find postgres binary"""
		if 'BINDIR' in pg_vars:
			postgres = os.path.join(pg_vars['BINDIR'], 'postgres')
			if os.path.isfile(postgres) and os.access(postgres, os.X_OK):
				return postgres
		
		# Fallback to PATH
		return shutil.which('postgres')
	
	@staticmethod
	def find_pg_ctl(pg_vars: Dict[str, str]) -> Optional[str]:
		"""Find pg_ctl binary"""
		if 'BINDIR' in pg_vars:
			pg_ctl = os.path.join(pg_vars['BINDIR'], 'pg_ctl')
			if os.path.isfile(pg_ctl) and os.access(pg_ctl, os.X_OK):
				return pg_ctl
		
		return shutil.which('pg_ctl')


# ============================================================================
# Valgrind Parser Module
# ============================================================================

class ValgrindParser:
	"""Parses Valgrind output into structured report"""
	
	ERROR_PATTERNS = {
		'invalid_read': re.compile(r'Invalid read of size (\d+)'),
		'invalid_write': re.compile(r'Invalid write of size (\d+)'),
		'use_after_free': re.compile(r'Use of uninitialised value of size (\d+)'),
		'invalid_free': re.compile(r'Invalid free\(\)'),
		'double_free': re.compile(r'Double free\(\)'),
		'memleak': re.compile(r'(\d+) bytes in (\d+) blocks are (definitely|indirectly|possibly) lost'),
		'uninit': re.compile(r'Use of uninitialised value'),
		'cond_jump': re.compile(r'Conditional jump or move depends on uninitialised value'),
		'syscall': re.compile(r'Syscall param .* points to .* unaddressable byte'),
	}
	
	STACK_FRAME_PATTERN = re.compile(
		r'^\s+at\s+(0x[0-9A-Fa-f]+|==\d+==):\s*(.+)'
	)
	
	FUNCTION_PATTERN = re.compile(
		r'^\s+by\s+0x[0-9A-Fa-f]+:\s*(.+)'
	)
	
	SOURCE_PATTERN = re.compile(
		r'^(.+):(\d+)(?::(\d+))?'
	)
	
	HEAP_BLOCK_PATTERN = re.compile(
		r'Address 0x([0-9A-Fa-f]+) is (\d+) bytes inside a block of size (\d+)'
	)
	
	PID_PATTERN = re.compile(r'^==(\d+)==')
	
	LEAK_SUMMARY_PATTERN = re.compile(
		r'LEAK SUMMARY:\s+definitely lost:\s+(\d+) bytes in (\d+) blocks'
		r'|indirectly lost:\s+(\d+) bytes in (\d+) blocks'
		r'|possibly lost:\s+(\d+) bytes in (\d+) blocks'
	)
	
	def __init__(self, output: str):
		self.output = output
		self.lines = output.split('\n')
		self.errors: List[ValgrindError] = []
		self.current_error: Optional[ValgrindError] = None
		self.current_pid: Optional[int] = None
		self.report = ValgrindReport()
		
	def parse(self) -> ValgrindReport:
		"""Parse the entire Valgrind output"""
		i = 0
		while i < len(self.lines):
			line = self.lines[i]
			
			# Detect process ID
			pid_match = self.PID_PATTERN.match(line)
			if pid_match:
				self.current_pid = int(pid_match.group(1))
			
			# Detect error start
			if self._is_error_start(line):
				if self.current_error:
					self.errors.append(self.current_error)
				self.current_error = self._create_error(line)
				i += 1
				i = self._parse_error_details(i)
			# Detect leak summary
			elif 'LEAK SUMMARY' in line:
				i = self._parse_leak_summary(i)
			# Detect Valgrind version
			elif line.startswith('valgrind-') or 'Valgrind-' in line:
				self.report.valgrind_version = line.strip()
			
			i += 1
		
		# Add last error if exists
		if self.current_error:
			self.errors.append(self.current_error)
		
		# Build report
		self.report.errors = self.errors
		self.report.total_errors = len(self.errors)
		self.report.pid = self.current_pid
		self._calculate_summary()
		
		return self.report
	
	def _is_error_start(self, line: str) -> bool:
		"""Check if line starts a new error"""
		error_markers = [
			'invalid read',
			'invalid write',
			'Use of uninitialised',
			'Invalid free()',
			'Double free()',
			'definitely lost',
			'indirectly lost',
			'possibly lost',
			'Conditional jump',
			'Syscall param',
		]
		return any(marker in line.lower() for marker in error_markers)
	
	def _create_error(self, line: str) -> ValgrindError:
		"""Create a new ValgrindError from error line"""
		error = ValgrindError(
			error_type="unknown",
			what=line.strip(),
			pid=self.current_pid
		)
		
		line_lower = line.lower()
		if 'invalid read' in line_lower:
			error.error_type = "Invalid read"
			error.severity = "high"
		elif 'invalid write' in line_lower:
			error.error_type = "Invalid write"
			error.severity = "critical"
		elif 'use of uninitialised' in line_lower or 'uninitialised' in line_lower:
			error.error_type = "Uninitialised value"
			error.severity = "high"
		elif 'invalid free()' in line_lower:
			error.error_type = "Invalid free()"
			error.severity = "critical"
		elif 'double free()' in line_lower:
			error.error_type = "Double free()"
			error.severity = "critical"
		elif 'definitely lost' in line_lower:
			error.error_type = "Memory leak (definitely lost)"
			error.severity = "high"
		elif 'indirectly lost' in line_lower:
			error.error_type = "Memory leak (indirectly lost)"
			error.severity = "medium"
		elif 'possibly lost' in line_lower:
			error.error_type = "Memory leak (possibly lost)"
			error.severity = "low"
		elif 'conditional jump' in line_lower:
			error.error_type = "Conditional jump"
			error.severity = "medium"
		elif 'syscall param' in line_lower:
			error.error_type = "Syscall param"
			error.severity = "high"
		
		return error
	
	def _parse_error_details(self, start_idx: int) -> int:
		"""Parse error details and stack trace"""
		if not self.current_error:
			return start_idx
		
		i = start_idx
		in_stack = False
		in_heap = False
		
		while i < len(self.lines):
			line = self.lines[i].strip()
			
			# Empty line might indicate end of error
			if not line and in_stack:
				if self.current_error and not self.current_error.stack_trace:
					i += 1
					continue
				else:
					break
			
			# Parse stack frames
			if line.startswith('at ') or line.startswith('by '):
				in_stack = True
				frame = self._parse_stack_frame(line)
				if frame:
					if not self.current_error:
						break
					self.current_error.stack_trace.append(frame)
			
			# Parse heap block information
			elif 'Address 0x' in line and 'is' in line and 'bytes' in line:
				match = self.HEAP_BLOCK_PATTERN.search(line)
				if match:
					addr, offset, size = match.groups()
					if self.current_error:
						self.current_error.heap_block = f"0x{addr} (offset {offset} of {size} bytes)"
			
			# Parse source location
			elif ':' in line and ('src/' in line or 'include/' in line):
				match = self.SOURCE_PATTERN.search(line)
				if match:
					file_path, line_num = match.group(1), match.group(2)
					if self.current_error:
						self.current_error.source_file = file_path
						if line_num:
							try:
								self.current_error.source_line = int(line_num)
							except ValueError:
								pass
			
			# Extract bytes lost for leaks
			if 'bytes in' in line and 'lost' in line:
				match = re.search(r'(\d+) bytes', line)
				if match and self.current_error:
					try:
						self.current_error.bytes_lost = int(match.group(1))
					except ValueError:
						pass
			
			i += 1
		
		return i
	
	def _parse_stack_frame(self, line: str) -> Optional[Dict[str, str]]:
		"""Parse a stack frame line"""
		# Remove PID prefix if present
		line = re.sub(r'^==\d+==\s*', '', line)
		
		frame = {}
		
		# Parse function name
		match = re.search(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', line)
		if match:
			frame['function'] = match.group(1)
		
		# Parse source file and line
		match = self.SOURCE_PATTERN.search(line)
		if match:
			file_path = match.group(1)
			if ':' in file_path:
				parts = file_path.rsplit(':', 1)
				frame['file'] = parts[0]
				if len(parts) > 1:
					frame['line'] = parts[1]
			else:
				frame['file'] = file_path
			
			line_num = match.group(2)
			if line_num:
				frame['line'] = line_num
		
		# Parse address
		match = re.search(r'(0x[0-9A-Fa-f]+)', line)
		if match:
			frame['address'] = match.group(1)
		
		if frame:
			frame['raw'] = line.strip()
			return frame
		
		return None
	
	def _parse_leak_summary(self, start_idx: int) -> int:
		"""Parse leak summary section"""
		i = start_idx
		leak_types = {
			'definitely lost': 'bytes_definitely_lost',
			'indirectly lost': 'bytes_indirectly_lost',
			'possibly lost': 'bytes_possibly_lost',
		}
		
		while i < len(self.lines) and i < start_idx + 10:
			line = self.lines[i]
			
			for leak_type, attr in leak_types.items():
				if leak_type in line.lower():
					match = re.search(r'(\d+) bytes', line)
					if match:
						try:
							bytes_val = int(match.group(1))
							setattr(self.report, attr, bytes_val)
							self.report.bytes_leaked += bytes_val
						except (ValueError, AttributeError):
							pass
			
			i += 1
		
		return i
	
	def _calculate_summary(self):
		"""Calculate summary statistics"""
		self.report.total_leaks = sum(
			1 for e in self.errors if 'leak' in e.error_type.lower()
		)
		
		# Count errors by severity
		self.report.total_errors = len(self.errors)


# ============================================================================
# Report Generator Module
# ============================================================================

class HTMLReportGenerator:
	"""Generates beautiful HTML reports from Valgrind data"""
	
	@staticmethod
	def generate(report: ValgrindReport, output_file: str):
		"""Generate HTML report"""
		html = HTMLReportGenerator._build_html(report)
		
		with open(output_file, 'w', encoding='utf-8') as f:
			f.write(html)
	
	@staticmethod
	def _build_html(report: ValgrindReport) -> str:
		"""Build complete HTML document"""
		severity_colors = {
			'critical': '#d32f2f',
			'high': '#f57c00',
			'medium': '#fbc02d',
			'low': '#689f38',
			'unknown': '#757575'
		}
		
		errors_by_type = defaultdict(list)
		for error in report.errors:
			errors_by_type[error.error_type].append(error)
		
		html = f"""<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8">
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<title>NeuronDB Valgrind Report - {report.timestamp}</title>
	<style>
		* {{ margin: 0; padding: 0; box-sizing: border-box; }}
		body {{
			font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
			background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
			min-height: 100vh;
			padding: 20px;
		}}
		.container {{
			max-width: 1400px;
			margin: 0 auto;
			background: white;
			border-radius: 12px;
			box-shadow: 0 20px 60px rgba(0,0,0,0.3);
			overflow: hidden;
		}}
		.header {{
			background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
			color: white;
			padding: 30px;
			text-align: center;
		}}
		.header h1 {{
			font-size: 2.5em;
			margin-bottom: 10px;
			font-weight: 300;
		}}
		.header .subtitle {{
			opacity: 0.9;
			font-size: 1.1em;
		}}
		.summary {{
			display: grid;
			grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
			gap: 20px;
			padding: 30px;
			background: #f5f7fa;
			border-bottom: 2px solid #e1e8ed;
		}}
		.summary-card {{
			background: white;
			padding: 20px;
			border-radius: 8px;
			box-shadow: 0 2px 4px rgba(0,0,0,0.1);
			text-align: center;
		}}
		.summary-card .value {{
			font-size: 2.5em;
			font-weight: bold;
			color: #667eea;
			margin-bottom: 5px;
		}}
		.summary-card .label {{
			color: #666;
			font-size: 0.9em;
			text-transform: uppercase;
			letter-spacing: 1px;
		}}
		.summary-card.critical .value {{ color: {severity_colors['critical']}; }}
		.summary-card.high .value {{ color: {severity_colors['high']}; }}
		.summary-card.medium .value {{ color: {severity_colors['medium']}; }}
		.content {{
			padding: 30px;
		}}
		.error-group {{
			margin-bottom: 40px;
		}}
		.error-group-header {{
			background: #667eea;
			color: white;
			padding: 15px 20px;
			border-radius: 8px 8px 0 0;
			font-size: 1.3em;
			font-weight: 600;
			display: flex;
			justify-content: space-between;
			align-items: center;
		}}
		.error-count {{
			background: rgba(255,255,255,0.2);
			padding: 5px 15px;
			border-radius: 20px;
			font-size: 0.8em;
		}}
		.error-item {{
			background: white;
			border: 1px solid #e1e8ed;
			border-top: none;
			padding: 20px;
			margin-bottom: 2px;
		}}
		.error-item:last-child {{
			border-radius: 0 0 8px 8px;
		}}
		.error-header {{
			display: flex;
			justify-content: space-between;
			align-items: center;
			margin-bottom: 15px;
		}}
		.error-type {{
			font-size: 1.2em;
			font-weight: 600;
			color: #333;
		}}
		.severity-badge {{
			padding: 5px 15px;
			border-radius: 20px;
			font-size: 0.85em;
			font-weight: 600;
			text-transform: uppercase;
		}}
		.severity-critical {{ background: {severity_colors['critical']}; color: white; }}
		.severity-high {{ background: {severity_colors['high']}; color: white; }}
		.severity-medium {{ background: {severity_colors['medium']}; color: white; }}
		.severity-low {{ background: {severity_colors['low']}; color: white; }}
		.error-details {{
			background: #f8f9fa;
			padding: 15px;
			border-radius: 6px;
			margin: 10px 0;
			font-family: 'Monaco', 'Courier New', monospace;
			font-size: 0.9em;
			overflow-x: auto;
		}}
		.stack-trace {{
			margin-top: 15px;
		}}
		.stack-frame {{
			padding: 8px 12px;
			margin: 5px 0;
			background: white;
			border-left: 3px solid #667eea;
			font-family: 'Monaco', 'Courier New', monospace;
			font-size: 0.85em;
		}}
		.stack-frame:hover {{
			background: #f0f4ff;
		}}
		.source-info {{
			display: flex;
			gap: 20px;
			margin-top: 10px;
			font-size: 0.9em;
			color: #666;
		}}
		.source-info strong {{
			color: #333;
		}}
		.bytes-lost {{
			color: {severity_colors['high']};
			font-weight: 600;
		}}
		.footer {{
			background: #f5f7fa;
			padding: 20px;
			text-align: center;
			color: #666;
			font-size: 0.9em;
			border-top: 1px solid #e1e8ed;
		}}
		.collapsible {{
			cursor: pointer;
		}}
		.collapsible-content {{
			display: none;
			margin-top: 10px;
		}}
		.collapsible-content.expanded {{
			display: block;
		}}
	</style>
</head>
<body>
	<div class="container">
		<div class="header">
			<h1>NeuronDB Valgrind Report</h1>
			<div class="subtitle">Memory Debugging Analysis • {report.timestamp}</div>
		</div>
		
		<div class="summary">
			<div class="summary-card {'critical' if report.total_errors > 0 else ''}">
				<div class="value">{report.total_errors}</div>
				<div class="label">Total Errors</div>
			</div>
			<div class="summary-card {'high' if report.total_leaks > 0 else ''}">
				<div class="value">{report.total_leaks}</div>
				<div class="label">Memory Leaks</div>
			</div>
			<div class="summary-card {'high' if report.bytes_definitely_lost > 0 else ''}">
				<div class="value">{report.bytes_definitely_lost:,}</div>
				<div class="label">Bytes Definitely Lost</div>
			</div>
			<div class="summary-card {'medium' if report.bytes_indirectly_lost > 0 else ''}">
				<div class="value">{report.bytes_indirectly_lost:,}</div>
				<div class="label">Bytes Indirectly Lost</div>
			</div>
			<div class="summary-card {'low' if report.bytes_possibly_lost > 0 else ''}">
				<div class="value">{report.bytes_possibly_lost:,}</div>
				<div class="label">Bytes Possibly Lost</div>
			</div>
			<div class="summary-card">
				<div class="value">{report.execution_time:.2f}s</div>
				<div class="label">Execution Time</div>
			</div>
		</div>
		
		<div class="content">
"""
		
		# Group errors by type
		if errors_by_type:
			for error_type, errors in sorted(errors_by_type.items()):
				html += f"""
			<div class="error-group">
				<div class="error-group-header">
					<span>{error_type}</span>
					<span class="error-count">{len(errors)} error(s)</span>
				</div>
"""
				
				for idx, error in enumerate(errors):
					severity_class = f"severity-{error.severity}"
					html += f"""
				<div class="error-item">
					<div class="error-header">
						<div class="error-type">Error #{idx + 1}</div>
						<span class="severity-badge {severity_class}">{error.severity.upper()}</span>
					</div>
					
					<div class="error-details">
						{HTMLReportGenerator._escape_html(error.what)}
					</div>
"""
					
					if error.source_file:
						html += f"""
					<div class="source-info">
						<strong>Source:</strong> {error.source_file}
"""
						if error.source_line:
							html += f" <strong>Line:</strong> {error.source_line}"
						if error.function:
							html += f" <strong>Function:</strong> {error.function}"
						html += """
					</div>
"""
					
					if error.bytes_lost:
						html += f"""
					<div class="source-info">
						<span class="bytes-lost">Bytes Lost: {error.bytes_lost:,}</span>
					</div>
"""
					
					if error.stack_trace:
						html += """
					<div class="stack-trace">
						<strong>Stack Trace:</strong>
"""
						for frame in error.stack_trace[:10]:  # Limit to 10 frames
							function = frame.get('function', 'unknown')
							file_info = frame.get('file', '')
							line_info = frame.get('line', '')
							
							if file_info or line_info:
								location = f"{file_info}:{line_info}" if line_info else file_info
							else:
								location = frame.get('raw', '')
							
							html += f"""
						<div class="stack-frame">
							<strong>{function}</strong> {location}
						</div>
"""
						html += """
					</div>
"""
					
					html += """
				</div>
"""
				
				html += """
			</div>
"""
		else:
			html += """
			<div style="text-align: center; padding: 60px; color: #666;">
				<h2 style="color: #689f38;">✓ No Errors Found!</h2>
				<p style="margin-top: 10px;">Valgrind did not detect any memory errors or leaks.</p>
			</div>
"""
		
		html += f"""
		</div>
		
		<div class="footer">
			<p>Generated by pg_valgrind.py • Valgrind Version: {report.valgrind_version or 'unknown'}</p>
			<p>Command: {HTMLReportGenerator._escape_html(report.command)}</p>
		</div>
	</div>
</body>
</html>
"""
		
		return html
	
	@staticmethod
	def _escape_html(text: str) -> str:
		"""Escape HTML special characters"""
		return (text.replace('&', '&amp;')
		            .replace('<', '&lt;')
		            .replace('>', '&gt;')
		            .replace('"', '&quot;')
		            .replace("'", '&#39;'))


class TextReportGenerator:
	"""Generates clean text reports"""
	
	@staticmethod
	def generate(report: ValgrindReport, output_file: Optional[str] = None):
		"""Generate text report"""
		lines = []
		
		lines.append("=" * 80)
		lines.append("NeuronDB Valgrind Report")
		lines.append("=" * 80)
		lines.append(f"Timestamp: {report.timestamp}")
		lines.append(f"Command: {report.command}")
		lines.append("")
		
		# Summary
		lines.append("SUMMARY")
		lines.append("-" * 80)
		lines.append(f"  Total Errors:     {report.total_errors}")
		lines.append(f"  Memory Leaks:     {report.total_leaks}")
		lines.append(f"  Bytes Leaked:     {report.bytes_leaked:,}")
		lines.append(f"  Definitely Lost:  {report.bytes_definitely_lost:,}")
		lines.append(f"  Indirectly Lost:  {report.bytes_indirectly_lost:,}")
		lines.append(f"  Possibly Lost:    {report.bytes_possibly_lost:,}")
		lines.append(f"  Execution Time:   {report.execution_time:.2f}s")
		lines.append("")
		
		# Errors by type
		errors_by_type = defaultdict(list)
		for error in report.errors:
			errors_by_type[error.error_type].append(error)
		
		if errors_by_type:
			lines.append("ERRORS BY TYPE")
			lines.append("-" * 80)
			for error_type, errors in sorted(errors_by_type.items()):
				lines.append(f"  {error_type}: {len(errors)} occurrence(s)")
			lines.append("")
			
			# Detailed errors
			lines.append("DETAILED ERRORS")
			lines.append("=" * 80)
			
			for error_type, errors in sorted(errors_by_type.items()):
				lines.append("")
				lines.append(f"[{error_type}] ({len(errors)} error(s))")
				lines.append("-" * 80)
				
				for idx, error in enumerate(errors, 1):
					lines.append(f"\nError #{idx} [{error.severity.upper()}]")
					lines.append(f"  {error.what}")
					
					if error.source_file:
						lines.append(f"  Source: {error.source_file}")
						if error.source_line:
							lines.append(f"  Line: {error.source_line}")
					
					if error.function:
						lines.append(f"  Function: {error.function}")
					
					if error.bytes_lost:
						lines.append(f"  Bytes Lost: {error.bytes_lost:,}")
					
					if error.stack_trace:
						lines.append("  Stack Trace:")
						for frame in error.stack_trace[:5]:  # Limit to 5 frames
							function = frame.get('function', 'unknown')
							file_info = frame.get('file', '')
							line_info = frame.get('line', '')
							if file_info or line_info:
								location = f"{file_info}:{line_info}" if line_info else file_info
							else:
								location = frame.get('raw', '')
							lines.append(f"    → {function} ({location})")
		else:
			lines.append("✓ No errors found!")
			lines.append("")
		
		lines.append("=" * 80)
		
		text = "\n".join(lines)
		
		if output_file:
			with open(output_file, 'w', encoding='utf-8') as f:
				f.write(text)
		else:
			print(text)
		
		return text


class JSONReportGenerator:
	"""Generates JSON reports"""
	
	@staticmethod
	def generate(report: ValgrindReport, output_file: str):
		"""Generate JSON report"""
		with open(output_file, 'w', encoding='utf-8') as f:
			json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)


# ============================================================================
# Main Valgrind Runner
# ============================================================================

class ValgrindRunner:
	"""Runs PostgreSQL under Valgrind and collects results"""
	
	def __init__(self, pgdata: str, postgres_bin: str, verbose: bool = False):
		self.pgdata = Path(pgdata).resolve()
		self.postgres_bin = postgres_bin
		self.verbose = verbose
		self.pg_detector = PostgreSQLDetector()
	
	def check_valgrind(self) -> bool:
		"""Check if Valgrind is available"""
		try:
			result = subprocess.run(
				['valgrind', '--version'],
				capture_output=True,
				text=True,
				timeout=5
			)
			return result.returncode == 0
		except (subprocess.TimeoutExpired, FileNotFoundError):
			return False
	
	def build_valgrind_command(
		self,
		leak_check: str = "yes",
		leak_resolution: str = "high",
		show_reachable: bool = True,
		gen_suppressions: bool = False,
		track_origins: bool = True,
		extra_opts: Optional[List[str]] = None
	) -> List[str]:
		"""Build Valgrind command with options"""
		cmd = ['valgrind']
		
		# Basic options
		cmd.extend([
			'--tool=memcheck',
			f'--leak-check={leak_check}',
			f'--leak-resolution={leak_resolution}',
			'--show-leak-kinds=all',
			'--errors-for-leak-kinds=all',
		])
		
		if show_reachable:
			cmd.append('--show-reachable=yes')
		
		if track_origins:
			cmd.append('--track-origins=yes')
		
		if gen_suppressions:
			cmd.append('--gen-suppressions=all')
		
		# Log file
		log_file = self.pgdata / 'valgrind.log'
		cmd.extend(['--log-file=' + str(log_file)])
		
		# XML output
		xml_file = self.pgdata / 'valgrind.xml'
		cmd.extend(['--xml=yes', '--xml-file=' + str(xml_file)])
		
		# Suppression file if exists
		supp_file = Path(__file__).parent / 'valgrind.supp'
		if supp_file.exists():
			cmd.append(f'--suppressions={supp_file}')
		
		if extra_opts:
			cmd.extend(extra_opts)
		
		# Postgres binary
		cmd.append(self.postgres_bin)
		
		return cmd
	
	def run_with_sql(
		self,
		sql_file: Optional[str] = None,
		sql_command: Optional[str] = None,
		dbname: str = "postgres",
		valgrind_opts: Optional[Dict[str, Any]] = None,
		timeout: int = 300
	) -> Tuple[ValgrindReport, int]:
		"""Run PostgreSQL under Valgrind and execute SQL"""
		
		if not self.check_valgrind():
			raise RuntimeError("Valgrind is not installed or not in PATH")
		
		# Build Valgrind command
		opts = valgrind_opts or {}
		valgrind_cmd = self.build_valgrind_command(**opts)
		
		if self.verbose:
			print(f"Valgrind command: {' '.join(valgrind_cmd)}")
		
		# Start PostgreSQL under Valgrind
		env = os.environ.copy()
		env['PGDATA'] = str(self.pgdata)
		
		start_time = time.time()
		
		process = subprocess.Popen(
			valgrind_cmd,
			env=env,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			cwd=str(self.pgdata.parent)
		)
		
		# Wait for PostgreSQL to start
		time.sleep(5)
		
		# Execute SQL if provided
		if sql_file or sql_command:
			psql_cmd = ['psql', '-d', dbname, '-q', '-v', 'ON_ERROR_STOP=1']
			
			if sql_file:
				psql_cmd.extend(['-f', sql_file])
			elif sql_command:
				psql_cmd.append('-c')
				psql_cmd.append(sql_command)
			
			try:
				sql_result = subprocess.run(
					psql_cmd,
					capture_output=True,
					text=True,
					timeout=timeout
				)
				
				if self.verbose:
					if sql_result.stdout:
						print("SQL Output:")
						print(sql_result.stdout)
					if sql_result.stderr:
						print("SQL Errors:")
						print(sql_result.stderr)
			except subprocess.TimeoutExpired:
				print("⚠ SQL execution timed out")
			except Exception as e:
				print(f"⚠ Error executing SQL: {e}")
		
		# Wait for process to finish or terminate
		try:
			process.wait(timeout=timeout)
			returncode = process.returncode
		except subprocess.TimeoutExpired:
			process.terminate()
			try:
				process.wait(timeout=10)
			except subprocess.TimeoutExpired:
				process.kill()
			returncode = -1
		
		execution_time = time.time() - start_time
		
		# Read Valgrind output
		log_file = self.pgdata / 'valgrind.log'
		if log_file.exists():
			with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
				valgrind_output = f.read()
		else:
			valgrind_output = ""
			if process.stderr:
				valgrind_output = process.stderr.read().decode('utf-8', errors='ignore')
		
		# Parse report
		parser = ValgrindParser(valgrind_output)
		report = parser.parse()
		report.execution_time = execution_time
		report.command = ' '.join(valgrind_cmd)
		
		return report, returncode


# ============================================================================
# CLI Interface
# ============================================================================

def main():
	"""Main entry point"""
	parser = argparse.ArgumentParser(
		description="Run PostgreSQL under Valgrind and generate detailed memory reports",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog=__doc__
	)
	
	parser.add_argument(
		'--pgdata',
		required=True,
		help='PostgreSQL data directory'
	)
	
	parser.add_argument(
		'--postgres',
		help='Path to postgres binary (auto-detected if not specified)'
	)
	
	parser.add_argument(
		'--sql-file',
		help='SQL file to execute after PostgreSQL starts'
	)
	
	parser.add_argument(
		'--sql-command',
		help='SQL command to execute after PostgreSQL starts'
	)
	
	parser.add_argument(
		'--dbname',
		default='postgres',
		help='Database name (default: postgres)'
	)
	
	parser.add_argument(
		'--html-report',
		help='Generate HTML report to this file'
	)
	
	parser.add_argument(
		'--json-report',
		help='Generate JSON report to this file'
	)
	
	parser.add_argument(
		'--text-report',
		help='Generate text report to this file (default: stdout)'
	)
	
	parser.add_argument(
		'--leak-check',
		default='yes',
		choices=['yes', 'no', 'full', 'summary'],
		help='Leak check level (default: yes)'
	)
	
	parser.add_argument(
		'--leak-resolution',
		default='high',
		choices=['low', 'med', 'high'],
		help='Leak resolution (default: high)'
	)
	
	parser.add_argument(
		'--track-origins',
		action='store_true',
		help='Track origins of uninitialised values'
	)
	
	parser.add_argument(
		'--timeout',
		type=int,
		default=300,
		help='Timeout in seconds (default: 300)'
	)
	
	parser.add_argument(
		'--verbose', '-v',
		action='store_true',
		help='Verbose output'
	)
	
	args = parser.parse_args()
	
	# Detect PostgreSQL
	pg_detector = PostgreSQLDetector()
	pg_config = pg_detector.find_pg_config()
	
	if not pg_config:
		print("✗ Error: Could not find pg_config. Set PG_CONFIG environment variable.")
		sys.exit(1)
	
	if args.verbose:
		print(f"✓ Found pg_config: {pg_config}")
	
	pg_vars = pg_detector.get_pg_vars(pg_config)
	postgres_bin = args.postgres or pg_detector.find_postgres(pg_vars)
	
	if not postgres_bin:
		print("✗ Error: Could not find postgres binary.")
		sys.exit(1)
	
	if args.verbose:
		print(f"✓ Found postgres: {postgres_bin}")
	
	# Create runner
	runner = ValgrindRunner(args.pgdata, postgres_bin, args.verbose)
	
	# Check Valgrind
	if not runner.check_valgrind():
		print("✗ Error: Valgrind is not installed or not in PATH")
		print("   Install with: sudo apt-get install valgrind")
		sys.exit(1)
	
	if args.verbose:
		print("✓ Valgrind found")
	
	# Build Valgrind options
	valgrind_opts = {
		'leak_check': args.leak_check,
		'leak_resolution': args.leak_resolution,
		'track_origins': args.track_origins,
	}
	
	# Run
	print("Starting PostgreSQL under Valgrind...")
	
	try:
		report, returncode = runner.run_with_sql(
			sql_file=args.sql_file,
			sql_command=args.sql_command,
			dbname=args.dbname,
			valgrind_opts=valgrind_opts,
			timeout=args.timeout
		)
		
		print(f"\n{'='*80}")
		print("Valgrind Report Summary")
		print(f"{'='*80}")
		print(f"  Total Errors:    {report.total_errors}")
		print(f"  Memory Leaks:    {report.total_leaks}")
		print(f"  Bytes Lost:      {report.bytes_leaked:,}")
		print(f"  Execution Time:  {report.execution_time:.2f}s")
		print()
		
		# Generate reports
		if args.html_report:
			print(f"Generating HTML report: {args.html_report}")
			HTMLReportGenerator.generate(report, args.html_report)
			print("✓ HTML report generated")
		
		if args.json_report:
			print(f"Generating JSON report: {args.json_report}")
			JSONReportGenerator.generate(report, args.json_report)
			print("✓ JSON report generated")
		
		if args.text_report:
			print(f"Generating text report: {args.text_report}")
			TextReportGenerator.generate(report, args.text_report)
			print("✓ Text report generated")
		else:
			# Print text report to stdout
			TextReportGenerator.generate(report)
		
		# Exit code based on errors
		if report.total_errors > 0:
			print(f"\n⚠ Found {report.total_errors} error(s)")
			sys.exit(1)
		else:
			print("\n✓ No errors found!")
			sys.exit(0)
	
	except KeyboardInterrupt:
		print("\n\n⚠ Interrupted by user")
		sys.exit(130)
	except Exception as e:
		print(f"\n✗ Error: {e}")
		if args.verbose:
			import traceback
			traceback.print_exc()
		sys.exit(1)


if __name__ == '__main__':
	main()

