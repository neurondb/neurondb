#!/usr/bin/env python3
"""
PostgreSQL Identifier and Code Style Checker
=============================================

A comprehensive tool for running pgident on PostgreSQL extensions and
generating detailed code style and identifier reports. Supports multiple
report formats including HTML, JSON, and text summaries.

Usage:
    python pg_iden.py [OPTIONS] [SOURCE_FILES...]

Examples:
    # Check all C files in src directory
    python pg_iden.py --source-dir src/ --extensions .c .h

    # Generate HTML report with detailed issues
    python pg_iden.py --html-report report.html --verbose src/*.c

    # Check specific files
    python pg_iden.py --json-report report.json src/ml/ml_knn.c src/vector/vector_types.c
"""

import os
import sys
import json
import subprocess
import argparse
import tempfile
import shutil
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
from enum import Enum


# ============================================================================
# Configuration and Constants
# ============================================================================

class IssueType(Enum):
	"""Types of identifier/style issues"""
	INVALID_IDENTIFIER = "Invalid Identifier"
	NAMING_CONVENTION = "Naming Convention"
	RESERVED_KEYWORD = "Reserved Keyword"
	LENGTH_VIOLATION = "Length Violation"
	CHARACTER_VIOLATION = "Character Violation"
	CASE_VIOLATION = "Case Violation"
	UNDERSCORE_VIOLATION = "Underscore Violation"
	PREFIX_VIOLATION = "Prefix Violation"
	SUFFIX_VIOLATION = "Suffix Violation"


@dataclass
class IdentifierIssue:
	"""Represents a single identifier/style issue"""
	file: str
	line: int
	column: int
	identifier: str
	issue_type: str
	message: str
	severity: str = "medium"
	suggestion: Optional[str] = None
	context: Optional[str] = None
	rule: Optional[str] = None
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary for JSON serialization"""
		return asdict(self)


@dataclass
class IdentifierReport:
	"""Complete identifier report with all issues and summary"""
	timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
	files_analyzed: int = 0
	issues: List[IdentifierIssue] = field(default_factory=list)
	total_issues: int = 0
	issues_by_type: Dict[str, int] = field(default_factory=dict)
	issues_by_file: Dict[str, int] = field(default_factory=dict)
	issues_by_severity: Dict[str, int] = field(default_factory=dict)
	execution_time: float = 0.0
	pgident_version: str = ""
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary for JSON serialization"""
		result = asdict(self)
		result['issues'] = [e.to_dict() for e in self.issues]
		return result


# ============================================================================
# PostgreSQL Identifier Checker
# ============================================================================

class PostgreSQLIdentifierChecker:
	"""Checks PostgreSQL identifiers and code style"""
	
	# PostgreSQL reserved keywords
	RESERVED_KEYWORDS = {
		'select', 'insert', 'update', 'delete', 'create', 'drop', 'alter',
		'table', 'index', 'view', 'function', 'procedure', 'trigger',
		'where', 'from', 'join', 'inner', 'outer', 'left', 'right',
		'group', 'order', 'by', 'having', 'union', 'intersect', 'except',
		'case', 'when', 'then', 'else', 'end', 'if', 'else', 'while',
		'for', 'loop', 'return', 'begin', 'commit', 'rollback', 'savepoint',
		'grant', 'revoke', 'user', 'role', 'database', 'schema', 'public'
	}
	
	# PostgreSQL naming conventions
	MAX_IDENTIFIER_LENGTH = 63  # PostgreSQL limit
	VALID_IDENTIFIER_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
	
	@staticmethod
	def check_identifier(identifier: str, context: Optional[str] = None) -> List[Tuple[str, str]]:
		"""Check a single identifier for issues"""
		issues = []
		
		# Check length
		if len(identifier) > PostgreSQLIdentifierChecker.MAX_IDENTIFIER_LENGTH:
			issues.append((
				IssueType.LENGTH_VIOLATION.value,
				f"Identifier '{identifier}' exceeds maximum length of {PostgreSQLIdentifierChecker.MAX_IDENTIFIER_LENGTH} characters"
			))
		
		# Check valid characters
		if not PostgreSQLIdentifierChecker.VALID_IDENTIFIER_PATTERN.match(identifier):
			issues.append((
				IssueType.CHARACTER_VIOLATION.value,
				f"Identifier '{identifier}' contains invalid characters"
			))
		
		# Check reserved keywords
		if identifier.lower() in PostgreSQLIdentifierChecker.RESERVED_KEYWORDS:
			issues.append((
				IssueType.RESERVED_KEYWORD.value,
				f"Identifier '{identifier}' is a PostgreSQL reserved keyword"
			))
		
		# Check naming conventions
		if identifier and identifier[0].isdigit():
			issues.append((
				IssueType.NAMING_CONVENTION.value,
				f"Identifier '{identifier}' cannot start with a digit"
			))
		
		# Check for double underscores (often indicates internal/private)
		if '__' in identifier:
			issues.append((
				IssueType.UNDERSCORE_VIOLATION.value,
				f"Identifier '{identifier}' contains double underscores (may indicate internal naming)"
			))
		
		return issues


# ============================================================================
# pgident Tool Integration
# ============================================================================

class PgidentTool:
	"""Wrapper for pgident tool"""
	
	@staticmethod
	def check_tool() -> bool:
		"""Check if pgident tool is available"""
		try:
			result = subprocess.run(
				['pgident', '--version'],
				capture_output=True,
				text=True,
				timeout=5
			)
			return result.returncode == 0
		except (subprocess.TimeoutExpired, FileNotFoundError):
			# Try alternative locations
			common_paths = [
				'/usr/local/bin/pgident',
				'/usr/bin/pgident',
				'/opt/postgresql/bin/pgident',
			]
			for path in common_paths:
				if os.path.isfile(path) and os.access(path, os.X_OK):
					return True
			return False
	
	@staticmethod
	def find_pgident() -> Optional[str]:
		"""Find pgident binary"""
		# Check PATH
		pgident = shutil.which('pgident')
		if pgident:
			return pgident
		
		# Check common locations
		common_paths = [
			'/usr/local/bin/pgident',
			'/usr/bin/pgident',
			'/opt/postgresql/bin/pgident',
		]
		for path in common_paths:
			if os.path.isfile(path) and os.access(path, os.X_OK):
				return path
		
		return None
	
	@staticmethod
	def analyze_file(
		file_path: str,
		pgident_bin: Optional[str] = None,
		verbose: bool = False
	) -> Tuple[str, int]:
		"""Run pgident on a single file"""
		if not pgident_bin:
			pgident_bin = PgidentTool.find_pgident()
		
		if not pgident_bin:
			return "pgident tool not found", -1
		
		cmd = [pgident_bin, file_path]
		
		if verbose:
			print(f"Running: {' '.join(cmd)}")
		
		try:
			result = subprocess.run(
				cmd,
				capture_output=True,
				text=True,
				timeout=60
			)
			
			return result.stdout + result.stderr, result.returncode
		except subprocess.TimeoutExpired:
			return "Analysis timed out", -1
		except Exception as e:
			return f"Error: {e}", -1
	
	@staticmethod
	def analyze_directory(
		source_dir: str,
		extensions: List[str] = None,
		pgident_bin: Optional[str] = None,
		verbose: bool = False
	) -> Tuple[str, int]:
		"""Run pgident on a directory"""
		if not pgident_bin:
			pgident_bin = PgidentTool.find_pgident()
		
		if not pgident_bin:
			return "pgident tool not found", -1
		
		if extensions is None:
			extensions = ['.c', '.h']
		
		# Find all files with specified extensions
		files = []
		for ext in extensions:
			files.extend(Path(source_dir).rglob(f'*{ext}'))
		
		if not files:
			return "No files found to analyze", 0
		
		all_output = []
		total_errors = 0
		
		for file_path in files:
			output, returncode = PgidentTool.analyze_file(str(file_path), pgident_bin, verbose)
			all_output.append(f"=== {file_path} ===\n{output}")
			if returncode != 0:
				total_errors += 1
		
		return "\n".join(all_output), total_errors


# ============================================================================
# Identifier Parser
# ============================================================================

class IdentifierParser:
	"""Parses pgident output and code analysis into structured report"""
	
	# Patterns for parsing pgident output
	ERROR_PATTERN = re.compile(
		r'(.+):(\d+):(\d+):\s*(.+)'
	)
	
	IDENTIFIER_PATTERN = re.compile(
		r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
	)
	
	def __init__(self, output: str, source_files: List[str] = None):
		self.output = output
		self.source_files = source_files or []
		self.issues: List[IdentifierIssue] = []
		self.report = IdentifierReport()
	
	def parse(self) -> IdentifierReport:
		"""Parse the output"""
		# Parse pgident output
		self._parse_pgident_output()
		
		# Also do custom analysis if files provided
		if self.source_files:
			self._analyze_files()
		
		self._build_report()
		return self.report
	
	def _parse_pgident_output(self):
		"""Parse pgident tool output"""
		lines = self.output.split('\n')
		
		for line in lines:
			match = self.ERROR_PATTERN.match(line)
			if match:
				file_path = match.group(1)
				line_num = int(match.group(2))
				column_num = int(match.group(3))
				message = match.group(4)
				
				# Extract identifier from message if possible
				identifier_match = re.search(r"'([^']+)'", message)
				identifier = identifier_match.group(1) if identifier_match else "unknown"
				
				# Determine issue type
				issue_type = self._classify_issue(message)
				
				issue = IdentifierIssue(
					file=file_path,
					line=line_num,
					column=column_num,
					identifier=identifier,
					issue_type=issue_type,
					message=message,
					severity=self._determine_severity(issue_type, message)
				)
				
				self.issues.append(issue)
	
	def _analyze_files(self):
		"""Analyze source files directly"""
		checker = PostgreSQLIdentifierChecker()
		
		for file_path in self.source_files:
			if not os.path.isfile(file_path):
				continue
			
			try:
				with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
					lines = f.readlines()
				
				for line_num, line in enumerate(lines, 1):
					# Find identifiers in the line
					identifiers = self.IDENTIFIER_PATTERN.findall(line)
					
					for identifier in identifiers:
						# Skip if it's a keyword or common pattern
						if len(identifier) < 2 or identifier.lower() in ['if', 'for', 'while', 'int', 'char', 'void']:
							continue
						
						# Check identifier
						issues = checker.check_identifier(identifier, line)
						
						for issue_type, message in issues:
							column = line.find(identifier)
							
							issue = IdentifierIssue(
								file=file_path,
								line=line_num,
								column=column if column >= 0 else 0,
								identifier=identifier,
								issue_type=issue_type,
								message=message,
								severity=self._determine_severity(issue_type, message),
								context=line.strip()
							)
							
							self.issues.append(issue)
			except Exception as e:
				print(f"Error analyzing {file_path}: {e}")
	
	def _classify_issue(self, message: str) -> str:
		"""Classify issue type from message"""
		message_lower = message.lower()
		
		if 'reserved' in message_lower or 'keyword' in message_lower:
			return IssueType.RESERVED_KEYWORD.value
		elif 'length' in message_lower or 'exceeds' in message_lower:
			return IssueType.LENGTH_VIOLATION.value
		elif 'invalid' in message_lower or 'character' in message_lower:
			return IssueType.CHARACTER_VIOLATION.value
		elif 'naming' in message_lower or 'convention' in message_lower:
			return IssueType.NAMING_CONVENTION.value
		elif 'underscore' in message_lower:
			return IssueType.UNDERSCORE_VIOLATION.value
		else:
			return IssueType.INVALID_IDENTIFIER.value
	
	def _determine_severity(self, issue_type: str, message: str) -> str:
		"""Determine severity based on issue type"""
		if issue_type == IssueType.RESERVED_KEYWORD.value:
			return "critical"
		elif issue_type == IssueType.LENGTH_VIOLATION.value:
			return "high"
		elif issue_type == IssueType.CHARACTER_VIOLATION.value:
			return "high"
		elif issue_type == IssueType.NAMING_CONVENTION.value:
			return "medium"
		else:
			return "low"
	
	def _build_report(self):
		"""Build report summary"""
		self.report.issues = self.issues
		self.report.total_issues = len(self.issues)
		self.report.files_analyzed = len(set(issue.file for issue in self.issues))
		
		# Count by type
		for issue in self.issues:
			issue_type = issue.issue_type
			self.report.issues_by_type[issue_type] = \
				self.report.issues_by_type.get(issue_type, 0) + 1
		
		# Count by file
		for issue in self.issues:
			file_path = issue.file
			self.report.issues_by_file[file_path] = \
				self.report.issues_by_file.get(file_path, 0) + 1
		
		# Count by severity
		for issue in self.issues:
			severity = issue.severity
			self.report.issues_by_severity[severity] = \
				self.report.issues_by_severity.get(severity, 0) + 1


# ============================================================================
# Report Generators
# ============================================================================

class HTMLReportGenerator:
	"""Generates beautiful HTML reports from identifier data"""
	
	@staticmethod
	def generate(report: IdentifierReport, output_file: str):
		"""Generate HTML report"""
		html = HTMLReportGenerator._build_html(report)
		
		with open(output_file, 'w', encoding='utf-8') as f:
			f.write(html)
	
	@staticmethod
	def _build_html(report: IdentifierReport) -> str:
		"""Build complete HTML document"""
		severity_colors = {
			'critical': '#d32f2f',
			'high': '#f57c00',
			'medium': '#fbc02d',
			'low': '#689f38',
			'unknown': '#757575'
		}
		
		issues_by_file = defaultdict(list)
		for issue in report.issues:
			issues_by_file[issue.file].append(issue)
		
		html = f"""<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8">
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<title>NeuronDB Identifier Report - {report.timestamp}</title>
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
		.summary {{
			display: grid;
			grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
			gap: 20px;
			padding: 30px;
			background: #f5f7fa;
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
		.content {{
			padding: 30px;
		}}
		.file-group {{
			margin-bottom: 40px;
		}}
		.file-group-header {{
			background: #667eea;
			color: white;
			padding: 15px 20px;
			border-radius: 8px 8px 0 0;
			font-size: 1.3em;
			font-weight: 600;
		}}
		.issue-item {{
			background: white;
			border: 1px solid #e1e8ed;
			border-top: none;
			padding: 20px;
		}}
		.issue-header {{
			display: flex;
			justify-content: space-between;
			align-items: center;
			margin-bottom: 15px;
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
		.source-info {{
			display: flex;
			gap: 20px;
			margin-top: 10px;
			font-size: 0.9em;
			color: #666;
		}}
		.context {{
			background: #f8f9fa;
			padding: 10px;
			border-radius: 4px;
			font-family: monospace;
			font-size: 0.9em;
			margin-top: 10px;
		}}
	</style>
</head>
<body>
	<div class="container">
		<div class="header">
			<h1>NeuronDB Identifier Report</h1>
			<div class="subtitle">Code Style & Identifier Analysis • {report.timestamp}</div>
		</div>
		
		<div class="summary">
			<div class="summary-card">
				<div class="value">{report.total_issues}</div>
				<div class="label">Total Issues</div>
			</div>
			<div class="summary-card">
				<div class="value">{report.files_analyzed}</div>
				<div class="label">Files Analyzed</div>
			</div>
"""
		
		# Add severity cards
		for severity, count in sorted(report.issues_by_severity.items()):
			html += f"""
			<div class="summary-card">
				<div class="value">{count}</div>
				<div class="label">{severity.upper()} Issues</div>
			</div>
"""
		
		html += """
		</div>
		
		<div class="content">
"""
		
		# Group issues by file
		if issues_by_file:
			for file_path, issues in sorted(issues_by_file.items()):
				html += f"""
			<div class="file-group">
				<div class="file-group-header">
					{file_path} ({len(issues)} issue(s))
				</div>
"""
				
				for idx, issue in enumerate(issues):
					severity_class = f"severity-{issue.severity}"
					html += f"""
				<div class="issue-item">
					<div class="issue-header">
						<div><strong>Line {issue.line}:{issue.column}</strong> - {issue.identifier}</div>
						<span class="severity-badge {severity_class}">{issue.severity.upper()}</span>
					</div>
					<div><strong>{issue.issue_type}:</strong> {HTMLReportGenerator._escape_html(issue.message)}</div>
					<div class="source-info">
						<span><strong>Type:</strong> {issue.issue_type}</span>
					</div>
"""
					if issue.context:
						html += f"""
					<div class="context">{HTMLReportGenerator._escape_html(issue.context)}</div>
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
				<h2 style="color: #689f38;">✓ No Issues Found!</h2>
			</div>
"""
		
		html += """
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
	def generate(report: IdentifierReport, output_file: Optional[str] = None):
		"""Generate text report"""
		lines = []
		
		lines.append("=" * 80)
		lines.append("NeuronDB Identifier Report")
		lines.append("=" * 80)
		lines.append(f"Timestamp: {report.timestamp}")
		lines.append("")
		
		# Summary
		lines.append("SUMMARY")
		lines.append("-" * 80)
		lines.append(f"  Total Issues:     {report.total_issues}")
		lines.append(f"  Files Analyzed:   {report.files_analyzed}")
		for severity, count in sorted(report.issues_by_severity.items()):
			lines.append(f"  {severity.upper()} Issues:     {count}")
		lines.append("")
		
		# Issues by type
		if report.issues_by_type:
			lines.append("ISSUES BY TYPE")
			lines.append("-" * 80)
			for issue_type, count in sorted(report.issues_by_type.items()):
				lines.append(f"  {issue_type}: {count}")
			lines.append("")
		
		# Detailed issues
		if report.issues:
			lines.append("DETAILED ISSUES")
			lines.append("=" * 80)
			
			issues_by_file = defaultdict(list)
			for issue in report.issues:
				issues_by_file[issue.file].append(issue)
			
			for file_path, issues in sorted(issues_by_file.items()):
				lines.append("")
				lines.append(f"File: {file_path} ({len(issues)} issue(s))")
				lines.append("-" * 80)
				
				for issue in issues:
					lines.append(f"  Line {issue.line}:{issue.column} - {issue.identifier}")
					lines.append(f"    [{issue.severity.upper()}] {issue.issue_type}: {issue.message}")
					if issue.context:
						lines.append(f"    Context: {issue.context}")
		else:
			lines.append("✓ No issues found!")
		
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
	def generate(report: IdentifierReport, output_file: str):
		"""Generate JSON report"""
		with open(output_file, 'w', encoding='utf-8') as f:
			json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)


# ============================================================================
# Main Identifier Runner
# ============================================================================

class IdentifierRunner:
	"""Runs identifier analysis and collects results"""
	
	def __init__(self, verbose: bool = False):
		self.verbose = verbose
		self.pgident_bin = PgidentTool.find_pgident()
	
	def run_analysis(
		self,
		source_files: List[str] = None,
		source_dir: Optional[str] = None,
		extensions: List[str] = None
	) -> IdentifierReport:
		"""Run identifier analysis"""
		start_time = time.time()
		
		if source_files:
			return self._analyze_files(source_files)
		elif source_dir:
			return self._analyze_directory(source_dir, extensions)
		else:
			raise ValueError("Either source_files or source_dir must be provided")
	
	def _analyze_files(self, source_files: List[str]) -> IdentifierReport:
		"""Analyze specific files"""
		all_output = []
		
		for file_path in source_files:
			if not os.path.isfile(file_path):
				if self.verbose:
					print(f"Warning: Skipping non-existent file: {file_path}")
				continue
			
			if self.pgident_bin and PgidentTool.check_tool():
				output, _ = PgidentTool.analyze_file(file_path, self.pgident_bin, self.verbose)
				all_output.append(output)
			else:
				if self.verbose:
					print(f"Warning: pgident not found, using custom analysis for: {file_path}")
		
		# Parse output
		parser = IdentifierParser("\n".join(all_output), source_files)
		report = parser.parse()
		
		report.execution_time = time.time() - start_time
		
		if self.pgident_bin:
			try:
				result = subprocess.run(
					[self.pgident_bin, '--version'],
					capture_output=True,
					text=True,
					timeout=5
				)
				report.pgident_version = result.stdout.strip()
			except:
				report.pgident_version = "unknown"
		
		return report
	
	def _analyze_directory(self, source_dir: str, extensions: List[str] = None) -> IdentifierReport:
		"""Analyze directory"""
		if extensions is None:
			extensions = ['.c', '.h']
		
		# Find all files
		files = []
		for ext in extensions:
			files.extend(Path(source_dir).rglob(f'*{ext}'))
		
		if not files:
			raise ValueError(f"No files found in {source_dir} with extensions {extensions}")
		
		return self._analyze_files([str(f) for f in files])


# ============================================================================
# CLI Interface
# ============================================================================

def main():
	"""Main entry point"""
	parser = argparse.ArgumentParser(
		description="Run pgident analysis and generate detailed identifier/style reports",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog=__doc__
	)
	
	parser.add_argument(
		'source_files',
		nargs='*',
		help='Source files to analyze'
	)
	
	parser.add_argument(
		'--source-dir',
		help='Source directory to analyze (recursive)'
	)
	
	parser.add_argument(
		'--extensions',
		nargs='+',
		default=['.c', '.h'],
		help='File extensions to analyze (default: .c .h)'
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
		'--verbose', '-v',
		action='store_true',
		help='Verbose output'
	)
	
	args = parser.parse_args()
	
	# Check for pgident
	pgident_available = PgidentTool.check_tool()
	if not pgident_available and args.verbose:
		print("Warning: pgident tool not found, using custom identifier analysis")
	
	# Create runner
	runner = IdentifierRunner(args.verbose)
	
	# Run analysis
	print("Starting identifier analysis...")
	
	try:
		if args.source_dir:
			report = runner.run_analysis(
				source_dir=args.source_dir,
				extensions=args.extensions
			)
		elif args.source_files:
			report = runner.run_analysis(source_files=args.source_files)
		else:
			# Default to current directory
			report = runner.run_analysis(source_dir='.', extensions=args.extensions)
		
		print(f"\n{'='*80}")
		print("Identifier Report Summary")
		print(f"{'='*80}")
		print(f"  Total Issues:    {report.total_issues}")
		print(f"  Files Analyzed:  {report.files_analyzed}")
		for severity, count in sorted(report.issues_by_severity.items()):
			print(f"  {severity.upper()} Issues:    {count}")
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
		
		# Exit code based on issues
		critical_count = report.issues_by_severity.get('critical', 0)
		high_count = report.issues_by_severity.get('high', 0)
		
		if critical_count > 0 or high_count > 10:
			print(f"\nWarning: Found {report.total_issues} issue(s)")
			sys.exit(1)
		else:
			print("\n✓ Analysis complete!")
			sys.exit(0)
	
	except KeyboardInterrupt:
		print("\n\nWarning: Interrupted by user")
		sys.exit(130)
	except Exception as e:
		print(f"\n✗ Error: {e}")
		if args.verbose:
			import traceback
			traceback.print_exc()
		sys.exit(1)


if __name__ == '__main__':
	main()

