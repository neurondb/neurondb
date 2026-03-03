#!/usr/bin/env python3
"""
PostgreSQL Codacy Code Quality Analysis Tool
============================================

A comprehensive tool for running Codacy analysis on PostgreSQL extensions and
generating detailed code quality reports. Supports multiple report formats
including HTML, JSON, and text summaries.

Usage:
    python pg_codacy.py [OPTIONS] [SOURCE_DIRECTORY]

Examples:
    # Analyze current directory with Codacy
    python pg_codacy.py --api-token YOUR_TOKEN --project-id PROJECT_ID

    # Generate HTML report with detailed issues
    python pg_codacy.py --api-token YOUR_TOKEN --html-report report.html --verbose

    # Analyze specific source directory
    python pg_codacy.py --api-token YOUR_TOKEN --source-dir src/ --json-report report.json
"""

import os
import sys
import json
import subprocess
import argparse
import tempfile
import shutil
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
from enum import Enum


# ============================================================================
# Configuration and Constants
# ============================================================================

class IssueSeverity(Enum):
	"""Severity levels for Codacy issues"""
	ERROR = "Error"
	WARNING = "Warning"
	INFO = "Info"
	CRITICAL = "Critical"


class IssueCategory(Enum):
	"""Categories of code quality issues"""
	SECURITY = "Security"
	PERFORMANCE = "Performance"
	COMPLEXITY = "Complexity"
	BUG_RISK = "Bug Risk"
	CODE_STYLE = "Code Style"
	COMPATIBILITY = "Compatibility"
	DOCUMENTATION = "Documentation"
	DUPLICATION = "Duplication"
	UNUSED_CODE = "Unused Code"


@dataclass
class CodacyIssue:
	"""Represents a single Codacy issue"""
	file: str
	line: int
	message: str
	pattern_id: str
	level: str = "Info"
	category: str = "Code Style"
	rule: Optional[str] = None
	column: Optional[int] = None
	severity: str = "medium"
	first_occurrence: bool = True
	description: Optional[str] = None
	url: Optional[str] = None
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary for JSON serialization"""
		return asdict(self)


@dataclass
class CodacyReport:
	"""Complete Codacy report with all issues and summary"""
	timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
	project_id: Optional[str] = None
	commit_sha: Optional[str] = None
	issues: List[CodacyIssue] = field(default_factory=list)
	total_issues: int = 0
	issues_by_severity: Dict[str, int] = field(default_factory=dict)
	issues_by_category: Dict[str, int] = field(default_factory=dict)
	issues_by_file: Dict[str, int] = field(default_factory=dict)
	execution_time: float = 0.0
	codacy_version: str = ""
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary for JSON serialization"""
		result = asdict(self)
		result['issues'] = [e.to_dict() for e in self.issues]
		return result


# ============================================================================
# Codacy API Client
# ============================================================================

class CodacyAPIClient:
	"""Client for interacting with Codacy API"""
	
	BASE_URL = "https://api.codacy.com"
	
	def __init__(self, api_token: str, project_id: Optional[str] = None):
		self.api_token = api_token
		self.project_id = project_id
		self.session = requests.Session()
		self.session.headers.update({
			'api-token': api_token,
			'Content-Type': 'application/json'
		})
	
	def check_connection(self) -> bool:
		"""Check if API connection is valid"""
		try:
			response = self.session.get(f"{self.BASE_URL}/2.0/account")
			return response.status_code == 200
		except Exception:
			return False
	
	def get_projects(self) -> List[Dict[str, Any]]:
		"""Get list of projects"""
		try:
			response = self.session.get(f"{self.BASE_URL}/2.0/projects")
			if response.status_code == 200:
				return response.json()
			return []
		except Exception as e:
			print(f"Error fetching projects: {e}")
			return []
	
	def trigger_analysis(self, commit_sha: Optional[str] = None) -> bool:
		"""Trigger a new analysis for the project"""
		if not self.project_id:
			return False
		
		try:
			url = f"{self.BASE_URL}/2.0/commit/{self.project_id}"
			if commit_sha:
				url += f"/{commit_sha}"
			
			response = self.session.post(url)
			return response.status_code in [200, 201, 202]
		except Exception as e:
			print(f"Error triggering analysis: {e}")
			return False
	
	def get_commit_issues(self, commit_sha: Optional[str] = None) -> List[Dict[str, Any]]:
		"""Get issues for a specific commit"""
		if not self.project_id:
			return []
		
		try:
			url = f"{self.BASE_URL}/2.0/commit/{self.project_id}"
			if commit_sha:
				url += f"/{commit_sha}"
			url += "/resultsFinal"
			
			response = self.session.get(url)
			if response.status_code == 200:
				data = response.json()
				return data.get('files', [])
			return []
		except Exception as e:
			print(f"Error fetching commit issues: {e}")
			return []
	
	def get_file_issues(self, file_path: str) -> List[Dict[str, Any]]:
		"""Get issues for a specific file"""
		if not self.project_id:
			return []
		
		try:
			url = f"{self.BASE_URL}/2.0/commit/{self.project_id}/file/{file_path}/issues"
			response = self.session.get(url)
			if response.status_code == 200:
				return response.json()
			return []
		except Exception as e:
			print(f"Error fetching file issues: {e}")
			return []


# ============================================================================
# Codacy CLI Integration
# ============================================================================

class CodacyCLI:
	"""Wrapper for Codacy CLI tool"""
	
	@staticmethod
	def check_cli() -> bool:
		"""Check if Codacy CLI is available"""
		try:
			result = subprocess.run(
				['codacy', '--version'],
				capture_output=True,
				text=True,
				timeout=5
			)
			return result.returncode == 0
		except (subprocess.TimeoutExpired, FileNotFoundError):
			return False
	
	@staticmethod
	def analyze_directory(
		source_dir: str,
		api_token: Optional[str] = None,
		project_id: Optional[str] = None,
		verbose: bool = False
	) -> Tuple[str, int]:
		"""Run Codacy CLI analysis on a directory"""
		cmd = ['codacy', 'analyse']
		
		if api_token:
			cmd.extend(['--api-token', api_token])
		
		if project_id:
			cmd.extend(['--project-id', project_id])
		
		cmd.append(source_dir)
		
		if verbose:
			print(f"Running: {' '.join(cmd)}")
		
		try:
			result = subprocess.run(
				cmd,
				capture_output=True,
				text=True,
				cwd=source_dir,
				timeout=600
			)
			
			return result.stdout + result.stderr, result.returncode
		except subprocess.TimeoutExpired:
			return "Analysis timed out", -1
		except Exception as e:
			return f"Error: {e}", -1


# ============================================================================
# Codacy Parser
# ============================================================================

class CodacyParser:
	"""Parses Codacy output into structured report"""
	
	def __init__(self, data: Any, source_type: str = "api"):
		self.data = data
		self.source_type = source_type
		self.issues: List[CodacyIssue] = []
		self.report = CodacyReport()
	
	def parse(self) -> CodacyReport:
		"""Parse the Codacy data"""
		if self.source_type == "api":
			return self._parse_api_response()
		elif self.source_type == "cli":
			return self._parse_cli_output()
		else:
			return self.report
	
	def _parse_api_response(self) -> CodacyReport:
		"""Parse API response data"""
		if isinstance(self.data, list):
			# List of files with issues
			for file_data in self.data:
				if isinstance(file_data, dict):
					file_path = file_data.get('file', '')
					file_issues = file_data.get('issues', [])
					
					for issue_data in file_issues:
						issue = self._create_issue_from_api(file_path, issue_data)
						if issue:
							self.issues.append(issue)
		elif isinstance(self.data, dict):
			# Single file or structured response
			if 'issues' in self.data:
				file_path = self.data.get('file', '')
				for issue_data in self.data['issues']:
					issue = self._create_issue_from_api(file_path, issue_data)
					if issue:
						self.issues.append(issue)
		
		self._build_report()
		return self.report
	
	def _parse_cli_output(self) -> CodacyReport:
		"""Parse CLI output"""
		# CLI output parsing would go here
		# For now, we'll focus on API parsing
		self._build_report()
		return self.report
	
	def _create_issue_from_api(self, file_path: str, issue_data: Dict[str, Any]) -> Optional[CodacyIssue]:
		"""Create a CodacyIssue from API data"""
		try:
			issue = CodacyIssue(
				file=file_path,
				line=issue_data.get('line', 0),
				message=issue_data.get('message', ''),
				pattern_id=issue_data.get('patternId', ''),
				level=issue_data.get('level', 'Info'),
				category=issue_data.get('category', 'Code Style'),
				rule=issue_data.get('rule', None),
				column=issue_data.get('column', None),
				description=issue_data.get('description', None),
				url=issue_data.get('url', None)
			)
			
			# Set severity based on level
			level_lower = issue.level.lower()
			if level_lower in ['error', 'critical']:
				issue.severity = "critical"
			elif level_lower == 'warning':
				issue.severity = "high"
			elif level_lower == 'info':
				issue.severity = "medium"
			else:
				issue.severity = "low"
			
			return issue
		except Exception as e:
			print(f"Error creating issue: {e}")
			return None
	
	def _build_report(self):
		"""Build report summary"""
		self.report.issues = self.issues
		self.report.total_issues = len(self.issues)
		
		# Count by severity
		for issue in self.issues:
			severity = issue.severity
			self.report.issues_by_severity[severity] = \
				self.report.issues_by_severity.get(severity, 0) + 1
		
		# Count by category
		for issue in self.issues:
			category = issue.category
			self.report.issues_by_category[category] = \
				self.report.issues_by_category.get(category, 0) + 1
		
		# Count by file
		for issue in self.issues:
			file_path = issue.file
			self.report.issues_by_file[file_path] = \
				self.report.issues_by_file.get(file_path, 0) + 1


# ============================================================================
# Report Generators
# ============================================================================

class HTMLReportGenerator:
	"""Generates beautiful HTML reports from Codacy data"""
	
	@staticmethod
	def generate(report: CodacyReport, output_file: str):
		"""Generate HTML report"""
		html = HTMLReportGenerator._build_html(report)
		
		with open(output_file, 'w', encoding='utf-8') as f:
			f.write(html)
	
	@staticmethod
	def _build_html(report: CodacyReport) -> str:
		"""Build complete HTML document"""
		severity_colors = {
			'critical': '#d32f2f',
			'high': '#f57c00',
			'medium': '#fbc02d',
			'low': '#689f38',
			'unknown': '#757575'
		}
		
		issues_by_type = defaultdict(list)
		for issue in report.issues:
			issues_by_type[issue.category].append(issue)
		
		html = f"""<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8">
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<title>NeuronDB Codacy Report - {report.timestamp}</title>
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
		.issue-group {{
			margin-bottom: 40px;
		}}
		.issue-group-header {{
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
	</style>
</head>
<body>
	<div class="container">
		<div class="header">
			<h1>NeuronDB Codacy Report</h1>
			<div class="subtitle">Code Quality Analysis • {report.timestamp}</div>
		</div>
		
		<div class="summary">
			<div class="summary-card">
				<div class="value">{report.total_issues}</div>
				<div class="label">Total Issues</div>
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
		
		# Group issues by category
		if issues_by_type:
			for category, issues in sorted(issues_by_type.items()):
				html += f"""
			<div class="issue-group">
				<div class="issue-group-header">
					{category} ({len(issues)} issue(s))
				</div>
"""
				
				for idx, issue in enumerate(issues):
					severity_class = f"severity-{issue.severity}"
					html += f"""
				<div class="issue-item">
					<div class="issue-header">
						<div><strong>{issue.file}:{issue.line}</strong></div>
						<span class="severity-badge {severity_class}">{issue.severity.upper()}</span>
					</div>
					<div>{HTMLReportGenerator._escape_html(issue.message)}</div>
					<div class="source-info">
						<span><strong>Pattern:</strong> {issue.pattern_id}</span>
						<span><strong>Level:</strong> {issue.level}</span>
					</div>
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
	def generate(report: CodacyReport, output_file: Optional[str] = None):
		"""Generate text report"""
		lines = []
		
		lines.append("=" * 80)
		lines.append("NeuronDB Codacy Report")
		lines.append("=" * 80)
		lines.append(f"Timestamp: {report.timestamp}")
		lines.append("")
		
		# Summary
		lines.append("SUMMARY")
		lines.append("-" * 80)
		lines.append(f"  Total Issues:     {report.total_issues}")
		for severity, count in sorted(report.issues_by_severity.items()):
			lines.append(f"  {severity.upper()} Issues:     {count}")
		lines.append("")
		
		# Issues by category
		if report.issues_by_category:
			lines.append("ISSUES BY CATEGORY")
			lines.append("-" * 80)
			for category, count in sorted(report.issues_by_category.items()):
				lines.append(f"  {category}: {count}")
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
					lines.append(f"  Line {issue.line}: [{issue.severity.upper()}] {issue.message}")
					lines.append(f"    Pattern: {issue.pattern_id} | Category: {issue.category}")
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
	def generate(report: CodacyReport, output_file: str):
		"""Generate JSON report"""
		with open(output_file, 'w', encoding='utf-8') as f:
			json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)


# ============================================================================
# Main Codacy Runner
# ============================================================================

class CodacyRunner:
	"""Runs Codacy analysis and collects results"""
	
	def __init__(self, api_token: str, project_id: Optional[str] = None, verbose: bool = False):
		self.api_token = api_token
		self.project_id = project_id
		self.verbose = verbose
		self.api_client = CodacyAPIClient(api_token, project_id)
	
	def run_analysis(
		self,
		source_dir: Optional[str] = None,
		commit_sha: Optional[str] = None,
		use_cli: bool = False
	) -> CodacyReport:
		"""Run Codacy analysis"""
		start_time = time.time()
		
		if use_cli:
			return self._run_cli_analysis(source_dir or ".")
		else:
			return self._run_api_analysis(commit_sha)
	
	def _run_api_analysis(self, commit_sha: Optional[str] = None) -> CodacyReport:
		"""Run analysis using Codacy API"""
		if not self.api_client.check_connection():
			raise RuntimeError("Failed to connect to Codacy API. Check your API token.")
		
		if self.verbose:
			print("✓ Connected to Codacy API")
		
		# Trigger analysis if commit_sha provided
		if commit_sha:
			if self.verbose:
				print(f"Triggering analysis for commit: {commit_sha}")
			self.api_client.trigger_analysis(commit_sha)
			time.sleep(5)  # Wait for analysis to start
		
		# Get issues
		if self.verbose:
			print("Fetching issues...")
		
		files_data = self.api_client.get_commit_issues(commit_sha)
		
		# Parse issues
		parser = CodacyParser(files_data, source_type="api")
		report = parser.parse()
		
		report.execution_time = time.time() - start_time
		report.project_id = self.project_id
		report.commit_sha = commit_sha
		
		return report
	
	def _run_cli_analysis(self, source_dir: str) -> CodacyReport:
		"""Run analysis using Codacy CLI"""
		if not CodacyCLI.check_cli():
			raise RuntimeError("Codacy CLI is not installed. Install with: npm install -g codacy")
		
		if self.verbose:
			print("✓ Codacy CLI found")
		
		output, returncode = CodacyCLI.analyze_directory(
			source_dir,
			self.api_token,
			self.project_id,
			self.verbose
		)
		
		# Parse CLI output (simplified - would need actual parsing)
		parser = CodacyParser(output, source_type="cli")
		report = parser.parse()
		
		report.execution_time = time.time() - start_time
		
		return report


# ============================================================================
# CLI Interface
# ============================================================================

def main():
	"""Main entry point"""
	parser = argparse.ArgumentParser(
		description="Run Codacy analysis and generate detailed code quality reports",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog=__doc__
	)
	
	parser.add_argument(
		'--api-token',
		default=None,
		help='Codacy API token (can also be set via CODACY_API_TOKEN env var)'
	)
	
	parser.add_argument(
		'--project-id',
		help='Codacy project ID (optional if using CLI)'
	)
	
	parser.add_argument(
		'--source-dir',
		default='.',
		help='Source directory to analyze (default: current directory)'
	)
	
	parser.add_argument(
		'--commit-sha',
		help='Git commit SHA to analyze (for API mode)'
	)
	
	parser.add_argument(
		'--use-cli',
		action='store_true',
		help='Use Codacy CLI instead of API'
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
	
	# Get API token from args or environment
	api_token = args.api_token or os.environ.get('CODACY_API_TOKEN')
	
	if not api_token and not args.use_cli:
		print("Error: Codacy API token is required.")
		print("Please provide it via --api-token or set CODACY_API_TOKEN environment variable.")
		print("Alternatively, use --use-cli to use Codacy CLI (requires Codacy CLI installed).")
		sys.exit(1)
	
	# Create runner (use placeholder if CLI mode)
	runner = CodacyRunner(api_token or "placeholder", args.project_id, args.verbose)
	
	# Run analysis
	print("Starting Codacy analysis...")
	
	try:
		report = runner.run_analysis(
			source_dir=args.source_dir,
			commit_sha=args.commit_sha,
			use_cli=args.use_cli
		)
		
		print(f"\n{'='*80}")
		print("Codacy Report Summary")
		print(f"{'='*80}")
		print(f"  Total Issues:    {report.total_issues}")
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

