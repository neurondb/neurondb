#!/usr/bin/env python3

"""
run_test.py
Run SQL testcases in tests/sql by category and print concise results.

Output format per test:
  ✓ 2025-11-16 12:34:56  001_linreg.sql             2.34s
  ✗ 2025-11-16 12:35:01  002_logreg_negative.sql     0.18s

Categories:
  - basic:     files without `_advance` or `_negative`
  - advance:   files with `_advance.sql`
  - negative:  files with `_negative.sql`
"""

import argparse
import csv
import glob
import gzip
import os
import platform
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
from urllib.request import urlopen

try:
	import psycopg2
	from psycopg2.extras import execute_batch
except ImportError:
	psycopg2 = None
	execute_batch = None

try:
	import numpy as np
	HAS_NUMPY = True
except ImportError:
	HAS_NUMPY = False

# Import GPUDetector from tools/gpu.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
try:
	from gpu import GPUDetector
except ImportError:
	GPUDetector = None


TESTS_SQL_DIR = os.path.join(os.path.dirname(__file__), "sql")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
DEFAULT_ERROR_DIR = os.path.join(os.path.dirname(__file__), "error")
DEFAULT_DB = "neurondb"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 5432
DEFAULT_MODE = "cpu"  # Default: CPU (0=CPU, 1=GPU, 2=AUTO)
DEFAULT_NUM_ROWS = 1000

# Output formatting constants for perfect alignment
ICON_WIDTH = 2
TIMESTAMP_WIDTH = 19
TEST_NUM_WIDTH = 8
TEST_NAME_WIDTH = 45
ELAPSED_WIDTH = 12
LINE_WIDTH = 80  # Standard line width for separators
HEADER_SEPARATOR = "-" * LINE_WIDTH
LABEL_WIDTH = 20  # Width for labels in header info sections

# ANSI escape codes for formatting
BOLD = "\033[1m"
RESET = "\033[0m"
GREEN = "\033[32m"
RED = "\033[31m"
GREEN_BOLD = "\033[1;32m"
RED_BOLD = "\033[1;31m"
GREEN = "\033[32m"
RED = "\033[31m"
RED_BOLD = "\033[1;31m"
GREEN_BOLD = "\033[1;32m"

# Script version
SCRIPT_VERSION = "1.0.0"
SCRIPT_NAME = "test_runner.py"

# Global flag for graceful shutdown on Ctrl+C
_shutdown_requested = False


def signal_handler(signum, frame):
	"""Handle SIGINT (Ctrl+C) gracefully."""
	global _shutdown_requested
	_shutdown_requested = True


def list_sql_files(category: str) -> List[str]:
	"""
	List SQL files by category from tests/sql (recursively).
	"""
	if not os.path.isdir(TESTS_SQL_DIR):
		raise FileNotFoundError(f"SQL directory not found: {TESTS_SQL_DIR}")

	# Recursively find all SQL files
	all_files = []
	for root, dirs, files in os.walk(TESTS_SQL_DIR):
		for f in files:
			if f.endswith(".sql"):
				rel_path = os.path.relpath(os.path.join(root, f), TESTS_SQL_DIR)
				all_files.append(rel_path)
	all_files.sort()

	if category == "basic":
		# Include files from basic/ subdirectory or files without _advance/_negative
		result = []
		for f in all_files:
			if "/basic/" in f or (f.startswith("basic/") or os.path.dirname(f) == "basic"):
				result.append(os.path.join(TESTS_SQL_DIR, f))
			elif "/" not in f and "_advance.sql" not in f and "_negative.sql" not in f and "_perf.sql" not in f:
				result.append(os.path.join(TESTS_SQL_DIR, f))
		return result
	elif category == "advance":
		return [
			os.path.join(TESTS_SQL_DIR, f)
			for f in all_files
			if "_advance.sql" in f or "/advance/" in f
		]
	elif category == "negative":
		return [
			os.path.join(TESTS_SQL_DIR, f)
			for f in all_files
			if "_negative.sql" in f or "/negative/" in f
		]
	elif category == "all":
		return [os.path.join(TESTS_SQL_DIR, f) for f in all_files]
	else:
		raise ValueError("category must be one of: basic, advance, negative, all")


def verify_gpu_usage(dbname: str, psql_path: str, compute_mode: str, test_name: str = "", host: Optional[str] = None, port: Optional[int] = None) -> Tuple[bool, str]:
	"""
	Verify that GPU-trained models actually used GPU.
	Only checks the most recent model to avoid false positives from previous tests.
	Returns (success, error_message).
	"""
	if compute_mode == "cpu":
		return True, ""  # Skip verification for CPU mode
	
	env = os.environ.copy()
	if host:
		env["PGHOST"] = host
	if port:
		env["PGPORT"] = str(port)
	
	# Check only the most recent model to avoid false positives from previous test runs
	check_sql = """
		SELECT 
			m.model_id,
			m.algorithm,
			COALESCE(m.metrics::jsonb->>'storage', 'cpu') AS storage
		FROM neurondb.ml_models m
		ORDER BY m.model_id DESC
		LIMIT 1;
	"""
	
	cmd = [
		psql_path,
		"-d", dbname,
		"-t", "-A",
		"-w",
		"-c", check_sql
	]
	
	try:
		proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
		out, err = proc.communicate(timeout=5)
		
		if proc.returncode != 0:
			return True, ""  # If query fails, don't fail the test
		
		if out.strip():
			lines = [line.strip() for line in out.strip().split('\n') if line.strip()]
			for line in lines:
				parts = line.split('|')
				if len(parts) >= 3:
					model_id = parts[0].strip()
					algorithm = parts[1].strip()
					storage = parts[2].strip()
					# Only check ML algorithms that should use GPU
					ml_algorithms = ['linear_regression', 'logistic_regression', 'random_forest', 'svm', 'ridge', 'lasso', 'decision_tree', 'naive_bayes']
					if storage != 'gpu' and algorithm in ml_algorithms:
						return False, f"GPU mode enabled but model_id={model_id} (algorithm={algorithm}) was trained on CPU (storage={storage})"
		
		return True, ""
	except Exception:
		return True, ""  # Don't fail tests if verification fails


def run_psql_file(dbname: str, sql_file: str, psql_path: str, verbose: bool = False) -> Tuple[bool, float, str, str]:
	"""
	Run a single SQL file through psql.
	Uses trust authentication by default (no password prompt).
	Returns (success, elapsed_seconds, stdout, stderr).
	"""
	start = time.perf_counter()
	cmd = [
		psql_path,
		"-v", "ON_ERROR_STOP=1",
		"-d", dbname,
		"-f", sql_file,
		"-w",  # Never prompt for password (trust auth)
	]
	proc = subprocess.Popen(
		cmd,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
		env=os.environ.copy(),
	)
	out, err = proc.communicate()
	elapsed = time.perf_counter() - start
	success = proc.returncode == 0
	if verbose:
		# Echo captured output on verbose mode for visibility
		if out:
			sys.stdout.write(out)
			if not out.endswith("\n"):
				sys.stdout.write("\n")
		if err:
			sys.stderr.write(err)
			if not err.endswith("\n"):
				sys.stderr.write("\n")
	return success, elapsed, out or "", err or ""


def format_status_line(ok: bool, when: datetime, message: str, elapsed: float) -> str:
	"""
	Format a status line (pre-test checks) with perfect alignment.
	Format: [icon] [timestamp] [message]...[elapsed]s
	"""
	icon = f"{GREEN_BOLD}✓{RESET}" if ok else f"{RED_BOLD}✗{RESET}"
	ts = when.strftime("%Y-%m-%d %H:%M:%S")
	elapsed_str = f"{elapsed:>8.2f}s"
	
	# Calculate available width for message
	available_width = 80 - ICON_WIDTH - 1 - TIMESTAMP_WIDTH - 2 - ELAPSED_WIDTH
	message_width = min(len(message), available_width)
	message_padded = message[:message_width].ljust(available_width)
	
	return f"{icon:<{ICON_WIDTH + len(GREEN_BOLD) + len(RESET) - 1}} {ts:<{TIMESTAMP_WIDTH}}  {message_padded:<{available_width}} {elapsed_str:>{ELAPSED_WIDTH}}"


def format_test_line(ok: bool, when: datetime, test_num: int, total: int, name: str, elapsed: float, critical_crash: bool = False) -> str:
	"""
	Format a test result line with perfect alignment.
	Format: [icon] [timestamp] [test_num/total] [test_name]...[elapsed]s
	"""
	if critical_crash:
		icon = f"{RED_BOLD}!!{RESET}"  # Crash marker
	elif ok:
		icon = f"{GREEN_BOLD}✓{RESET}"
	else:
		icon = f"{RED_BOLD}✗{RESET}"
	ts = when.strftime("%Y-%m-%d %H:%M:%S")
	test_info = f"{test_num}/{total}"
	elapsed_str = f"{elapsed:>8.2f}s"
	
	# Calculate available width for test name
	available_width = TEST_NAME_WIDTH
	name_padded = name[:available_width].ljust(available_width)
	
	status_text = "CRITICAL CRASH" if critical_crash else ""
	if status_text:
		name_padded = f"{name_padded[:available_width - len(status_text) - 1]} {status_text}"
	
	return f"{icon:<{ICON_WIDTH + len(GREEN_BOLD) + len(RESET) - 1}} {ts:<{TIMESTAMP_WIDTH}}  {test_info:<{TEST_NUM_WIDTH}} {name_padded:<{TEST_NAME_WIDTH}} {elapsed_str:>{ELAPSED_WIDTH}}"


def find_psql() -> str:
	"""
	Resolve psql executable in PATH.
	"""
	psql = os.environ.get("PSQL", "psql")
	return psql


def find_pg_ctl() -> Optional[str]:
	"""
	Find pg_ctl executable in PATH or common PostgreSQL locations.
	"""
	# Try environment variable first
	pg_ctl = os.environ.get("PG_CTL")
	if pg_ctl and os.path.isfile(pg_ctl) and os.access(pg_ctl, os.X_OK):
		return pg_ctl
	
	# Try to find from psql path
	psql_path = find_psql()
	if psql_path != "psql" and os.path.isfile(psql_path):
		# psql is in a specific directory, try pg_ctl in same directory
		psql_dir = os.path.dirname(os.path.abspath(psql_path))
		pg_ctl_candidate = os.path.join(psql_dir, "pg_ctl")
		if os.path.isfile(pg_ctl_candidate) and os.access(pg_ctl_candidate, os.X_OK):
			return pg_ctl_candidate
	
	# Try common locations
	common_paths = [
		"/usr/lib/postgresql/*/bin/pg_ctl",
		"/usr/local/pgsql*/bin/pg_ctl",
		"/opt/homebrew/opt/postgresql@*/bin/pg_ctl",
		"/usr/pgsql-*/bin/pg_ctl",
	]
	
	for pattern in common_paths:
		matches = glob.glob(pattern)
		if matches:
			candidate = matches[0]
			if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
				return candidate
	
	# Try which command
	try:
		result = subprocess.run(
			["which", "pg_ctl"],
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True
		)
		if result.returncode == 0:
			candidate = result.stdout.strip()
			if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
				return candidate
	except Exception:
		pass
	
	return None


def find_pg_data_dir(host: Optional[str] = None, port: Optional[int] = None) -> Optional[str]:
	"""
	Try to find PostgreSQL data directory.
	Returns None if not found.
	"""
	# Try environment variable
	pgdata = os.environ.get("PGDATA")
	if pgdata and os.path.isdir(pgdata):
		return pgdata
	
	# Try common locations
	common_paths = [
		"/var/lib/postgresql/*/main",
		"/usr/local/pgsql*/data",
		"/opt/homebrew/var/postgresql@*",
		"/Users/*/neurondb_data*",
		os.path.expanduser("~/pgdata"),
		os.path.expanduser("~/postgres_data"),
	]
	
	for pattern in common_paths:
		matches = glob.glob(pattern)
		if matches:
			candidate = matches[0]
			if os.path.isdir(candidate):
				# Check if it looks like a PostgreSQL data directory
				if os.path.isfile(os.path.join(candidate, "postgresql.conf")):
					return candidate
	
	# Try to query PostgreSQL for data directory
	try:
		psql_path = find_psql()
		cmd = [
			psql_path,
			"-t", "-A",
			"-c", "SHOW data_directory;",
			"-w"
		]
		env = os.environ.copy()
		if host:
			env["PGHOST"] = host
		if port:
			env["PGPORT"] = str(port)
		
		proc = subprocess.Popen(
			cmd,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
			env=env
		)
		out, err = proc.communicate(timeout=5)
		if proc.returncode == 0 and out.strip():
			candidate = out.strip()
			if os.path.isdir(candidate):
				return candidate
	except Exception:
		pass
	
	return None


def restart_postgresql(
	dbname: str,
	psql_path: str,
	host: Optional[str] = None,
	port: Optional[int] = None,
	verbose: bool = False
) -> Tuple[bool, str]:
	"""
	Attempt to restart PostgreSQL using various methods.
	Returns (success, message).
	"""
	# Method 1: Try systemctl (if running as systemd service)
	try:
		result = subprocess.run(
			["systemctl", "restart", "postgresql"],
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
			timeout=30
		)
		if result.returncode == 0:
			# Wait a bit for PostgreSQL to start
			time.sleep(3)
			# Verify connection
			conn_ok, _, _ = check_postgresql_connection(dbname, psql_path, host, port)
			if conn_ok:
				return True, "Restarted via systemctl"
	except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
		pass
	
	# Method 2: Try pg_ctl restart
	pg_ctl = find_pg_ctl()
	pgdata = find_pg_data_dir(host, port)
	
	if pg_ctl and pgdata:
		try:
			# Try restart first
			result = subprocess.run(
				[pg_ctl, "restart", "-D", pgdata, "-w"],
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE,
				text=True,
				timeout=30
			)
			if result.returncode == 0:
				# Wait a bit
				time.sleep(2)
				conn_ok, _, _ = check_postgresql_connection(dbname, psql_path, host, port)
				if conn_ok:
					return True, f"Restarted via pg_ctl restart"
		except (subprocess.TimeoutExpired, Exception) as e:
			if verbose:
				print(f"pg_ctl restart failed: {e}", file=sys.stderr)
		
		# If restart failed, try stop then start
		try:
			# Stop
			subprocess.run(
				[pg_ctl, "stop", "-D", pgdata, "-m", "fast"],
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE,
				text=True,
				timeout=10
			)
			time.sleep(1)
			# Start
			result = subprocess.run(
				[pg_ctl, "start", "-D", pgdata, "-w", "-l", os.path.join(pgdata, "postgresql.log")],
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE,
				text=True,
				timeout=30
			)
			if result.returncode == 0:
				time.sleep(2)
				conn_ok, _, _ = check_postgresql_connection(dbname, psql_path, host, port)
				if conn_ok:
					return True, f"Restarted via pg_ctl stop/start"
		except (subprocess.TimeoutExpired, Exception) as e:
			if verbose:
				print(f"pg_ctl stop/start failed: {e}", file=sys.stderr)
	
	# Method 3: Try killing and restarting (last resort)
	if pg_ctl and pgdata:
		try:
			# Kill all postgres processes (careful!)
			subprocess.run(
				["pkill", "-9", "postgres"],
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE,
				timeout=5
			)
			time.sleep(2)
			# Start
			result = subprocess.run(
				[pg_ctl, "start", "-D", pgdata, "-w", "-l", os.path.join(pgdata, "postgresql.log")],
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE,
				text=True,
				timeout=30
			)
			if result.returncode == 0:
				time.sleep(2)
				conn_ok, _, _ = check_postgresql_connection(dbname, psql_path, host, port)
				if conn_ok:
					return True, f"Restarted via kill and start"
		except (subprocess.TimeoutExpired, Exception) as e:
			if verbose:
				print(f"Kill and restart failed: {e}", file=sys.stderr)
	
	return False, "Could not restart PostgreSQL automatically. Please restart manually."


def check_postgresql_crashed(
	dbname: str,
	psql_path: str,
	host: Optional[str] = None,
	port: Optional[int] = None
) -> bool:
	"""
	Check if PostgreSQL has crashed (connection fails).
	Returns True if PostgreSQL appears to be down.
	"""
	conn_ok, _, _ = check_postgresql_connection(dbname, psql_path, host, port)
	return not conn_ok


def check_postgresql_connection(dbname: str, psql_path: str, host: Optional[str] = None, port: Optional[int] = None) -> Tuple[bool, float, str]:
	"""
	Check PostgreSQL connection.
	Uses trust authentication by default (no password prompt).
	Returns (success, elapsed_seconds, connection_info).
	"""
	start = time.perf_counter()
	
	# Build command with trust authentication (no password prompt)
	cmd = [
		psql_path,
		"-d", dbname,
		"-c", "SELECT version();",
		"-w",  # Never prompt for password (trust auth)
	]
	
	# Use environment variables (which may include PGPASSWORD if set)
	env = os.environ.copy()
	if host:
		env["PGHOST"] = host
	if port:
		env["PGPORT"] = str(port)
	
	try:
		proc = subprocess.Popen(
			cmd,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
			env=env,
		)
		out, err = proc.communicate()
		elapsed = time.perf_counter() - start
		success = proc.returncode == 0
		
		host_display = host or os.environ.get("PGHOST", "localhost")
		port_display = port or os.environ.get("PGPORT", "5432")
		conn_info = f"postgresql://{host_display}:{port_display}"
		
		return success, elapsed, conn_info
	except Exception as e:
		elapsed = time.perf_counter() - start
		host_display = host or os.environ.get("PGHOST", "localhost")
		port_display = port or os.environ.get("PGPORT", "5432")
		return False, elapsed, f"postgresql://{host_display}:{port_display}"


def get_platform_info() -> Dict[str, str]:
	"""Gather platform information."""
	info = {}
	
	# CPU information
	try:
		if platform.system() == "Linux":
			with open('/proc/cpuinfo', 'r') as f:
				cpuinfo = f.read()
				for line in cpuinfo.split('\n'):
					if 'model name' in line:
						info['cpu'] = line.split(':')[1].strip()
						break
					elif 'processor' in line and 'cpu' not in info:
						info['cpu'] = platform.processor() or "Unknown"
		else:
			info['cpu'] = platform.processor() or "Unknown"
	except Exception:
		info['cpu'] = platform.processor() or "Unknown"
	
	# Memory information
	try:
		if platform.system() == "Linux":
			with open('/proc/meminfo', 'r') as f:
				meminfo = f.read()
				for line in meminfo.split('\n'):
					if 'MemTotal' in line:
						mem_kb = int(line.split()[1])
						mem_gb = mem_kb / (1024 * 1024)
						info['memory'] = f"{mem_gb:.2f} GB"
						break
		elif platform.system() == "Darwin":
			result = subprocess.run(['sysctl', '-n', 'hw.memsize'], 
								capture_output=True, text=True)
			if result.returncode == 0:
				mem_bytes = int(result.stdout.strip())
				mem_gb = mem_bytes / (1024 ** 3)
				info['memory'] = f"{mem_gb:.2f} GB"
			else:
				info['memory'] = "Unknown"
		else:
			info['memory'] = "Unknown"
	except Exception:
		info['memory'] = "Unknown"
	
	# Disk information (works on Linux, macOS, Rocky, Ubuntu)
	try:
		result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True, timeout=5)
		if result.returncode == 0:
			lines = result.stdout.strip().split('\n')
			if len(lines) > 1:
				parts = lines[1].split()
				if len(parts) >= 4:
					# Linux format: Filesystem Size Used Avail Use% Mounted
					# macOS format: Filesystem  Size Used Avail Capacity iused ifree %iused Mounted
					if platform.system() == "Darwin":
						# macOS: parts[8] is total, parts[7] is used
						if len(parts) >= 9:
							total = parts[8]
							used = parts[7]
							info['disk'] = f"{used} / {total}"
						else:
							total = parts[1]
							used = parts[2]
							info['disk'] = f"{used} / {total}"
					else:
						# Linux (including Rocky, Ubuntu)
						total = parts[1]
						used = parts[2]
						info['disk'] = f"{used} / {total}"
		else:
			info['disk'] = "Unknown"
	except (subprocess.TimeoutExpired, Exception):
		info['disk'] = "Unknown"
	
	return info


def get_os_info() -> Dict[str, str]:
	"""Gather OS information."""
	return {
		'system': platform.system(),
		'release': platform.release(),
		'version': platform.version(),
		'machine': platform.machine(),
		'architecture': platform.architecture()[0],
	}


def get_gpu_info(dbname: str, psql_path: str, host: Optional[str] = None, port: Optional[int] = None) -> Dict[str, str]:
	"""Gather GPU information using Python-based system detection."""
	info = {}
	
	# Use Python-based GPU detection
	if GPUDetector is None:
		info['available'] = "Unknown"
		info['device_name'] = "Unknown"
		info['device_id'] = "Unknown"
		return info
	
	try:
		# Detect all GPUs using Python system detection
		gpus = GPUDetector.detect_all()
		
		if gpus and len(gpus) > 0:
			gpu = gpus[0]  # Use first GPU
			info['available'] = "Yes"
			info['device_id'] = str(gpu.get('id', 0))
			info['device_name'] = gpu.get('name', 'Unknown')
			info['backend'] = gpu.get('backend', 'Unknown')
			info['type'] = gpu.get('type', 'Unknown')
			
			# Memory information
			if 'memory_mb' in gpu and gpu['memory_mb'] > 0:
				info['memory_total'] = f"{gpu['memory_mb']} MB"
			
			# Compute capability
			if 'compute_cap' in gpu and gpu['compute_cap']:
				info['compute_capability'] = gpu['compute_cap']
			
			# Platform information
			if 'platform' in gpu:
				info['platform'] = gpu['platform']
			
			# Chip information for Apple Silicon
			if 'chip' in gpu:
				info['chip'] = gpu['chip']
		else:
			info['available'] = "No"
			info['device_name'] = "None detected"
			info['device_id'] = "N/A"
	except Exception as e:
		info['available'] = "Unknown"
		info['device_name'] = f"Detection error: {str(e)}"
		info['device_id'] = "Unknown"
	
	# Try to get GPU settings from PostgreSQL GUC (optional, for enabled status)
	env = os.environ.copy()
	if host:
		env["PGHOST"] = host
	if port:
		env["PGPORT"] = str(port)
	
	try:
		cmd = [psql_path, "-d", dbname, "-t", "-A", "-w", "-c", 
		       "SELECT current_setting('neurondb.gpu_enabled', true), current_setting('neurondb.gpu_device', true), current_setting('neurondb.gpu_kernels', true);"]
		proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, timeout=5)
		out, err = proc.communicate()
		if proc.returncode == 0 and out.strip():
			parts = out.strip().split('|')
			if len(parts) >= 3:
				if parts[0].strip() and parts[0].strip() != '':
					info['enabled'] = parts[0].strip()
				if parts[1].strip() and parts[1].strip() != '':
					if not info.get('device_id') or info.get('device_id') == 'Unknown':
						info['device_id'] = parts[1].strip()
				if parts[2].strip() and parts[2].strip() != '':
					info['kernels'] = parts[2].strip()
	except (subprocess.TimeoutExpired, Exception):
		# If PostgreSQL query fails, that's okay - we have system detection
		pass
	
	return info


def get_postgresql_info(dbname: str, psql_path: str, host: Optional[str] = None, port: Optional[int] = None) -> Dict[str, str]:
	"""Gather PostgreSQL information."""
	info = {}
	
	# Get PostgreSQL version
	cmd = [psql_path, "-d", dbname, "-t", "-A", "-c", "SELECT version();"]
	env = os.environ.copy()
	if host:
		env["PGHOST"] = host
	if port:
		env["PGPORT"] = str(port)
	
	try:
		proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
		out, err = proc.communicate()
		if proc.returncode == 0 and out.strip():
			version_line = out.strip()
			# Extract PostgreSQL version
			if 'PostgreSQL' in version_line:
				info['version'] = version_line.split('PostgreSQL')[1].split()[0] if 'PostgreSQL' in version_line else version_line
			else:
				info['version'] = version_line
		else:
			info['version'] = "Unable to connect"
	except Exception:
		info['version'] = "Unable to connect"
	
	# Get server information
	try:
		cmd = [psql_path, "-d", dbname, "-t", "-A", "-c", 
		       "SELECT current_setting('server_version');"]
		proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
		out, err = proc.communicate()
		if proc.returncode == 0 and out.strip():
			info['server_version'] = out.strip()
	except Exception:
		pass
	
	# Get database information
	try:
		cmd = [psql_path, "-d", dbname, "-t", "-A", "-c",
		       f"SELECT current_database(), current_user, inet_server_addr(), inet_server_port();"]
		proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
		out, err = proc.communicate()
		if proc.returncode == 0 and out.strip():
			parts = out.strip().split('|')
			if len(parts) >= 4:
				info['database'] = parts[0]
				info['user'] = parts[1]
				info['host'] = parts[2] if parts[2] else (host or os.environ.get('PGHOST', 'localhost'))
				info['port'] = parts[3] if parts[3] else (str(port) if port else os.environ.get('PGPORT', '5432'))
	except Exception:
		info['host'] = host or os.environ.get('PGHOST', 'localhost')
		info['port'] = str(port) if port else os.environ.get('PGPORT', '5432')
	
	return info


def print_header_info(script_name: str, version: str, dbname: str, psql_path: str, 
		      host: Optional[str] = None, port: Optional[int] = None, mode: Optional[str] = None) -> None:
	"""Print detailed header information about the test runner and system."""
	print()
	print(f"NeuronDB Test Suite Version {version}")
	print("-" * LINE_WIDTH)
	print("Platform Information:")
	
	platform_info = get_platform_info()
	
	# Print CPU (Bold, aligned)
	cpu_label = f"{BOLD}CPU:{RESET}"
	cpu_value = platform_info.get('cpu', 'Unknown')
	print(f"\t{cpu_label:<{LABEL_WIDTH + len(BOLD) + len(RESET) - 2}} {cpu_value}")
	
	# Print Memory (Bold, aligned)
	memory_label = f"{BOLD}Memory:{RESET}"
	memory_value = platform_info.get('memory', 'Unknown')
	print(f"\t{memory_label:<{LABEL_WIDTH + len(BOLD) + len(RESET) - 2}} {memory_value}")
	
	# Print Disk (Bold, aligned)
	disk_label = f"{BOLD}Disk:{RESET}"
	disk_value = platform_info.get('disk', 'Unknown')
	print(f"\t{disk_label:<{LABEL_WIDTH + len(BOLD) + len(RESET) - 2}} {disk_value}")
	
	print("OS Information:")
	
	os_info = get_os_info()
	for key, value in sorted(os_info.items()):
		key_formatted = key.capitalize() + ":"
		key_bold = f"{BOLD}{key_formatted}{RESET}"
		print(f"\t{key_bold:<{LABEL_WIDTH + len(BOLD) + len(RESET) - 2}} {value}")
	
	print("PostgreSQL Information:")
	
	pg_info = get_postgresql_info(dbname, psql_path, host, port)
	
	# Print PostgreSQL info in order with perfect alignment
	pg_order = ['version', 'server_version', 'host', 'port', 'database', 'user']
	for key in pg_order:
		if key in pg_info:
			key_formatted = key.replace('_', ' ').title() + ":"
			key_bold = f"{BOLD}{key_formatted}{RESET}"
			print(f"\t{key_bold:<{LABEL_WIDTH + len(BOLD) + len(RESET) - 2}} {pg_info[key]}")
	
	# Print any remaining keys
	for key, value in sorted(pg_info.items()):
		if key not in pg_order:
			key_formatted = key.replace('_', ' ').title() + ":"
			key_bold = f"{BOLD}{key_formatted}{RESET}"
			print(f"\t{key_bold:<{LABEL_WIDTH + len(BOLD) + len(RESET) - 2}} {value}")
	
	# Print GPU Information if compute mode is GPU or AUTO
	if mode in ("gpu", "auto"):
		print("GPU Information:")
		
		gpu_info = get_gpu_info(dbname, psql_path, host, port)
		
		# Print GPU info in order with perfect alignment
		gpu_order = ['available', 'enabled', 'device_id', 'device_name', 'compute_capability', 'memory_total', 'memory_free', 'kernels']
		for key in gpu_order:
			if key in gpu_info:
				key_formatted = key.replace('_', ' ').title() + ":"
				key_bold = f"{BOLD}{key_formatted}{RESET}"
				print(f"\t{key_bold:<{LABEL_WIDTH + len(BOLD) + len(RESET) - 2}} {gpu_info[key]}")
		
		# Print any remaining keys
		for key, value in sorted(gpu_info.items()):
			if key not in gpu_order:
				key_formatted = key.replace('_', ' ').title() + ":"
				key_bold = f"{BOLD}{key_formatted}{RESET}"
				print(f"\t{key_bold:<{LABEL_WIDTH + len(BOLD) + len(RESET) - 2}} {value}")
	
	print("-" * LINE_WIDTH)
	print()
	print()


def create_test_views(dbname: str, psql_path: str, num_rows: int, host: Optional[str] = None, port: Optional[int] = None) -> Tuple[bool, int]:
	"""
	Create test views with specified number of rows.
	Returns (success, row_count).
	"""
	env = os.environ.copy()
	if host:
		env["PGHOST"] = host
	if port:
		env["PGPORT"] = str(port)
	
	# Check which source table exists (dataset.test_train/test_test, sample_train/sample_test, or test_train/test_test)
	try:
		# Check for dataset.test_train first (preferred), then sample_train, then test_train
		check_train_cmd = [
			psql_path, "-d", dbname, "-t", "-A", "-w",
			"-c", "SELECT table_schema || '.' || table_name FROM information_schema.tables WHERE (table_schema = 'dataset' AND table_name = 'test_train') OR (table_schema = 'public' AND table_name IN ('sample_train', 'test_train')) ORDER BY CASE WHEN table_schema = 'dataset' THEN 0 WHEN table_name = 'sample_train' THEN 1 ELSE 2 END LIMIT 1;"
		]
		
		train_proc = subprocess.Popen(check_train_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
		train_out, train_err = train_proc.communicate()
		
		if train_proc.returncode == 0 and train_out.strip():
			train_table_full = train_out.strip()
			
			# Determine corresponding test table
			if train_table_full.startswith('dataset.'):
				test_table_full = "dataset.test_test"
			elif 'sample_train' in train_table_full:
				test_table_full = "sample_test"
			else:
				test_table_full = "test_test"
			
			# Verify test table exists
			if '.' in test_table_full:
				schema, table = test_table_full.split('.')
				check_test_cmd = [
					psql_path, "-d", dbname, "-t", "-A", "-w",
					"-c", f"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = '{schema}' AND table_name = '{table}';"
				]
			else:
				check_test_cmd = [
					psql_path, "-d", dbname, "-t", "-A", "-w",
					"-c", f"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_name = '{test_table_full}';"
				]
			
			test_proc = subprocess.Popen(check_test_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
			test_out, test_err = test_proc.communicate()
			
			if test_proc.returncode == 0 and test_out.strip() and int(test_out.strip()) > 0:
				# Check if vector extension is available
				check_vector_cmd = [
					psql_path, "-d", dbname, "-t", "-A", "-w",
					"-c", "SELECT EXISTS(SELECT 1 FROM pg_type WHERE typname = 'vector');"
				]
				vector_check = subprocess.Popen(check_vector_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
				vector_out, vector_err = vector_check.communicate()
				has_vector = vector_check.returncode == 0 and vector_out.strip() == "t"
				
				# Create views with LIMIT num_rows
				# Try to convert REAL[] to vector if vector extension is available, otherwise use REAL[]
				# Use DO block to handle potential cast failures gracefully
				if has_vector:
					# Try vector cast, fallback to REAL[] if it fails
					create_sql = f"""
DROP VIEW IF EXISTS test_train_view CASCADE;
DROP VIEW IF EXISTS test_test_view CASCADE;

DO $$
BEGIN
	-- Try to create views with vector cast
	BEGIN
		EXECUTE format('CREATE VIEW test_train_view AS SELECT features::vector(28) as features, label FROM %s LIMIT %s', '{train_table_full}', {num_rows});
		EXECUTE format('CREATE VIEW test_test_view AS SELECT features::vector(28) as features, label FROM %s LIMIT %s', '{test_table_full}', {num_rows});
	EXCEPTION WHEN OTHERS THEN
		-- Fallback to REAL[] if vector cast fails
		EXECUTE format('CREATE VIEW test_train_view AS SELECT features as features, label FROM %s LIMIT %s', '{train_table_full}', {num_rows});
		EXECUTE format('CREATE VIEW test_test_view AS SELECT features as features, label FROM %s LIMIT %s', '{test_table_full}', {num_rows});
	END;
END $$;

-- Create or truncate test settings table for test runner configuration
CREATE TABLE IF NOT EXISTS test_settings (
	setting_key TEXT PRIMARY KEY,
	setting_value TEXT,
	updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create or truncate test metrics table for storing test results
CREATE TABLE IF NOT EXISTS test_metrics (
	test_name TEXT PRIMARY KEY,
	algorithm TEXT,
	model_id INTEGER,
	train_samples BIGINT,
	test_samples BIGINT,
	-- Regression metrics
	mse NUMERIC,
	rmse NUMERIC,
	mae NUMERIC,
	r_squared NUMERIC,
	-- Classification metrics
	accuracy NUMERIC,
	precision NUMERIC,
	recall NUMERIC,
	f1_score NUMERIC,
	-- Clustering metrics
	silhouette_score NUMERIC,
	inertia NUMERIC,
	n_clusters INTEGER,
	-- Time series metrics
	mape NUMERIC,
	-- Predictions (stored as JSONB for flexibility)
	predictions JSONB,
	-- Additional metadata
	metadata JSONB,
	created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

TRUNCATE TABLE test_metrics;
"""
				else:
					# No vector extension, use REAL[] directly
					create_sql = f"""
DROP VIEW IF EXISTS test_train_view CASCADE;
DROP VIEW IF EXISTS test_test_view CASCADE;

CREATE VIEW test_train_view AS
SELECT features as features, label 
FROM {train_table_full} 
LIMIT {num_rows};

CREATE VIEW test_test_view AS
SELECT features as features, label 
FROM {test_table_full} 
LIMIT {num_rows};

-- Create or truncate test settings table for test runner configuration
CREATE TABLE IF NOT EXISTS test_settings (
	setting_key TEXT PRIMARY KEY,
	setting_value TEXT,
	updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create or truncate test metrics table for storing test results
CREATE TABLE IF NOT EXISTS test_metrics (
	test_name TEXT PRIMARY KEY,
	algorithm TEXT,
	model_id INTEGER,
	train_samples BIGINT,
	test_samples BIGINT,
	-- Regression metrics
	mse NUMERIC,
	rmse NUMERIC,
	mae NUMERIC,
	r_squared NUMERIC,
	-- Classification metrics
	accuracy NUMERIC,
	precision NUMERIC,
	recall NUMERIC,
	f1_score NUMERIC,
	-- Clustering metrics
	silhouette_score NUMERIC,
	inertia NUMERIC,
	n_clusters INTEGER,
	-- Time series metrics
	mape NUMERIC,
	-- Predictions (stored as JSONB for flexibility)
	predictions JSONB,
	-- Additional metadata
	metadata JSONB,
	created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
	updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

TRUNCATE TABLE test_metrics;
"""
				
				create_proc = subprocess.Popen(
					[psql_path, "-d", dbname, "-w", "-c", create_sql],
					stdout=subprocess.PIPE,
					stderr=subprocess.PIPE,
					text=True,
					env=env
				)
				create_out, create_err = create_proc.communicate()
				
				if create_proc.returncode == 0:
					# Verify views were created and get row count
					count_cmd = [
						psql_path, "-d", dbname, "-t", "-A", "-w",
						"-c", "SELECT COALESCE((SELECT COUNT(*) FROM test_train_view), 0);"
					]
					
					count_proc = subprocess.Popen(count_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
					count_out, count_err = count_proc.communicate()
					
					if count_proc.returncode == 0 and count_out.strip():
						row_count = int(count_out.strip())
						if row_count > 0:
							return True, row_count
					# If row count is 0, something went wrong
					if count_err:
						print(f"Warning: View created but has 0 rows. Error: {count_err}", file=sys.stderr)
				else:
					# View creation failed - log error for debugging
					if create_err:
						# Only show error if it's not about vector type (we handle that gracefully)
						if "type \"vector\" does not exist" not in create_err.lower():
							print(f"Warning: View creation failed: {create_err}", file=sys.stderr)
		
		return False, 0
	except (subprocess.TimeoutExpired, Exception):
		return False, 0


def run_psql_command(dbname: str, sql_command: str, psql_path: str, verbose: bool = False) -> Tuple[bool, str, str]:
	"""
	Run a SQL command through psql.
	Uses trust authentication by default (no password prompt).
	Returns (success, stdout, stderr).
	"""
	cmd = [
		psql_path,
		"-v", "ON_ERROR_STOP=1",
		"-d", dbname,
		"-c", sql_command,
		"-w",  # Never prompt for password (trust auth)
	]
	proc = subprocess.Popen(
		cmd,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
		env=os.environ.copy(),
	)
	out, err = proc.communicate()
	success = proc.returncode == 0
	if verbose:
		if out:
			sys.stdout.write(out)
		if err:
			sys.stderr.write(err)
	return success, out or "", err or ""


def switch_gpu_mode(dbname: str, compute_mode: str, psql_path: str, gpu_kernels: str = None, verbose: bool = False) -> bool:
	"""
	Switch compute mode using ALTER SYSTEM and reload configuration.
	Compute mode should be 'gpu', 'cpu', or 'auto'.
	Returns True on success, False on failure.
	"""
	if compute_mode not in ("gpu", "cpu", "auto"):
		print(f"Invalid compute mode: {compute_mode}. Must be 'gpu', 'cpu', or 'auto'.", file=sys.stderr)
		return False

	# Map compute mode to enum value: cpu=0, gpu=1, auto=2
	mode_enum = {"cpu": 0, "gpu": 1, "auto": 2}.get(compute_mode, 2)

	# Set compute mode using new GUC
	cmd1 = f"ALTER SYSTEM SET neurondb.compute_mode = {mode_enum};"
	success1, out1, err1 = run_psql_command(dbname, cmd1, psql_path, verbose)
	if not success1:
		print(f"Failed to set compute mode: {err1}", file=sys.stderr)
		return False

	# Set GPU kernels - include all ML algorithm kernels for GPU or auto mode
	if compute_mode in ("gpu", "auto"):
		# Default kernels plus all ML training/prediction kernels
		default_kernels = "l2,cosine,ip,rf_split,rf_predict"
		ml_kernels = "linreg_train,linreg_predict,lr_train,lr_predict,rf_train,svm_train,svm_predict,ridge_train,ridge_predict,lasso_train,lasso_predict,dt_train,dt_predict,nb_train,nb_predict"
		full_kernels = f"{default_kernels},{ml_kernels}"
		cmd_kernels = f"ALTER SYSTEM SET neurondb.gpu_kernels = '{full_kernels}';"
		success_k, out_k, err_k = run_psql_command(dbname, cmd_kernels, psql_path, verbose)
		if not success_k:
			print(f"Warning: Failed to set GPU kernels: {err_k}", file=sys.stderr)
	elif gpu_kernels:
		# Use provided kernels if specified
		cmd_kernels = f"ALTER SYSTEM SET neurondb.gpu_kernels = '{gpu_kernels}';"
		success_k, out_k, err_k = run_psql_command(dbname, cmd_kernels, psql_path, verbose)
		if not success_k:
			print(f"Warning: Failed to set GPU kernels: {err_k}", file=sys.stderr)

	# Reload configuration
	cmd2 = "SELECT pg_reload_conf();"
	success2, out2, err2 = run_psql_command(dbname, cmd2, psql_path, verbose)
	if not success2:
		print(f"Failed to reload configuration: {err2}", file=sys.stderr)
		return False

	# Initialize GPU if GPU or auto mode is enabled
	if compute_mode in ("gpu", "auto"):
		cmd3 = "SELECT neurondb_gpu_enable();"
		success3, out3, err3 = run_psql_command(dbname, cmd3, psql_path, verbose)
		if not success3:
			if compute_mode == "gpu":
				print(f"Warning: Failed to enable GPU: {err3}", file=sys.stderr)
				# In GPU mode, this is a warning but we continue (let the mode handle errors)
			else:
				print(f"Warning: Failed to enable GPU (auto mode will fallback to CPU): {err3}", file=sys.stderr)
		
		# Force GPU initialization by querying GPU info
		cmd4 = "SELECT * FROM neurondb_gpu_info() LIMIT 1;"
		success4, out4, err4 = run_psql_command(dbname, cmd4, psql_path, verbose)
		if not success4:
			if verbose:
				if compute_mode == "gpu":
					print(f"Warning: GPU info query failed (GPU may not be available): {err4}", file=sys.stderr)
				else:
					print(f"Info: GPU info query failed (auto mode will fallback to CPU): {err4}", file=sys.stderr)

	if verbose:
		print(f"Switched to {compute_mode.upper()} mode successfully.")
	return True


# HIGGS dataset constants
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/280/higgs.zip"
HIGGS_CSV_BASENAME = "HIGGS.csv"
EXPECTED_NUM_COLUMNS = 29
MB = 1024 * 1024


def find_local_higgs_csv() -> Optional[str]:
	"""Try to locate an existing HIGGS.csv in common locations."""
	candidates = [
		os.path.join(os.getcwd(), HIGGS_CSV_BASENAME),  # Current directory first
		os.path.join(os.path.dirname(__file__), "datasets", HIGGS_CSV_BASENAME),
		os.path.join(os.path.dirname(__file__), HIGGS_CSV_BASENAME),
		os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", HIGGS_CSV_BASENAME),
		HIGGS_CSV_BASENAME,
	]
	for path in candidates:
		if os.path.isfile(path):
			return path
	return None


def find_local_higgs_zip() -> Optional[str]:
	"""Try to locate a local higgs.zip - checks current directory first."""
	candidates = [
		os.path.join(os.getcwd(), "higgs.zip"),  # Current directory first
		os.path.join(os.path.dirname(__file__), "datasets", "higgs.zip"),
		os.path.join(os.path.dirname(__file__), "higgs.zip"),
		os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "higgs.zip"),
		"higgs.zip",
	]
	for path in candidates:
		if os.path.isfile(path):
			return path
	return None


def download_higgs_zip(dest_path: str) -> None:
	"""Download HIGGS dataset zip file."""
	print(f"Downloading HIGGS dataset from {UCI_ZIP_URL}...")
	print("This may take several minutes (file is ~2.8GB)...")
	
	with urlopen(UCI_ZIP_URL) as resp, open(dest_path, "wb") as out:
		chunk_size = 1 * MB
		total = 0
		total_size = 0
		try:
			hdr_len = resp.getheader("Content-Length")
			total_size = int(hdr_len) if hdr_len else 0
		except Exception:
			total_size = 0
		
		start_ts = last_ts = time.time()
		last_reported_percent = -1
		while True:
			chunk = resp.read(chunk_size)
			if not chunk:
				break
			out.write(chunk)
			total += len(chunk)
			now = time.time()
			
			if total_size > 0:
				percent = int((total * 100) / total_size)
			else:
				percent = -1
			
			if percent != last_reported_percent or (now - last_ts) >= 0.5:
				elapsed = max(0.001, now - start_ts)
				speed = total / elapsed
				mb_done = total / MB
				if total_size > 0:
					mb_total = total_size / MB
					sys.stderr.write(
						f"\r[DOWNLOAD] {percent:3d}% "
						f"({mb_done:.1f}/{mb_total:.1f} MB) "
						f"@ {speed/MB:.2f} MB/s"
					)
				else:
					sys.stderr.write(
						f"\r[DOWNLOAD] {mb_done:.1f} MB "
						f"@ {speed/MB:.2f} MB/s"
					)
				sys.stderr.flush()
				last_ts = now
				last_reported_percent = percent
		sys.stderr.write("\n")
		sys.stderr.flush()
	print(f"Download complete: {dest_path}")


def extract_higgs_csv(zip_path: str, output_dir: str) -> str:
	"""Extract HIGGS.csv from zip file."""
	print(f"Extracting HIGGS.csv from {zip_path}...")
	with zipfile.ZipFile(zip_path, "r") as zf:
		names = zf.namelist()
		csv_name = None
		for n in names:
			lower = n.lower()
			if lower.endswith(".csv") and "higgs" in lower:
				csv_name = n
				break
		
		if csv_name is None:
			raise FileNotFoundError("No HIGGS CSV file found in zip.")
		
		os.makedirs(output_dir, exist_ok=True)
		target_path = os.path.join(output_dir, HIGGS_CSV_BASENAME)
		with zf.open(csv_name, "r") as src, open(target_path, "wb") as dst:
			while True:
				chunk = src.read(1 * MB)
				if not chunk:
					break
				dst.write(chunk)
		
		print(f"Extracted to: {target_path}")
		return target_path


def get_higgs_csv_path(csv_path: Optional[str] = None) -> str:
	"""
	Get path to HIGGS.csv from local sources only.
	Checks current directory first, then extracts from zip if found.
	Does NOT download - quits if not found.
	"""
	if csv_path and os.path.isfile(csv_path):
		return csv_path
	
	# Try to find local CSV
	local_csv = find_local_higgs_csv()
	if local_csv:
		return local_csv
	
	# Try to find local ZIP (current directory first)
	local_zip = find_local_higgs_zip()
	if local_zip:
		try:
			# Extract to current directory or datasets subdirectory
			output_dir = os.getcwd()
			datasets_dir = os.path.join(os.path.dirname(__file__), "datasets")
			# Prefer extracting to datasets if it exists, otherwise current dir
			if os.path.isdir(datasets_dir):
				output_dir = datasets_dir
			return extract_higgs_csv(local_zip, output_dir)
		except (zipfile.BadZipFile, zipfile.LargeZipFile) as e:
			# Invalid zip file
			raise FileNotFoundError(
				f"\n{'='*80}\n"
				f"ERROR: Invalid or corrupted zip file: {local_zip}\n"
				f"{'='*80}\n"
				f"\n"
				f"Please download the HIGGS dataset from:\n"
				f"  {UCI_ZIP_URL}\n"
				f"\n"
				f"Then place 'higgs.zip' in this directory:\n"
				f"  {os.getcwd()}\n"
				f"\n"
				f"The file should be approximately 2.8GB in size.\n"
				f"{'='*80}\n"
			) from e
		except Exception as e:
			# Other extraction errors
			raise FileNotFoundError(
				f"\n{'='*80}\n"
				f"ERROR: Failed to extract HIGGS.csv from zip file: {e}\n"
				f"{'='*80}\n"
				f"\n"
				f"Please ensure 'higgs.zip' is a valid zip file.\n"
				f"Download from: {UCI_ZIP_URL}\n"
				f"Place it in: {os.getcwd()}\n"
				f"{'='*80}\n"
			) from e
	
	# Not found - raise error with clear instructions
	current_dir = os.getcwd()
	raise FileNotFoundError(
		f"\n{'='*80}\n"
		f"ERROR: HIGGS dataset not found\n"
		f"{'='*80}\n"
		f"\n"
		f"Please download the HIGGS dataset from:\n"
		f"  {UCI_ZIP_URL}\n"
		f"\n"
		f"Then place 'higgs.zip' in this directory:\n"
		f"  {current_dir}\n"
		f"\n"
		f"The file should be approximately 2.8GB in size.\n"
		f"\n"
		f"Alternatively, you can use --dataset=synthetic to generate synthetic data instead.\n"
		f"{'='*80}\n"
	)


def generate_synthetic_higgs_data(
	num_rows: int,
	seed: Optional[int] = None
) -> Tuple[List[Tuple], List[Tuple]]:
	"""
	Generate synthetic HIGGS-style dataset.
	Returns (train_data, test_data) where each is a list of (features, label) tuples.
	
	HIGGS format:
	- 29 columns: label (0 or 1) + 28 features
	- Features are real-valued
	- 21 low-level features + 7 high-level features
	
	The synthetic data mimics HIGGS characteristics:
	- Binary classification (signal=1, background=0)
	- 28 real-valued features
	- Some correlation between features and labels for realistic ML testing
	"""
	if seed is not None:
		random.seed(seed)
		if HAS_NUMPY:
			np.random.seed(seed)
	
	train_data = []
	test_data = []
	
	# Generate data with some correlation between features and labels
	# to make it somewhat realistic for ML testing
	for i in range(num_rows):
		# Generate label (0 or 1) - balanced dataset
		label = 1 if random.random() < 0.5 else 0
		
		# Generate 28 features
		# First 21: low-level features (more variation, independent)
		# Last 7: high-level features (derived-like, more correlated)
		features = []
		
		# Low-level features (0-20): more independent, wider range
		# These simulate raw measurements (lepton pT, jet properties, etc.)
		for j in range(21):
			if label == 1:
				# Signal: slightly higher values on average
				base = random.gauss(0.5, 1.5)
			else:
				# Background: slightly lower values
				base = random.gauss(0.0, 1.5)
			# Clamp to reasonable range
			features.append(max(-10.0, min(10.0, base)))
		
		# High-level features (21-27): more correlated, derived-like
		# These simulate derived quantities (m_jj, m_jjj, m_lv, etc.)
		# Make them functions of low-level features to simulate derived features
		for j in range(7):
			# Use some combination of previous features
			idx1 = random.randint(0, 20)
			idx2 = random.randint(0, 20)
			# Derived feature as combination of two low-level features
			derived = (features[idx1] + features[idx2]) / 2.0 + random.gauss(0, 0.5)
			if label == 1:
				derived += 0.3  # Slight bias for signal
			# Clamp to reasonable range
			features.append(max(-10.0, min(10.0, derived)))
		
		# Split 80/20 train/test (consistent with HIGGS loading)
		if i % 10 < 8:
			train_data.append((features, label))
		else:
			test_data.append((features, label))
	
	return train_data, test_data


def load_synthetic_dataset(
	dbname: str,
	num_rows: int = 100000,
	seed: Optional[int] = None,
	host: Optional[str] = None,
	port: Optional[int] = None
) -> bool:
	"""
	Generate and load synthetic HIGGS-style dataset into dataset.test_train and dataset.test_test tables.
	Returns True on success, False on failure.
	"""
	if psycopg2 is None:
		print("ERROR: psycopg2 is required for dataset loading. Install with: pip install psycopg2-binary", file=sys.stderr)
		return False
	
	# Get database connection
	env = os.environ.copy()
	if host:
		env["PGHOST"] = host
	if port:
		env["PGPORT"] = str(port)
	
	user = env.get("PGUSER") or env.get("USER")
	password = env.get("PGPASSWORD")
	
	try:
		conn = psycopg2.connect(
			dbname=dbname,
			user=user,
			password=password,
			host=host or env.get("PGHOST", "localhost"),
			port=port or int(env.get("PGPORT", "5432"))
		)
		cur = conn.cursor()
	except Exception as e:
		print(f"ERROR: Failed to connect to database: {e}", file=sys.stderr)
		return False
	
	try:
		# Create dataset schema
		print("Creating dataset schema...")
		cur.execute("CREATE SCHEMA IF NOT EXISTS dataset")
		conn.commit()
		
		# Create tables
		print("Creating tables in dataset schema...")
		cur.execute("""
			DROP TABLE IF EXISTS dataset.test_train CASCADE;
			DROP TABLE IF EXISTS dataset.test_test CASCADE;
			
			CREATE TABLE dataset.test_train (
				features REAL[],
				label integer
			);
			
			CREATE TABLE dataset.test_test (
				features REAL[],
				label integer
			);
		""")
		conn.commit()
		
		# Generate synthetic data
		print(f"Generating {num_rows:,} synthetic HIGGS-style rows...")
		start_time = time.time()
		train_data, test_data = generate_synthetic_higgs_data(num_rows, seed)
		gen_elapsed = time.time() - start_time
		
		print(f"Generated {len(train_data):,} train rows and {len(test_data):,} test rows in {gen_elapsed:.1f}s")
		
		# Insert train data
		print("\nInserting into dataset.test_train...")
		start_time = time.time()
		batch_size = 1000
		
		for i in range(0, len(train_data), batch_size):
			batch = train_data[i:i+batch_size]
			values = [(features, label) for features, label in batch]
			
			execute_batch(
				cur,
				"INSERT INTO dataset.test_train (features, label) VALUES (%s::REAL[], %s)",
				values
			)
			conn.commit()
			
			if (i // batch_size) % 10 == 0:
				elapsed = time.time() - start_time
				print(f"  Inserted {i + len(batch):,} / {len(train_data):,} train rows ({elapsed:.1f}s)")
		
		train_elapsed = time.time() - start_time
		print(f"✓ Loaded {len(train_data):,} rows into dataset.test_train in {train_elapsed:.1f}s")
		
		# Insert test data
		print("\nInserting into dataset.test_test...")
		start_time = time.time()
		
		for i in range(0, len(test_data), batch_size):
			batch = test_data[i:i+batch_size]
			values = [(features, label) for features, label in batch]
			
			execute_batch(
				cur,
				"INSERT INTO dataset.test_test (features, label) VALUES (%s::REAL[], %s)",
				values
			)
			conn.commit()
			
			if (i // batch_size) % 10 == 0:
				elapsed = time.time() - start_time
				print(f"  Inserted {i + len(batch):,} / {len(test_data):,} test rows ({elapsed:.1f}s)")
		
		test_elapsed = time.time() - start_time
		print(f"✓ Loaded {len(test_data):,} rows into dataset.test_test in {test_elapsed:.1f}s")
		
		# Verify
		cur.execute("SELECT COUNT(*) FROM dataset.test_train")
		train_count = cur.fetchone()[0]
		cur.execute("SELECT COUNT(*) FROM dataset.test_test")
		test_count = cur.fetchone()[0]
		
		print(f"\nFinal counts:")
		print(f"  dataset.test_train: {train_count:,} rows")
		print(f"  dataset.test_test: {test_count:,} rows")
		
		cur.close()
		conn.close()
		return True
		
	except Exception as e:
		print(f"ERROR: Failed to load synthetic dataset: {e}", file=sys.stderr)
		import traceback
		traceback.print_exc()
		if cur:
			cur.close()
		if conn:
			conn.close()
		return False


def load_higgs_dataset(
	dbname: str,
	csv_path: Optional[str] = None,
	limit: Optional[int] = None,
	host: Optional[str] = None,
	port: Optional[int] = None,
	train_split: float = 0.8
) -> bool:
	"""
	Load HIGGS dataset into dataset.test_train and dataset.test_test tables.
	Returns True on success, False on failure.
	"""
	if psycopg2 is None:
		print("ERROR: psycopg2 is required for dataset loading. Install with: pip install psycopg2-binary", file=sys.stderr)
		return False
	
	# Get CSV path
	try:
		csv_file = get_higgs_csv_path(csv_path)
	except FileNotFoundError as e:
		# Print the detailed error message
		print(str(e), file=sys.stderr)
		return False
	except Exception as e:
		print(f"\n{'='*80}\n", file=sys.stderr)
		print(f"ERROR: Failed to get HIGGS CSV: {e}\n", file=sys.stderr)
		print(f"Please download higgs.zip from: {UCI_ZIP_URL}\n", file=sys.stderr)
		print(f"Place it in: {os.getcwd()}\n", file=sys.stderr)
		print(f"{'='*80}\n", file=sys.stderr)
		return False
	
	# Get database connection
	env = os.environ.copy()
	if host:
		env["PGHOST"] = host
	if port:
		env["PGPORT"] = str(port)
	
	user = env.get("PGUSER") or env.get("USER")
	password = env.get("PGPASSWORD")
	
	try:
		conn = psycopg2.connect(
			dbname=dbname,
			user=user,
			password=password,
			host=host or env.get("PGHOST", "localhost"),
			port=port or int(env.get("PGPORT", "5432"))
		)
		cur = conn.cursor()
	except Exception as e:
		print(f"ERROR: Failed to connect to database: {e}", file=sys.stderr)
		return False
	
	try:
		# Create dataset schema
		print("Creating dataset schema...")
		cur.execute("CREATE SCHEMA IF NOT EXISTS dataset")
		conn.commit()
		
		# Create tables
		print("Creating tables in dataset schema...")
		cur.execute("""
			DROP TABLE IF EXISTS dataset.test_train CASCADE;
			DROP TABLE IF EXISTS dataset.test_test CASCADE;
			
			CREATE TABLE dataset.test_train (
				features REAL[],
				label integer
			);
			
			CREATE TABLE dataset.test_test (
				features REAL[],
				label integer
			);
		""")
		conn.commit()
		
		# Read and split data
		train_data = []
		test_data = []
		
		print(f"Reading {csv_file}...")
		start_time = time.time()
		row_count = 0
		
		# Open file (handle both .gz and regular files)
		if csv_file.endswith('.gz'):
			f = gzip.open(csv_file, 'rt')
		else:
			f = open(csv_file, 'r')
		
		try:
			reader = csv.reader(f)
			for row in reader:
				if limit and row_count >= limit:
					break
				
				if len(row) != EXPECTED_NUM_COLUMNS:
					continue
				
				label = int(float(row[0].strip()))
				features = [float(x.strip()) for x in row[1:29]]
				
				# Split into train/test (80/20)
				if row_count % 10 < int(train_split * 10):
					train_data.append((features, label))
				else:
					test_data.append((features, label))
				
				row_count += 1
				
				if row_count % 100000 == 0:
					elapsed = time.time() - start_time
					print(f"  Processed {row_count:,} rows ({elapsed:.1f}s)")
		finally:
			f.close()
		
		print(f"\nTotal rows processed: {row_count:,}")
		print(f"Train rows: {len(train_data):,}")
		print(f"Test rows: {len(test_data):,}")
		
		# Insert train data
		print("\nInserting into dataset.test_train...")
		start_time = time.time()
		batch_size = 1000
		
		for i in range(0, len(train_data), batch_size):
			batch = train_data[i:i+batch_size]
			values = [(features, label) for features, label in batch]
			
			execute_batch(
				cur,
				"INSERT INTO dataset.test_train (features, label) VALUES (%s::REAL[], %s)",
				values
			)
			conn.commit()
			
			if (i // batch_size) % 10 == 0:
				elapsed = time.time() - start_time
				print(f"  Inserted {i + len(batch):,} / {len(train_data):,} train rows ({elapsed:.1f}s)")
		
		train_elapsed = time.time() - start_time
		print(f"✓ Loaded {len(train_data):,} rows into dataset.test_train in {train_elapsed:.1f}s")
		
		# Insert test data
		print("\nInserting into dataset.test_test...")
		start_time = time.time()
		
		for i in range(0, len(test_data), batch_size):
			batch = test_data[i:i+batch_size]
			values = [(features, label) for features, label in batch]
			
			execute_batch(
				cur,
				"INSERT INTO dataset.test_test (features, label) VALUES (%s::REAL[], %s)",
				values
			)
			conn.commit()
			
			if (i // batch_size) % 10 == 0:
				elapsed = time.time() - start_time
				print(f"  Inserted {i + len(batch):,} / {len(test_data):,} test rows ({elapsed:.1f}s)")
		
		test_elapsed = time.time() - start_time
		print(f"✓ Loaded {len(test_data):,} rows into dataset.test_test in {test_elapsed:.1f}s")
		
		# Verify
		cur.execute("SELECT COUNT(*) FROM dataset.test_train")
		train_count = cur.fetchone()[0]
		cur.execute("SELECT COUNT(*) FROM dataset.test_test")
		test_count = cur.fetchone()[0]
		
		print(f"\nFinal counts:")
		print(f"  dataset.test_train: {train_count:,} rows")
		print(f"  dataset.test_test: {test_count:,} rows")
		
		cur.close()
		conn.close()
		return True
		
	except Exception as e:
		print(f"ERROR: Failed to load dataset: {e}", file=sys.stderr)
		if cur:
			cur.close()
		if conn:
			conn.close()
		return False


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run SQL testcases by category with clean output."
	)
	parser.add_argument(
		"--category",
		choices=["basic", "advance", "negative", "all"],
		default="basic",
		help=f"Which test category to run (default: basic).",
	)
	parser.add_argument(
		"--db",
		default=DEFAULT_DB,
		help=f"Database name (default: {DEFAULT_DB}).",
	)
	parser.add_argument(
		"--host",
		default=DEFAULT_HOST,
		help=f"PostgreSQL host (default: {DEFAULT_HOST}).",
	)
	parser.add_argument(
		"--port",
		type=int,
		default=DEFAULT_PORT,
		help=f"PostgreSQL port (default: {DEFAULT_PORT}).",
	)
	parser.add_argument(
		"--user",
		default=None,
		help="PostgreSQL user (default: system user).",
	)
	parser.add_argument(
		"--password",
		default=None,
		help="PostgreSQL password (if required). Will set PGPASSWORD environment variable.",
	)
	parser.add_argument(
		"--psql",
		default=find_psql(),
		help="Path to psql executable (default: resolve from PATH).",
	)
	parser.add_argument(
		"--compute",
		choices=["gpu", "cpu", "auto"],
		default=DEFAULT_MODE,
		help=f"Compute mode: gpu (GPU only), cpu (CPU only), or auto (try GPU, fallback to CPU) (default: {DEFAULT_MODE}).",
	)
	parser.add_argument(
		"--gpu-kernels",
		default=None,
		help="GPU kernels to enable (e.g., 'l2,cosine,ip,linreg_train,linreg_predict'). Only used with --mode gpu.",
	)
	parser.add_argument(
		"--num-rows",
		type=int,
		default=DEFAULT_NUM_ROWS,
		help=f"Number of rows for test views (default: {DEFAULT_NUM_ROWS}).",
	)
	parser.add_argument(
		"-v", "--verbose",
		action="store_true",
		help="Verbose mode: print psql stdout/stderr for each test.",
	)
	parser.add_argument(
		"--dataset",
		choices=["higgs", "synthetic"],
		default=None,
		help="Dataset to load before running tests. Options: higgs (downloads real HIGGS dataset), synthetic (generates HIGGS-style synthetic data)",
	)
	parser.add_argument(
		"--dataset-path",
		default=None,
		help="Path to dataset CSV file. If not provided, will download or search for HIGGS.csv",
	)
	parser.add_argument(
		"--dataset-limit",
		type=int,
		default=None,
		help="Limit number of rows to load from dataset (for testing with smaller datasets). For synthetic dataset, this is the total number of rows to generate.",
	)
	parser.add_argument(
		"--dataset-seed",
		type=int,
		default=None,
		help="Random seed for synthetic dataset generation (for reproducibility).",
	)
	parser.add_argument(
		"--test",
		default="all",
		help="Test name to run (default: all). If specified, only runs tests matching this name. Use 'all' to run all tests in the category.",
	)
	parser.add_argument(
		"--no-gcov",
		"--no-coverage",
		dest="no_coverage",
		action="store_true",
		help="Disable code coverage analysis using gcov (coverage is enabled by default).",
	)
	parser.add_argument(
		"--coverage",
		action="store_true",
		help="[DEPRECATED] Coverage is now enabled by default. Use --no-gcov to disable.",
	)
	parser.add_argument(
		"--coverage-dir",
		default=None,
		help="Directory to store coverage reports (default: tests/coverage).",
	)
	parser.add_argument(
		"--coverage-html",
		action="store_true",
		help="Generate HTML coverage report (requires gcovr or lcov). Enabled by default if gcovr is available.",
	)
	parser.add_argument(
		"--coverage-xml",
		action="store_true",
		help="Generate XML coverage report (requires gcovr).",
	)
	return parser.parse_args()


def find_gcov() -> Optional[str]:
	"""Find gcov executable in PATH."""
	for gcov_name in ["gcov", "gcov-12", "gcov-11", "gcov-10", "gcov-9"]:
		try:
			result = subprocess.run(
				["which", gcov_name],
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE,
				text=True
			)
			if result.returncode == 0:
				return result.stdout.strip()
		except Exception:
			continue
	return None


def find_gcovr() -> Optional[str]:
	"""Find gcovr executable in PATH."""
	try:
		result = subprocess.run(
			["which", "gcovr"],
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True
		)
		if result.returncode == 0:
			return result.stdout.strip()
	except Exception:
		pass
	return None


def find_lcov() -> Optional[str]:
	"""Find lcov executable in PATH."""
	try:
		result = subprocess.run(
			["which", "lcov"],
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True
		)
		if result.returncode == 0:
			return result.stdout.strip()
	except Exception:
		pass
	return None


def check_coverage_enabled(project_root: str) -> Tuple[bool, str]:
	"""
	Check if code was compiled with coverage flags.
	Returns (enabled, message).
	"""
	# Check for .gcda files (coverage data files)
	gcda_files = []
	for root, dirs, files in os.walk(project_root):
		# Skip hidden directories and build artifacts
		dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', 'venv']]
		for f in files:
			if f.endswith('.gcda'):
				gcda_files.append(os.path.join(root, f))
	
	if gcda_files:
		return True, f"Found {len(gcda_files)} coverage data files"
	
	# Check if .o files have coverage info by looking for .gcno files
	gcno_files = []
	for root, dirs, files in os.walk(project_root):
		dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', 'venv']]
		for f in files:
			if f.endswith('.gcno'):
				gcno_files.append(os.path.join(root, f))
	
	if gcno_files:
		return True, f"Found {len(gcno_files)} coverage note files (code compiled with coverage)"
	
	return False, "No coverage data found. Code must be compiled with --coverage or -fprofile-arcs -ftest-coverage flags."


def collect_gcov_data(project_root: str, coverage_dir: str, gcov_path: Optional[str], verbose: bool = False) -> Tuple[bool, List[str]]:
	"""
	Run gcov on all source files and collect coverage data.
	Returns (success, list of .gcov files generated).
	"""
	if not gcov_path:
		return False, []
	
	gcov_files = []
	
	# Find all .gcda files (coverage data files)
	gcda_files = []
	for root, dirs, files in os.walk(project_root):
		dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', 'venv', 'coverage']]
		for f in files:
			if f.endswith('.gcda'):
				gcda_path = os.path.join(root, f)
				gcda_files.append((gcda_path, root))
	
	if not gcda_files:
		return False, []
	
	# Run gcov on each .gcda file
	os.makedirs(coverage_dir, exist_ok=True)
	
	for gcda_path, gcda_dir in gcda_files:
		try:
			# Run gcov in the directory where .gcda file is located
			# -b: branch probabilities
			# -c: branch counts
			# -o: object directory
			gcda_basename = os.path.basename(gcda_path)
			cmd = [gcov_path, "-b", "-c", "-o", gcda_dir, gcda_basename]
			result = subprocess.run(
				cmd,
				cwd=gcda_dir,
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE,
				text=True,
				timeout=30
			)
			
			if result.returncode != 0 and verbose:
				print(f"Warning: gcov returned {result.returncode} for {gcda_path}", file=sys.stderr)
				if result.stderr:
					print(f"  stderr: {result.stderr[:200]}", file=sys.stderr)
			
			# Find generated .gcov files in the gcda directory
			if os.path.isdir(gcda_dir):
				for f in os.listdir(gcda_dir):
					if f.endswith('.gcov'):
						gcov_file = os.path.join(gcda_dir, f)
						# Move to coverage directory, handling name conflicts
						dest = os.path.join(coverage_dir, f)
						if os.path.exists(dest):
							# Add prefix to avoid conflicts
							base, ext = os.path.splitext(f)
							dest = os.path.join(coverage_dir, f"{os.path.basename(gcda_dir)}_{base}{ext}")
						try:
							shutil.move(gcov_file, dest)
							if dest not in gcov_files:
								gcov_files.append(dest)
						except Exception as e:
							if verbose:
								print(f"Warning: Failed to move {gcov_file} to {dest}: {e}", file=sys.stderr)
		except subprocess.TimeoutExpired:
			if verbose:
				print(f"Warning: gcov timed out for {gcda_path}", file=sys.stderr)
			continue
		except Exception as e:
			if verbose:
				print(f"Warning: Failed to run gcov on {gcda_path}: {e}", file=sys.stderr)
			continue
	
	return len(gcov_files) > 0, gcov_files


def generate_gcovr_report(
	project_root: str,
	coverage_dir: str,
	output_file: str,
	format: str = "txt",
	gcovr_path: Optional[str] = None
) -> Tuple[bool, str]:
	"""
	Generate coverage report using gcovr.
	format: 'txt', 'html', 'xml'
	Returns (success, message).
	"""
	if not gcovr_path:
		return False, "gcovr not found. Install with: pip install gcovr"
	
	os.makedirs(coverage_dir, exist_ok=True)
	
	# Build gcovr command - focus on NeuronDB source code only
	cmd = [gcovr_path, "-r", project_root]
	
	# Include only NeuronDB source files
	cmd.extend([
		"--filter", "src/",
		"--exclude", ".*/tests/.*",
		"--exclude", ".*/test/.*",
		"--exclude", ".*/tools/.*",
		"--exclude", ".*/dataset/.*",
		"--exclude", ".*/demo/.*",
		"--exclude", ".*/build/.*",
		"--exclude", ".*/node_modules/.*",
		"--exclude", ".*/venv/.*",
		"--exclude", r".*\/\.git\/.*",
	])
	
	if format == "html":
		cmd.extend(["--html", "--html-details", "-o", output_file])
	elif format == "xml":
		cmd.extend(["--xml", "-o", output_file])
	else:  # txt
		cmd.extend(["-o", output_file])
	
	# Add options for better reports
	cmd.extend([
		"--exclude-unreachable-branches",
		"--exclude-throw-branches",
		"--print-summary",
		"--sort-percentage",
		"--sort-uncovered",  # Show uncovered files first
		"--sort-reverse",   # Sort by coverage (lowest first)
	])
	
	try:
		result = subprocess.run(
			cmd,
			cwd=project_root,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
			timeout=300
		)
		
		if result.returncode == 0:
			return True, result.stdout
		else:
			return False, f"gcovr failed: {result.stderr}"
	except subprocess.TimeoutExpired:
		return False, "gcovr timed out"
	except Exception as e:
		return False, f"gcovr error: {str(e)}"


def generate_lcov_report(
	project_root: str,
	coverage_dir: str,
	output_file: str,
	lcov_path: Optional[str] = None
) -> Tuple[bool, str]:
	"""
	Generate coverage report using lcov.
	Returns (success, message).
	"""
	if not lcov_path:
		return False, "lcov not found. Install with: apt-get install lcov or brew install lcov"
	
	os.makedirs(coverage_dir, exist_ok=True)
	
	# Capture coverage data
	capture_cmd = [lcov_path, "--capture", "--directory", project_root, "--output-file", output_file]
	
	try:
		result = subprocess.run(
			capture_cmd,
			cwd=project_root,
			stdout=subprocess.PIPE,
			stderr=subprocess.PIPE,
			text=True,
			timeout=300
		)
		
		if result.returncode == 0:
			# Generate HTML report
			html_dir = os.path.join(coverage_dir, "html")
			genhtml_cmd = ["genhtml", output_file, "-o", html_dir]
			genhtml_result = subprocess.run(
				genhtml_cmd,
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE,
				text=True,
				timeout=300
			)
			
			if genhtml_result.returncode == 0:
				return True, f"HTML report generated in {html_dir}"
			else:
				return True, f"Coverage data captured, but HTML generation failed: {genhtml_result.stderr}"
		else:
			return False, f"lcov capture failed: {result.stderr}"
	except subprocess.TimeoutExpired:
		return False, "lcov timed out"
	except Exception as e:
		return False, f"lcov error: {str(e)}"


def parse_gcov_summary(gcov_output: str) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
	"""
	Parse gcovr summary output to extract coverage percentages and file details.
	Returns (summary_dict, file_list) where:
	- summary_dict: {'lines': float, 'branches': float, 'functions': float}
	- file_list: [{'file': str, 'lines_pct': float, 'branches_pct': float, 'functions_pct': float}, ...]
	"""
	summary = {"lines": 0.0, "branches": 0.0, "functions": 0.0}
	files = []
	
	# Parse gcovr output - look for summary lines
	for line in gcov_output.split('\n'):
		line_lower = line.lower()
		if 'lines:' in line_lower or 'line coverage:' in line_lower:
			# Extract percentage
			match = re.search(r'(\d+\.\d+)%', line)
			if match:
				summary["lines"] = float(match.group(1))
		elif 'branches:' in line_lower or 'branch coverage:' in line_lower:
			match = re.search(r'(\d+\.\d+)%', line)
			if match:
				summary["branches"] = float(match.group(1))
		elif 'functions:' in line_lower or 'function coverage:' in line_lower:
			match = re.search(r'(\d+\.\d+)%', line)
			if match:
				summary["functions"] = float(match.group(1))
	
	# If no percentages found, try to parse from summary table
	if summary["lines"] == 0.0 and summary["branches"] == 0.0 and summary["functions"] == 0.0:
		# Look for lines like: "TOTAL    1234    567    89.12%    234    123    52.56%    345    234    67.89%"
		for line in gcov_output.split('\n'):
			if 'TOTAL' in line.upper() and '%' in line:
				# Extract all percentages
				matches = re.findall(r'(\d+\.\d+)%', line)
				if len(matches) >= 3:
					summary["lines"] = float(matches[0])
					summary["branches"] = float(matches[1])
					summary["functions"] = float(matches[2])
					break
	
	# Parse file-by-file coverage
	for line in gcov_output.split('\n'):
		line = line.strip()
		if not line or line.startswith('-') or 'TOTAL' in line.upper() or 'lines:' in line.lower():
			continue
		
		# Look for file coverage lines (format: "src/path/file.c    1234    567    89.12%    234    123    52.56%    345    234    67.89%")
		# Match lines that have a file path and percentages
		if '/' in line and '%' in line and ('src/' in line or line.startswith('src/')):
			parts = line.split()
			if len(parts) >= 4:
				# Try to find file path (first part that contains '/')
				file_path = None
				percentages = []
				
				for i, part in enumerate(parts):
					if '/' in part and not part.endswith('%'):
						file_path = part
						# Look for percentages after the file path
						for j in range(i + 1, min(i + 10, len(parts))):
							if '%' in parts[j]:
								match = re.search(r'(\d+\.\d+)%', parts[j])
								if match:
									percentages.append(float(match.group(1)))
				
				if file_path and len(percentages) >= 3:
					files.append({
						'file': file_path,
						'lines_pct': percentages[0] if len(percentages) > 0 else 0.0,
						'branches_pct': percentages[1] if len(percentages) > 1 else 0.0,
						'functions_pct': percentages[2] if len(percentages) > 2 else 0.0,
					})
	
	return summary, files


def parse_gcov_file(gcov_file: str) -> Dict[str, Any]:
	"""
	Parse a .gcov file to extract coverage details.
	Returns dict with file info and uncovered lines.
	"""
	result = {
		'file': '',
		'total_lines': 0,
		'executed_lines': 0,
		'uncovered_lines': [],
		'functions': {},
	}
	
	try:
		with open(gcov_file, 'r') as f:
			current_function = None
			line_num = 0
			
			for line in f:
				line = line.rstrip()
				# GCOV format: "        -:   45:void function_name()"
				#              "    12345:   46:  code here"
				#              "    #####:   47:  uncovered code"
				
				if line.startswith('function ') or ':' in line:
					parts = line.split(':', 2)
					if len(parts) >= 2:
						count_str = parts[0].strip()
						line_info = parts[1].strip() if len(parts) > 1 else ''
						
						# Extract line number
						if line_info.isdigit():
							line_num = int(line_info)
							
							# Check if line is uncovered
							if count_str == '#####' or count_str == '-':
								result['uncovered_lines'].append(line_num)
							elif count_str.isdigit() or (count_str.startswith('-') and count_str[1:].isdigit()):
								if count_str != '0' and count_str != '-':
									result['executed_lines'] += 1
							
							result['total_lines'] += 1
						
						# Check for function definitions
						if len(parts) >= 3 and 'function' in parts[2].lower():
							func_name = parts[2].strip()
							if func_name.startswith('function '):
								func_name = func_name[9:].strip()
							current_function = func_name
							if current_function not in result['functions']:
								result['functions'][current_function] = {'covered': False, 'line': line_num}
		
		# Extract source file name from gcov filename
		base_name = os.path.basename(gcov_file)
		if base_name.endswith('.gcov'):
			result['file'] = base_name[:-5]  # Remove .gcov extension
		
	except Exception as e:
		pass
	
	return result


def generate_uncovered_details_report(coverage_dir: str, project_root: Optional[str] = None) -> Optional[str]:
	"""
	Generate a detailed report showing uncovered lines and functions.
	Returns path to report file if successful.
	"""
	gcov_files = [f for f in os.listdir(coverage_dir) if f.endswith('.gcov')]
	if not gcov_files:
		return None
	
	report_file = os.path.join(coverage_dir, "uncovered_details.txt")
	
	try:
		with open(report_file, 'w') as f:
			f.write("NeuronDB Detailed Coverage Report - Uncovered Areas\n")
			f.write("=" * 80 + "\n\n")
			f.write("This report shows specific lines and functions that are NOT covered by tests.\n")
			f.write("Each .gcov file contains detailed line-by-line execution information.\n\n")
			f.write("=" * 80 + "\n\n")
			
			# Process each .gcov file
			neurondb_files = []
			for gcov_file in sorted(gcov_files):
				gcov_path = os.path.join(coverage_dir, gcov_file)
				details = parse_gcov_file(gcov_path)
				
				# Only include NeuronDB source files
				if details['file'] and ('src/' in details['file'] or details['file'].startswith('src/')):
					neurondb_files.append((gcov_file, details))
			
			# Sort by number of uncovered lines (most uncovered first)
			neurondb_files.sort(key=lambda x: len(x[1]['uncovered_lines']), reverse=True)
			
			for gcov_file, details in neurondb_files:
				if details['uncovered_lines'] or details['total_lines'] == 0:
					f.write(f"\nFile: {details['file']}\n")
					f.write("-" * 80 + "\n")
					
					coverage_pct = 0.0
					if details['total_lines'] > 0:
						coverage_pct = (details['executed_lines'] / details['total_lines']) * 100.0
					
					f.write(f"Coverage: {coverage_pct:.2f}% ({details['executed_lines']}/{details['total_lines']} lines)\n")
					f.write(f"Uncovered lines: {len(details['uncovered_lines'])}\n\n")
					
					if details['uncovered_lines']:
						f.write("Uncovered line numbers:\n")
						# Show in ranges for compactness
						uncovered = sorted(details['uncovered_lines'])
						ranges = []
						start = uncovered[0]
						end = uncovered[0]
						
						for line in uncovered[1:]:
							if line == end + 1:
								end = line
							else:
								if start == end:
									ranges.append(str(start))
								else:
									ranges.append(f"{start}-{end}")
								start = line
								end = line
						
						if start == end:
							ranges.append(str(start))
						else:
							ranges.append(f"{start}-{end}")
						
						# Write ranges, 10 per line
						for i in range(0, len(ranges), 10):
							f.write("  " + ", ".join(ranges[i:i+10]) + "\n")
					
					f.write("\n")
			
			f.write("\n" + "=" * 80 + "\n")
			f.write("Note: Open individual .gcov files for line-by-line execution counts.\n")
			f.write("      Lines marked with '#####' were never executed.\n")
			f.write("      Lines with numbers show execution count.\n")
		
		return report_file
	except Exception as e:
		return None


def print_coverage_report(
	coverage_dir: str,
	summary: Dict[str, float],
	html_report: Optional[str] = None,
	xml_report: Optional[str] = None,
	files: Optional[List[Dict[str, Any]]] = None,
	project_root: Optional[str] = None
) -> None:
	"""Print a beautifully formatted coverage report with detailed breakdown."""
	print()
	print("=" * LINE_WIDTH)
	print(f"{BOLD}NeuronDB Code Coverage Report{RESET}")
	print("=" * LINE_WIDTH)
	print()
	
	# Coverage summary
	print(f"{BOLD}NeuronDB Coverage Summary:{RESET}")
	print()
	
	lines_pct = summary.get("lines", 0.0)
	branches_pct = summary.get("branches", 0.0)
	functions_pct = summary.get("functions", 0.0)
	
	# Color code based on coverage percentage
	def coverage_color(pct: float) -> str:
		if pct >= 80:
			return GREEN_BOLD
		elif pct >= 60:
			return GREEN
		elif pct >= 40:
			return ""
		else:
			return RED_BOLD
	
	lines_color = coverage_color(lines_pct)
	branches_color = coverage_color(branches_pct)
	functions_color = coverage_color(functions_pct)
	
	print(f"  {BOLD}Lines:{RESET}       {lines_color}{lines_pct:>6.2f}%{RESET}")
	print(f"  {BOLD}Branches:{RESET}    {branches_color}{branches_pct:>6.2f}%{RESET}")
	print(f"  {BOLD}Functions:{RESET}   {functions_color}{functions_pct:>6.2f}%{RESET}")
	print()
	
	# Overall coverage
	overall = (lines_pct + branches_pct + functions_pct) / 3.0
	overall_color = coverage_color(overall)
	print(f"  {BOLD}Overall:{RESET}     {overall_color}{overall:>6.2f}%{RESET}")
	print()
	
	# File-by-file breakdown (show uncovered/low coverage files)
	if files:
		# Sort by coverage (lowest first)
		files_sorted = sorted(files, key=lambda x: x.get('lines_pct', 0.0))
		
		# Show files with low coverage (< 80%)
		low_coverage = [f for f in files_sorted if f.get('lines_pct', 0.0) < 80.0]
		
		if low_coverage:
			print(f"{BOLD}Files Needing More Coverage (< 80%):{RESET}")
			print()
			print(f"  {'File':<50} {'Lines':<10} {'Branches':<10} {'Functions':<10}")
			print(f"  {'-' * 50} {'-' * 10} {'-' * 10} {'-' * 10}")
			
			for f in low_coverage[:20]:  # Show top 20
				file_name = f['file']
				if len(file_name) > 47:
					file_name = "..." + file_name[-44:]
				
				lines_p = f['lines_pct']
				branches_p = f['branches_pct']
				functions_p = f['functions_pct']
				
				lines_c = coverage_color(lines_p)
				branches_c = coverage_color(branches_p)
				functions_c = coverage_color(functions_p)
				
				print(f"  {file_name:<50} {lines_c}{lines_p:>6.2f}%{RESET}  {branches_c}{branches_p:>6.2f}%{RESET}  {functions_c}{functions_p:>6.2f}%{RESET}")
			
			if len(low_coverage) > 20:
				print(f"  ... and {len(low_coverage) - 20} more files with low coverage")
			print()
		
		# Show completely uncovered files (0% coverage)
		uncovered = [f for f in files_sorted if f.get('lines_pct', 0.0) == 0.0]
		if uncovered:
			print(f"{BOLD}Completely Uncovered Files (0%):{RESET}")
			print()
			for f in uncovered[:15]:  # Show top 15
				file_name = f['file']
				if len(file_name) > 70:
					file_name = "..." + file_name[-67:]
				print(f"  {RED_BOLD}✗{RESET} {file_name}")
			if len(uncovered) > 15:
				print(f"  ... and {len(uncovered) - 15} more uncovered files")
			print()
	
	# Report locations
	print(f"{BOLD}Detailed Coverage Reports:{RESET}")
	print()
	print(f"  Coverage Directory: {coverage_dir}")
	if html_report and os.path.exists(html_report):
		print(f"  {GREEN_BOLD}✓{RESET} HTML Report:     {html_report}")
		print(f"     Open in browser for line-by-line coverage details")
	if xml_report and os.path.exists(xml_report):
		print(f"  {GREEN_BOLD}✓{RESET} XML Report:      {xml_report}")
	
	# Find text report
	txt_report = os.path.join(coverage_dir, "coverage.txt")
	if os.path.exists(txt_report):
		print(f"  {GREEN_BOLD}✓{RESET} Text Report:      {txt_report}")
		print(f"     Contains detailed file-by-file coverage breakdown")
	
	# Find .gcov files for detailed analysis
	gcov_files = [f for f in os.listdir(coverage_dir) if f.endswith('.gcov')]
	if gcov_files:
		print(f"  {GREEN_BOLD}✓{RESET} GCOV Files:      {len(gcov_files)} detailed .gcov files")
		print(f"     Each .gcov file shows line-by-line execution counts")
		
		# Generate detailed uncovered report
		if project_root:
			uncovered_report = generate_uncovered_details_report(coverage_dir, project_root)
			if uncovered_report:
				print(f"  {GREEN_BOLD}✓{RESET} Uncovered Report: {uncovered_report}")
				print(f"     Lists all uncovered lines and functions in detail")
	
	print()
	print("=" * LINE_WIDTH)
	print()
	print(f"{BOLD}Note:{RESET} Coverage is calculated only for NeuronDB source code (src/ directory).")
	print(f"      System libraries and external dependencies are excluded.")
	print(f"      Check .gcov files for detailed line-by-line execution information.")
	print()


def ensure_dir(path: str) -> None:
	"""Ensure directory exists, creating it if necessary."""
	if not os.path.isdir(path):
		try:
			os.makedirs(path, exist_ok=True)
		except Exception as e:
			print(f"Warning: Failed to create directory {path}: {e}", file=sys.stderr)


def write_artifacts(name: str, ok: bool, out_dir: str, err_dir: str,
		    stdout_text: str, stderr_text: str) -> None:
	"""
	Write per-test artifacts.
	- Always write stdout to output directory
	- Always write stderr to error directory (if present)
	- On failure, also write both stdout/stderr to error directory
	"""
	try:
		# Ensure both directories exist
		ensure_dir(out_dir)
		ensure_dir(err_dir)
		
		base = os.path.splitext(os.path.basename(name))[0]
		
		# Always write stdout to output directory
		out_path = os.path.join(out_dir, f"{base}.out")
		try:
			with open(out_path, "w", encoding="utf8") as f:
				f.write(stdout_text or "")
		except Exception as e:
			print(f"Warning: Failed to write stdout to {out_path}: {e}", file=sys.stderr)
		
		# Write stderr to error directory if present
		if stderr_text and stderr_text.strip():
			err_path = os.path.join(err_dir, f"{base}.err")
			try:
				with open(err_path, "w", encoding="utf8") as f:
					f.write(stderr_text)
			except Exception as e:
				print(f"Warning: Failed to write stderr to {err_path}: {e}", file=sys.stderr)
		
		# On failure, also write both stdout and stderr to error directory
		if not ok:
			err_out_path = os.path.join(err_dir, f"{base}.out")
			err_err_path = os.path.join(err_dir, f"{base}.err")
			try:
				with open(err_out_path, "w", encoding="utf8") as f:
					f.write(stdout_text or "")
			except Exception as e:
				print(f"Warning: Failed to write stdout to error directory {err_out_path}: {e}", file=sys.stderr)
			try:
				with open(err_err_path, "w", encoding="utf8") as f:
					f.write(stderr_text or "")
			except Exception as e:
				print(f"Warning: Failed to write stderr to error directory {err_err_path}: {e}", file=sys.stderr)
	except Exception as e:
		print(f"Warning: Failed to write artifacts for {name}: {e}", file=sys.stderr)


def main() -> int:
	global _shutdown_requested
	
	# Register signal handler for graceful shutdown on Ctrl+C
	signal.signal(signal.SIGINT, signal_handler)
	_shutdown_requested = False
	
	args = parse_args()
	# Extend parser with output/error dirs without breaking existing users
	# Backward-compatible defaults
	if not hasattr(args, "output_dir"):
		setattr(args, "output_dir", DEFAULT_OUTPUT_DIR)
	if not hasattr(args, "error_dir"):
		setattr(args, "error_dir", DEFAULT_ERROR_DIR)

	# Set password if provided
	if args.password:
		os.environ["PGPASSWORD"] = args.password
	
	# Set host, port, and user in environment if provided
	if args.host:
		os.environ["PGHOST"] = args.host
	if args.port:
		os.environ["PGPORT"] = str(args.port)
	if args.user:
		os.environ["PGUSER"] = args.user
	
	# Print header information
	print_header_info(SCRIPT_NAME, SCRIPT_VERSION, args.db, args.psql, args.host, args.port, args.compute)
	
	# Load dataset if requested
	if args.dataset == "higgs":
		when = datetime.now()
		load_start = time.perf_counter()
		load_ok = load_higgs_dataset(
			args.db,
			csv_path=args.dataset_path,
			limit=args.dataset_limit,
			host=args.host,
			port=args.port
		)
		load_elapsed = time.perf_counter() - load_start
		print(format_status_line(load_ok, when, "Loading HIGGS dataset...", load_elapsed))
		if not load_ok:
			print("Failed to load HIGGS dataset. Aborting.", file=sys.stderr)
			return 1
	elif args.dataset == "synthetic":
		when = datetime.now()
		load_start = time.perf_counter()
		num_rows = args.dataset_limit or 100000  # Default 100k rows for synthetic
		load_ok = load_synthetic_dataset(
			args.db,
			num_rows=num_rows,
			seed=args.dataset_seed,
			host=args.host,
			port=args.port
		)
		load_elapsed = time.perf_counter() - load_start
		print(format_status_line(load_ok, when, f"Generating synthetic dataset ({num_rows:,} rows)...", load_elapsed))
		if not load_ok:
			print("Failed to generate synthetic dataset. Aborting.", file=sys.stderr)
			return 1
	
	# Pre-test checks with status lines
	# 1. Check PostgreSQL connection
	when = datetime.now()
	conn_ok, conn_elapsed, conn_info = check_postgresql_connection(args.db, args.psql, args.host, args.port)
	print(format_status_line(conn_ok, when, f"Checking postgresql on {conn_info}...", conn_elapsed))
	if not conn_ok:
		print(f"Failed to connect to PostgreSQL at {conn_info}. Aborting.", file=sys.stderr)
		return 1
	
	# 2. Switch compute mode (using ALTER SYSTEM)
	when = datetime.now()
	mode_start = time.perf_counter()
	mode_ok = switch_gpu_mode(args.db, args.compute, args.psql, args.gpu_kernels, args.verbose)
	mode_elapsed = time.perf_counter() - mode_start
	print(format_status_line(mode_ok, when, f"Configuring postgresql for {args.compute.upper()} compute mode...", mode_elapsed))
	if not mode_ok:
		print(f"Failed to switch to {args.compute.upper()} compute mode. Aborting.", file=sys.stderr)
		return 1
	
	# 3. Create test views (this also creates test_settings table)
	when = datetime.now()
	views_start = time.perf_counter()
	views_ok, row_count = create_test_views(args.db, args.psql, args.num_rows, args.host, args.port)
	views_elapsed = time.perf_counter() - views_start
	print(format_status_line(views_ok, when, f"Creating dataset for tests (rows={row_count})...", views_elapsed))
	if not views_ok:
		print(f"Warning: Failed to create test views. Some tests may fail.", file=sys.stderr)
		# Continue anyway - some tests might not need the views
	
	# 3.5. Store all test settings in test_settings table for tests to read
	when = datetime.now()
	settings_start = time.perf_counter()
	settings_sql = f"""
	-- Store compute mode setting
	INSERT INTO test_settings (setting_key, setting_value, updated_at)
	VALUES ('compute_mode', '{args.compute}', CURRENT_TIMESTAMP)
	ON CONFLICT (setting_key) DO UPDATE SET
		setting_value = EXCLUDED.setting_value,
		updated_at = CURRENT_TIMESTAMP;
	
	-- Store number of rows used for test views
	INSERT INTO test_settings (setting_key, setting_value, updated_at)
	VALUES ('num_rows', '{args.num_rows}', CURRENT_TIMESTAMP)
	ON CONFLICT (setting_key) DO UPDATE SET
		setting_value = EXCLUDED.setting_value,
		updated_at = CURRENT_TIMESTAMP;
	"""
	if args.gpu_kernels:
		settings_sql += f"""
	-- Store GPU kernels if provided
	INSERT INTO test_settings (setting_key, setting_value, updated_at)
	VALUES ('gpu_kernels', '{args.gpu_kernels}', CURRENT_TIMESTAMP)
	ON CONFLICT (setting_key) DO UPDATE SET
		setting_value = EXCLUDED.setting_value,
		updated_at = CURRENT_TIMESTAMP;
	"""
	settings_ok, settings_out, settings_err = run_psql_command(args.db, settings_sql, args.psql, args.verbose)
	settings_elapsed = time.perf_counter() - settings_start
	if not settings_ok:
		print(f"Warning: Failed to set test settings: {settings_err}", file=sys.stderr)
	
	sql_files = list_sql_files(args.category)
	if not sql_files:
		print(f"No SQL files found for category '{args.category}' in {TESTS_SQL_DIR}")
		return 2

	# Filter by test name if specified (and not "all")
	if args.test and args.test != "all":
		test_name = args.test.lower()
		# Remove .sql extension if provided for matching
		if test_name.endswith(".sql"):
			test_name = test_name[:-4]
		
		filtered_files = []
		for sql_file in sql_files:
			basename = os.path.basename(sql_file).lower()
			basename_no_ext = basename[:-4] if basename.endswith(".sql") else basename
			# Match if test name is in the filename (case-insensitive)
			if test_name in basename_no_ext:
				filtered_files.append(sql_file)
		
		if not filtered_files:
			print(f"No test files found matching '{args.test}' in category '{args.category}'")
			return 2
		
		sql_files = filtered_files

	total = len(sql_files)
	passed = 0
	failed = 0
	critical_crashes = 0
	t0 = time.perf_counter()

	# Print separator before tests
	print()
	print(HEADER_SEPARATOR)
	print()

	for idx, path in enumerate(sql_files, 1):
		# Check for shutdown request
		if _shutdown_requested:
			print("\n\nShutdown requested. Stopping test execution...")
			break
		
		when = datetime.now()
		name = os.path.basename(path)
		critical_crash = False
		
		# Show which test is starting (will be overwritten)
		temp_line = format_test_line(True, when, idx, total, name, 0.0)
		print(temp_line, end="\r", flush=True)
		
		# Run the test (continue on failure)
		try:
			ok, elapsed, out_text, err_text = run_psql_file(args.db, path, args.psql, verbose=args.verbose)
		except Exception as e:
			# If test execution throws an exception, mark as failed but continue
			ok = False
			elapsed = 0.0
			out_text = ""
			err_text = str(e)
		
		# Check again after test execution
		if _shutdown_requested:
			print("\n\nShutdown requested. Stopping test execution...")
			break
		
		# Check if PostgreSQL crashed (either during or after this test)
		# This catches crashes that happen during test execution or immediately after
		if check_postgresql_crashed(args.db, args.psql, args.host, args.port):
			critical_crash = True
			critical_crashes += 1
			print(f"\n    {RED_BOLD}!! PostgreSQL CRASHED during/after this test!{RESET}")
			print(f"    {RED_BOLD}Attempting to restart PostgreSQL...{RESET}")
			
			restart_ok, restart_msg = restart_postgresql(args.db, args.psql, args.host, args.port, args.verbose)
			if restart_ok:
				# Wait a bit more and verify connection is stable
				time.sleep(2)
				conn_ok, _, _ = check_postgresql_connection(args.db, args.psql, args.host, args.port)
				if conn_ok:
					print(f"    {GREEN_BOLD}✓ PostgreSQL restarted successfully: {restart_msg}{RESET}")
					# Reconfigure compute mode after restart
					mode_ok = switch_gpu_mode(args.db, args.compute, args.psql, args.gpu_kernels, args.verbose)
					if mode_ok:
						print(f"    {GREEN_BOLD}✓ Reconfigured for {args.compute.upper()} compute mode{RESET}")
					else:
						print(f"    {RED_BOLD}⚠ Failed to reconfigure mode after restart{RESET}")
				else:
					print(f"    {RED_BOLD}⚠ PostgreSQL restarted but connection verification failed{RESET}")
					print(f"    {RED_BOLD}Waiting 5 more seconds and retrying...{RESET}")
					time.sleep(5)
					conn_ok, _, _ = check_postgresql_connection(args.db, args.psql, args.host, args.port)
					if conn_ok:
						print(f"    {GREEN_BOLD}✓ Connection verified after additional wait{RESET}")
						mode_ok = switch_gpu_mode(args.db, args.compute, args.psql, args.gpu_kernels, args.verbose)
						if mode_ok:
							print(f"    {GREEN_BOLD}✓ Reconfigured for {args.compute.upper()} compute mode{RESET}")
					else:
						print(f"    {RED_BOLD}✗ Connection still failing after restart{RESET}")
			else:
				print(f"    {RED_BOLD}✗ Failed to restart PostgreSQL: {restart_msg}{RESET}")
				print(f"    {RED_BOLD}Please restart PostgreSQL manually and re-run tests{RESET}")
				# Continue anyway - maybe it will come back, or user will restart manually
		
		# Persist artifacts
		write_artifacts(name, ok, DEFAULT_OUTPUT_DIR, DEFAULT_ERROR_DIR, out_text, err_text)
		
		if ok and not critical_crash:
			passed += 1
		else:
			failed += 1
		
		# Overwrite the starting line with final result (colored: green ✓ or red ✗ or crash 💥)
		print(format_test_line(ok, when, idx, total, name, elapsed, critical_crash))
		
		# Verify GPU usage for ML training tests in GPU or auto mode (only if not crashed)
		if ok and not critical_crash and args.compute in ("gpu", "auto") and ("train" in name.lower() or "linreg" in name.lower() or "logreg" in name.lower() or "rf" in name.lower() or "svm" in name.lower() or "ridge" in name.lower() or "lasso" in name.lower() or "dt" in name.lower() or "nb" in name.lower()):
			gpu_ok, gpu_err = verify_gpu_usage(args.db, args.psql, args.compute, name, args.host, args.port)
			if not gpu_ok:
				print(f"    {RED_BOLD}⚠ GPU Verification Failed: {gpu_err}{RESET}")
				# Don't mark test as failed, just warn
		
		# Only show error details in verbose mode
		if not ok and args.verbose:
			err_tail = "\n".join((err_text or "").strip().splitlines()[-5:])
			if err_tail:
				print(f"    {err_tail}")

	# Print separator after tests
	print()
	print(HEADER_SEPARATOR)
	print()

	# Print test report
	total_elapsed = time.perf_counter() - t0
	
	if _shutdown_requested:
		print("Test Report (Interrupted):")
	else:
		print("Test Report:")
	
	print()
	print(f"   Total Tests:        {total}")
	print(f"   Completed:          {idx if 'idx' in locals() else 0}")
	print(f"   Passed:             {passed}")
	print(f"   Failed:             {failed}")
	if critical_crashes > 0:
		print(f"   {RED_BOLD}Critical Crashes:   {critical_crashes}{RESET}")
	print(f"   Total Elapsed:      {total_elapsed:.2f}s")
	print()
	
	# Generate coverage report (enabled by default, unless --no-gcov is specified)
	# Default: enabled, --no-gcov: disabled, --coverage: enabled (for backward compatibility)
	enable_coverage = not getattr(args, 'no_coverage', False)
	if enable_coverage:
		print()
		print(HEADER_SEPARATOR)
		print()
		print(f"{BOLD}Generating Code Coverage Report...{RESET}")
		print()
		
		# Determine project root (parent of tests directory)
		project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
		
		# Determine coverage directory
		if args.coverage_dir:
			coverage_dir = os.path.abspath(args.coverage_dir)
		else:
			coverage_dir = os.path.join(os.path.dirname(__file__), "coverage")
		
		# Check if code was compiled with coverage flags
		coverage_compiled, coverage_msg = check_coverage_enabled(project_root)
		if not coverage_compiled:
			print(f"{RED_BOLD}⚠ Coverage Warning:{RESET} {coverage_msg}")
			print()
			print("To enable coverage:")
			print("  1. Compile code with coverage flags:")
			print("     make clean")
			print("     make CFLAGS='-fprofile-arcs -ftest-coverage' LDFLAGS='-lgcov'")
			print("  2. Or use: make COVERAGE=1")
			print()
		else:
			print(f"{GREEN_BOLD}✓{RESET} {coverage_msg}")
			print()
			
			# Find gcov
			gcov_path = find_gcov()
			if not gcov_path:
				print(f"{RED_BOLD}✗{RESET} gcov not found. Install gcc or build-essential package.")
			else:
				print(f"{GREEN_BOLD}✓{RESET} Found gcov: {gcov_path}")
				
				# Collect coverage data
				print("Collecting coverage data...")
				gcov_success, gcov_files = collect_gcov_data(project_root, coverage_dir, gcov_path, args.verbose)
				
				if gcov_success:
					print(f"{GREEN_BOLD}✓{RESET} Collected coverage data from {len(gcov_files)} files")
					
					# Generate reports
					summary = {"lines": 0.0, "branches": 0.0, "functions": 0.0}
					files = []
					html_report = None
					xml_report = None
					
					# Try gcovr first (preferred)
					gcovr_path = find_gcovr()
					if gcovr_path:
						print(f"{GREEN_BOLD}✓{RESET} Found gcovr: {gcovr_path}")
						
						# Generate text report
						txt_report = os.path.join(coverage_dir, "coverage.txt")
						success, output = generate_gcovr_report(project_root, coverage_dir, txt_report, "txt", gcovr_path)
						if success:
							print(f"{GREEN_BOLD}✓{RESET} Generated text report: {txt_report}")
							summary, files = parse_gcov_summary(output)
							# Also print summary to console if verbose
							if args.verbose and output:
								print()
								for line in output.split('\n')[:30]:  # First 30 lines
									if line.strip():
										print(f"  {line}")
								print()
						else:
							print(f"{RED_BOLD}✗{RESET} Failed to generate text report: {output}")
						
						# Generate HTML report (enabled by default, or if explicitly requested)
						generate_html = args.coverage_html or True  # Default to True for detailed view
						if generate_html:
							html_report = os.path.join(coverage_dir, "coverage.html")
							success, output = generate_gcovr_report(project_root, coverage_dir, html_report, "html", gcovr_path)
							if success:
								print(f"{GREEN_BOLD}✓{RESET} Generated HTML report: {html_report}")
								print(f"     Open in browser to see line-by-line coverage details")
							else:
								print(f"{RED_BOLD}✗{RESET} Failed to generate HTML report: {output}")
						
						# Generate XML report if requested
						if args.coverage_xml:
							xml_report = os.path.join(coverage_dir, "coverage.xml")
							success, output = generate_gcovr_report(project_root, coverage_dir, xml_report, "xml", gcovr_path)
							if success:
								print(f"{GREEN_BOLD}✓{RESET} Generated XML report: {xml_report}")
							else:
								print(f"{RED_BOLD}✗{RESET} Failed to generate XML report: {output}")
					else:
						# Try lcov as fallback
						lcov_path = find_lcov()
						if lcov_path:
							print(f"{GREEN_BOLD}✓{RESET} Found lcov: {lcov_path}")
							lcov_file = os.path.join(coverage_dir, "coverage.info")
							success, output = generate_lcov_report(project_root, coverage_dir, lcov_file, lcov_path)
							if success:
								print(f"{GREEN_BOLD}✓{RESET} {output}")
								html_report = os.path.join(coverage_dir, "html", "index.html")
							else:
								print(f"{RED_BOLD}✗{RESET} {output}")
						else:
							# Fallback: create a simple summary from .gcov files
							print(f"{RED_BOLD}⚠{RESET} Neither gcovr nor lcov found. Generating basic summary from .gcov files.")
							print("  Install gcovr for better reports: pip install gcovr")
							print("  Or install lcov: apt-get install lcov or brew install lcov")
							
							# Create a simple text summary
							txt_report = os.path.join(coverage_dir, "coverage_summary.txt")
							try:
								with open(txt_report, "w") as f:
									f.write("NeuronDB Code Coverage Summary (from .gcov files)\n")
									f.write("=" * 80 + "\n\n")
									f.write(f"Coverage data files found: {len(gcov_files)}\n")
									f.write(f"Coverage directory: {coverage_dir}\n\n")
									f.write("Note: Install gcovr or lcov for detailed coverage percentages.\n")
									f.write("      gcovr: pip install gcovr\n")
									f.write("      lcov: apt-get install lcov or brew install lcov\n")
									f.write("\n")
									f.write("Detailed .gcov files are available in the coverage directory.\n")
									f.write("Each .gcov file shows:\n")
									f.write("  - Line numbers\n")
									f.write("  - Execution counts (how many times each line was executed)\n")
									f.write("  - Uncovered lines (marked with #####)\n")
								print(f"{GREEN_BOLD}✓{RESET} Created basic summary: {txt_report}")
							except Exception as e:
								if args.verbose:
									print(f"Warning: Failed to create summary file: {e}", file=sys.stderr)
					
					# Print coverage report with detailed breakdown
					print_coverage_report(coverage_dir, summary, html_report, xml_report, files, project_root)
				else:
					print(f"{RED_BOLD}✗{RESET} Failed to collect coverage data")
					print("  Make sure code was compiled with coverage flags and tests were executed.")
		
		print()

	if _shutdown_requested:
		return 130  # Exit code for SIGINT
	return 0 if failed == 0 else 1


if __name__ == "__main__":
	sys.exit(main())


