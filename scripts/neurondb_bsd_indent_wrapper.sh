#!/bin/bash
# Wrapper for pg_bsd_indent that filters out profiling warnings
# Only filter stderr, not stdout, and preserve exit codes
/home/pge/pge/postgres.18/src/tools/pg_bsd_indent/pg_bsd_indent "$@" 2> >(grep -v "^profiling:" >&2)
exit ${PIPESTATUS[0]}
