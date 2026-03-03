# Platform wrapper: Debian/Ubuntu
# Import detected settings from build.sh first (PG_CONFIG, CUDA_PATH, etc.)
-include Makefile.local

# Fallback PG_CONFIG only if not provided by Makefile.local or environment
PG_CONFIG ?= /usr/bin/pg_config
export PG_CONFIG

# The original top-level Makefile is preserved as Makefile.core by build.sh
# to avoid losing project build logic. We include it here.
include Makefile.core


