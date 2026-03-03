# Platform wrapper: macOS (Homebrew)
# Import detected settings from build.sh first
-include Makefile.local

# Prefer Homebrew pg_config only if not defined in Makefile.local
PG_CONFIG ?= /opt/homebrew/bin/pg_config
ifeq ("$(wildcard $(PG_CONFIG))","")
PG_CONFIG := $(shell command -v pg_config 2>/dev/null || echo /usr/local/bin/pg_config)
endif
export PG_CONFIG

# Defer to preserved project build logic
include Makefile.core


