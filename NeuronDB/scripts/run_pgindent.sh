#!/bin/bash
#
# Script to run pg_indent on all C and header files in NeuronDB
#
# Usage:
#   ./run_pgindent.sh [--check|--diff] [files...]
#
# Options:
#   --check    Show which files would be changed (dry-run)
#   --diff     Show the diff of changes that would be made
#   files      Specific files to format (default: all .c and .h files)
#

set -e

# Configuration
PG_INDENT_SCRIPT="/home/pge/pge/postgres.18/src/tools/pgindent/pgindent"
PG_BSD_INDENT_WRAPPER="$(dirname "$0")/pg_bsd_indent_wrapper.sh"
PG_BSD_INDENT="/home/pge/pge/postgres.18/src/tools/pg_bsd_indent/pg_bsd_indent"
TYPEDEFS_FILE="$(dirname "$0")/../typedefs.list"
NEURONDB_ROOT="$(dirname "$0")/.."

# Check if pgindent script exists
if [ ! -f "$PG_INDENT_SCRIPT" ]; then
    echo "Error: pgindent script not found at $PG_INDENT_SCRIPT" >&2
    exit 1
fi

# Check if pg_bsd_indent binary exists
if [ ! -f "$PG_BSD_INDENT" ]; then
    echo "Error: pg_bsd_indent binary not found at $PG_BSD_INDENT" >&2
    echo "Please build it first: cd $(dirname "$PG_BSD_INDENT") && make" >&2
    exit 1
fi

# Create wrapper script if it doesn't exist
if [ ! -f "$PG_BSD_INDENT_WRAPPER" ]; then
    cat > "$PG_BSD_INDENT_WRAPPER" << 'WRAPPER_EOF'
#!/bin/bash
# Wrapper for pg_bsd_indent that filters out profiling warnings
# Only filter stderr, not stdout, and preserve exit codes
/home/pge/pge/postgres.18/src/tools/pg_bsd_indent/pg_bsd_indent "$@" 2> >(grep -v "^profiling:" >&2)
exit ${PIPESTATUS[0]}
WRAPPER_EOF
    chmod +x "$PG_BSD_INDENT_WRAPPER"
fi

# Check if typedefs file exists
if [ ! -f "$TYPEDEFS_FILE" ]; then
    echo "Error: typedefs.list not found at $TYPEDEFS_FILE" >&2
    exit 1
fi

# Change to NeuronDB root directory
cd "$NEURONDB_ROOT" || exit 1

# Parse arguments
MODE=""
FILES=()

for arg in "$@"; do
    case "$arg" in
        --check)
            MODE="--check"
            ;;
        --diff)
            MODE="--diff"
            ;;
        --help|-h)
            echo "Usage: $0 [--check|--diff] [files...]"
            echo ""
            echo "Options:"
            echo "  --check    Show which files would be changed (dry-run)"
            echo "  --diff     Show the diff of changes that would be made"
            echo "  files      Specific files to format (default: all .c and .h files)"
            exit 0
            ;;
        *)
            FILES+=("$arg")
            ;;
    esac
done

# Set up environment
export PGINDENT="$PG_INDENT_SCRIPT"
export PGINDENT_INDENT="$PG_BSD_INDENT"
export PGTYPEDEFS="$TYPEDEFS_FILE"
# Suppress profiling warnings from pg_bsd_indent
export GCOV_PREFIX="/dev/null"
export GCOV_PREFIX_STRIP=999

# Build command
CMD=("$PG_INDENT_SCRIPT")
CMD+=("--typedefs=$TYPEDEFS_FILE")
CMD+=("--indent=$PG_BSD_INDENT_WRAPPER")

if [ -n "$MODE" ]; then
    CMD+=("$MODE")
fi

# If specific files provided, use them; otherwise find all .c and .h files
if [ ${#FILES[@]} -eq 0 ]; then
    echo "Finding all .c and .h files in src/ and include/ directories..."
    # Find all .c and .h files, excluding object files and other generated files
    while IFS= read -r -d '' file; do
        FILES+=("$file")
    done < <(find src include -type f \( -name "*.c" -o -name "*.h" \) ! -name "*.o" ! -name "*.gcov" ! -name "*.gcda" -print0 2>/dev/null)
fi

if [ ${#FILES[@]} -eq 0 ]; then
    echo "No files found to process." >&2
    exit 1
fi

echo "Processing ${#FILES[@]} file(s)..."
echo ""

# Run pgindent
"${CMD[@]}" "${FILES[@]}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    if [ "$MODE" = "--check" ]; then
        echo ""
        echo "All files are properly formatted!"
    elif [ "$MODE" = "--diff" ]; then
        echo ""
        echo "Diff complete."
    else
        echo ""
        echo "Formatting complete!"
    fi
elif [ $EXIT_CODE -eq 2 ]; then
    echo ""
    echo "Some files need formatting. Run without --check to apply changes."
    exit 2
else
    echo ""
    echo "Error occurred during formatting." >&2
    exit $EXIT_CODE
fi

