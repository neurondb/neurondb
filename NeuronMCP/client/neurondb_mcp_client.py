#!/usr/bin/env python3
"""
NeuronMCP CLI Client

A professional command-line interface for interacting with NeuronMCP servers.
Supports Claude Desktop configuration format and batch command execution.
"""

import argparse
import sys
import os
from pathlib import Path

# Add client directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from mcp_client.client import MCPClient
from mcp_client.config import load_config
from mcp_client.output import OutputManager


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="NeuronMCP CLI Client - Connect to MCP servers and execute commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Execute single command
  %(prog)s -c neuronmcp_server.json -e "list_tools"

  # Execute commands from file
  %(prog)s -c neuronmcp_server.json -f commands.txt

  # Execute commands and save output
  %(prog)s -c neuronmcp_server.json -f commands.txt -o results.json

  # Verbose mode
  %(prog)s -c neuronmcp_server.json -e "list_tools" -v
        """
    )

    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Path to NeuronMCP server configuration file (neuronmcp_server.json)"
    )

    parser.add_argument(
        "-e", "--execute",
        help="Execute a single command (format: tool_name or tool_name:arg1=val1,arg2=val2)"
    )

    parser.add_argument(
        "-f", "--file",
        help="Path to file containing commands to execute (one per line)"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output file path for results (default: results_<timestamp>.json)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--server-name",
        default="neurondb",
        help="Server name from config (default: neurondb)"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.execute and not args.file:
        print("Error: Either -e/--execute or -f/--file must be provided", file=sys.stderr)
        sys.exit(1)

    if args.execute and args.file:
        print("Error: Cannot use both -e/--execute and -f/--file", file=sys.stderr)
        sys.exit(1)

    # Load configuration
    try:
        config = load_config(args.config, args.server_name)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # Initialize output manager
    output_manager = OutputManager(args.output)

    # Create and connect client
    try:
        client = MCPClient(config, verbose=args.verbose)
        client.connect()
    except Exception as e:
        print(f"Error connecting to MCP server: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        # Execute commands
        if args.execute:
            # Single command execution
            result = client.execute_command(args.execute)
            output_manager.add_result(args.execute, result)
            print(f"Command executed: {args.execute}")
            if args.verbose:
                print(f"Result: {result}")

        elif args.file:
            # Batch command execution
            if not os.path.exists(args.file):
                print(f"Error: Command file not found: {args.file}", file=sys.stderr)
                sys.exit(1)

            with open(args.file, 'r') as f:
                commands = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

            print(f"Executing {len(commands)} commands from {args.file}...")
            for i, command in enumerate(commands, 1):
                print(f"[{i}/{len(commands)}] Executing: {command}")
                try:
                    result = client.execute_command(command)
                    output_manager.add_result(command, result)
                    if args.verbose:
                        print(f"  Result: {result}")
                except Exception as e:
                    error_result = {"error": str(e), "command": command}
                    output_manager.add_result(command, error_result)
                    print(f"  Error: {e}", file=sys.stderr)

        # Save output
        output_file = output_manager.save()
        print(f"\nResults saved to: {output_file}")

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during execution: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()

