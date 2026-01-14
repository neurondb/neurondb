/*-------------------------------------------------------------------------
 *
 * main.go
 *    CLI tool for NeuronAgent management (Legacy - use cli/main.go)
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/cmd/neuronagent-cli/main.go
 *
 *-------------------------------------------------------------------------
 *
 * NOTE: This is the legacy CLI implementation. The new comprehensive CLI
 * is located in cli/main.go. This file is kept for backward compatibility.
 *
 *-------------------------------------------------------------------------
 */

package main

import (
	"fmt"
	"os"

	"github.com/neurondb/NeuronAgent/cli/cmd"
)

func main() {
	fmt.Fprintf(os.Stderr, "Warning: This CLI is deprecated. Please use the new CLI at cli/main.go\n")
	cmd.Execute()
}
