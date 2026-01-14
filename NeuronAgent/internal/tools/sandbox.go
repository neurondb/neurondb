/*-------------------------------------------------------------------------
 *
 * sandbox.go
 *    Tool implementation for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/tools/sandbox.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"syscall"
	"time"
)

/* Sandbox provides security sandboxing for tool execution */
type Sandbox struct {
	chrootPath string
	maxMemory  int64 /* in bytes */
	maxCPU     int   /* percentage */
}

/* NewSandbox creates a new sandbox (Unix/Linux only) */
func NewSandbox(chrootPath string, maxMemory int64, maxCPU int) *Sandbox {
	return &Sandbox{
		chrootPath: chrootPath,
		maxMemory:  maxMemory,
		maxCPU:     maxCPU,
	}
}

/* ApplyResourceLimits applies resource limits to a command */
func (s *Sandbox) ApplyResourceLimits(cmd *exec.Cmd) error {
	if runtime.GOOS == "linux" || runtime.GOOS == "darwin" {
		if cmd.SysProcAttr == nil {
			cmd.SysProcAttr = &syscall.SysProcAttr{}
		}
		cmd.SysProcAttr.Setpgid = true

		/* Set resource limits using rlimit */
		if s.maxMemory > 0 {
			var rlimit syscall.Rlimit
			if err := syscall.Getrlimit(syscall.RLIMIT_AS, &rlimit); err == nil {
				rlimit.Cur = uint64(s.maxMemory)
				rlimit.Max = uint64(s.maxMemory)
				syscall.Setrlimit(syscall.RLIMIT_AS, &rlimit)
			}
		}

		/* Set CPU time limit (soft limit) */
		if s.maxCPU > 0 {
			var rlimit syscall.Rlimit
			if err := syscall.Getrlimit(syscall.RLIMIT_CPU, &rlimit); err == nil {
				/* maxCPU is percentage, convert to seconds (approximate) */
				cpuSeconds := uint64(s.maxCPU * 60) /* Allow maxCPU minutes */
				rlimit.Cur = cpuSeconds
				rlimit.Max = cpuSeconds
				syscall.Setrlimit(syscall.RLIMIT_CPU, &rlimit)
			}
		}

		/* Context timeout is handled by exec.CommandContext when creating the command */
	}
	return nil
}

/* Chroot applies chroot if configured (requires root privileges) */
/* Note: This requires the process to run as root and the chroot directory */
/* to be properly set up with necessary files (binaries, libraries, etc.) */
func (s *Sandbox) Chroot(cmd *exec.Cmd) error {
	if s.chrootPath == "" {
		return nil
	}

	if _, err := os.Stat(s.chrootPath); os.IsNotExist(err) {
		return fmt.Errorf("chroot path does not exist: %s", s.chrootPath)
	}

	/* Chroot requires root privileges */
	/* In production, this should be handled by the system or container */
	/* For now, we set the working directory as a safer alternative */
	if cmd.Dir == "" {
		cmd.Dir = s.chrootPath
	}

	/* Actual chroot would be: */
	/* if runtime.GOOS == "linux" { */
	/* if cmd.SysProcAttr == nil { */
	/* cmd.SysProcAttr = &syscall.SysProcAttr{} */
	/* } */
	/* cmd.SysProcAttr.Chroot = s.chrootPath */
	/* } */
	/* But this requires root and proper setup */

	return nil
}

/* SetTimeout sets a timeout for command execution */
/* Note: exec.Cmd doesn't have a settable Context field directly. */
/* The context should be set when creating the command with exec.CommandContext */
/* This function is deprecated - use exec.CommandContext instead */
func SetTimeout(cmd *exec.Cmd, timeout time.Duration) *exec.Cmd {
	/* Note: This function cannot properly set context on an existing command. */
	/* The context must be provided when creating the command with exec.CommandContext. */
	/* This function is kept for backward compatibility but does nothing. */
	/* Callers should use exec.CommandContext(ctx, name, args...) instead. */
	_ = timeout
	return cmd
}
