/*-------------------------------------------------------------------------
 *
 * sandbox_enhanced.go
 *    Enhanced sandbox for tool execution
 *
 * Provides sandboxed code execution with resource limits, file allowlists,
 * network egress rules, and container-based isolation.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/tools/sandbox_enhanced.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"
)

/* SandboxConfig defines sandbox configuration */
type SandboxConfig struct {
	MaxMemory    int64         `json:"max_memory"`    // Bytes
	MaxCPU       float64       `json:"max_cpu"`       // CPU percentage (0-100)
	MaxDisk      int64         `json:"max_disk"`      // Bytes
	Timeout      time.Duration `json:"timeout"`       // Execution timeout
	AllowedDirs  []string      `json:"allowed_dirs"`  // Allowed directories
	AllowedFiles []string      `json:"allowed_files"` // Allowed specific files
	NetworkRules NetworkRules  `json:"network_rules"` // Network egress rules
	Isolation    IsolationType `json:"isolation"`     // Isolation type
}

/* NetworkRules defines network egress rules */
type NetworkRules struct {
	AllowedDomains []string `json:"allowed_domains"` // Allowed domain names
	AllowedIPs     []string `json:"allowed_ips"`     // Allowed IP addresses/CIDR
	BlockAll       bool     `json:"block_all"`       // Block all network access
}

/* IsolationType defines isolation method */
type IsolationType string

const (
	IsolationNone      IsolationType = "none"      // No isolation
	IsolationChroot    IsolationType = "chroot"    // Chroot isolation
	IsolationContainer IsolationType = "container" // Container isolation (Docker, etc.)
)

/* EnhancedSandbox provides enhanced sandboxing capabilities */
type EnhancedSandbox struct {
	config      SandboxConfig
	base        *Sandbox
	dockerClient DockerClient
	mu          sync.Mutex
	containers  map[string]string /* containerID -> containerName mapping for cleanup */
}

/* DockerClient interface for Docker operations */
type DockerClient interface {
	CreateContainer(ctx context.Context, config ContainerConfig) (string, error)
	StartContainer(ctx context.Context, containerID string) error
	WaitContainer(ctx context.Context, containerID string) error
	GetContainerLogs(ctx context.Context, containerID string) ([]byte, error)
	RemoveContainer(ctx context.Context, containerID string) error
	ContainerExists(ctx context.Context, containerID string) (bool, error)
}

/* ContainerConfig defines container configuration */
type ContainerConfig struct {
	Image       string
	Command     []string
	WorkingDir  string
	Env         []string
	MemoryLimit int64  /* bytes */
	CPULimit    float64 /* percentage */
	NetworkMode string
	Volumes     map[string]string /* hostPath -> containerPath */
	AutoRemove  bool
}

/* NewEnhancedSandbox creates a new enhanced sandbox */
func NewEnhancedSandbox(config SandboxConfig) *EnhancedSandbox {
	sandbox := &EnhancedSandbox{
		config:     config,
		base:       NewSandbox("", config.MaxMemory, int(config.MaxCPU)),
		containers: make(map[string]string),
	}

	/* Initialize Docker client if container isolation is enabled */
	if config.Isolation == IsolationContainer {
		sandbox.dockerClient = NewDockerClient()
	}

	return sandbox
}

/* ExecuteCommand executes a command in the sandbox */
func (s *EnhancedSandbox) ExecuteCommand(ctx context.Context, command string, args []string, workingDir string) ([]byte, error) {
	/* Create context with timeout */
	if s.config.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, s.config.Timeout)
		defer cancel()
	}

	/* Validate working directory */
	if workingDir != "" {
		if err := s.validatePath(workingDir); err != nil {
			return nil, fmt.Errorf("invalid working directory: %w", err)
		}
	}

	/* Validate command path */
	if err := s.validatePath(command); err != nil {
		return nil, fmt.Errorf("invalid command path: %w", err)
	}

	/* Create command */
	cmd := exec.CommandContext(ctx, command, args...)
	if workingDir != "" {
		cmd.Dir = workingDir
	}

	/* Apply resource limits */
	if s.base != nil {
		if err := s.base.ApplyResourceLimits(cmd); err != nil {
			return nil, fmt.Errorf("failed to apply resource limits: %w", err)
		}
	}

	/* Apply isolation based on type */
	switch s.config.Isolation {
	case IsolationChroot:
		/* Apply chroot isolation if possible */
		if err := s.applyChrootIsolation(cmd, workingDir); err != nil {
			/* Log warning but continue - chroot may not be available */
			fmt.Printf("Warning: chroot isolation failed: %v\n", err)
		}
	case IsolationContainer:
		/* Execute in Docker container */
		return s.executeInContainer(ctx, command, args, workingDir)
	}

	/* Execute command */
	output, err := cmd.CombinedOutput()
	if err != nil {
		return output, fmt.Errorf("command execution failed: %w", err)
	}

	return output, nil
}

/* validatePath validates that a path is allowed */
func (s *EnhancedSandbox) validatePath(path string) error {
	/* Resolve absolute path */
	absPath, err := filepath.Abs(path)
	if err != nil {
		return fmt.Errorf("failed to resolve path: %w", err)
	}

	/* Check if path is in allowed directories */
	if len(s.config.AllowedDirs) > 0 {
		allowed := false
		for _, allowedDir := range s.config.AllowedDirs {
			allowedAbs, err := filepath.Abs(allowedDir)
			if err != nil {
				continue
			}
			if strings.HasPrefix(absPath, allowedAbs) {
				allowed = true
				break
			}
		}
		if !allowed {
			return fmt.Errorf("path not in allowed directories: %s", path)
		}
	}

	/* Check if path is in allowed files */
	if len(s.config.AllowedFiles) > 0 {
		for _, allowedFile := range s.config.AllowedFiles {
			allowedAbs, err := filepath.Abs(allowedFile)
			if err != nil {
				continue
			}
			if absPath == allowedAbs {
				return nil /* Allowed */
			}
		}
		/* If allowed files specified but not found, reject */
		if len(s.config.AllowedDirs) == 0 {
			return fmt.Errorf("path not in allowed files: %s", path)
		}
	}

	return nil
}

/* ValidateNetworkAccess validates network access based on rules */
func (s *EnhancedSandbox) ValidateNetworkAccess(host string) error {
	if s.config.NetworkRules.BlockAll {
		return fmt.Errorf("network access blocked")
	}

	/* Check allowed domains */
	if len(s.config.NetworkRules.AllowedDomains) > 0 {
		allowed := false
		for _, domain := range s.config.NetworkRules.AllowedDomains {
			if strings.HasSuffix(host, domain) || host == domain {
				allowed = true
				break
			}
		}
		if !allowed {
			return fmt.Errorf("domain not in allowed list: %s", host)
		}
	}

	/* Check allowed IPs */
	if len(s.config.NetworkRules.AllowedIPs) > 0 {
		hostIP, err := net.LookupIP(host)
		if err != nil {
			return fmt.Errorf("failed to resolve host IP: %w", err)
		}

		allowed := false
		for _, allowedIPStr := range s.config.NetworkRules.AllowedIPs {
			/* Try parsing as CIDR */
			_, ipNet, err := net.ParseCIDR(allowedIPStr)
			if err == nil {
				/* It's a CIDR, check if host IP is in range */
				for _, ip := range hostIP {
					if ipNet.Contains(ip) {
						allowed = true
						break
					}
				}
				if allowed {
					break
				}
			} else {
				/* Try parsing as single IP */
				allowedIP := net.ParseIP(allowedIPStr)
				if allowedIP != nil {
					for _, ip := range hostIP {
						if ip.Equal(allowedIP) {
							allowed = true
							break
						}
					}
					if allowed {
						break
					}
				}
			}
		}

		if !allowed {
			return fmt.Errorf("IP not in allowed list: %s", host)
		}
	}

	return nil
}

/* applyChrootIsolation applies chroot isolation to command */
func (s *EnhancedSandbox) applyChrootIsolation(cmd *exec.Cmd, workingDir string) error {
	/* Chroot requires root privileges or CAP_SYS_CHROOT capability */
	/* Check if we have the capability */
	if os.Geteuid() != 0 {
		/* Not root, chroot won't work */
		return fmt.Errorf("chroot requires root privileges")
	}

	/* Create isolated directory if workingDir is set */
	chrootDir := workingDir
	if chrootDir == "" {
		/* Create temporary chroot directory */
		tmpDir, err := os.MkdirTemp("", "sandbox-chroot-*")
		if err != nil {
			return fmt.Errorf("failed to create chroot directory: %w", err)
		}
		chrootDir = tmpDir
		defer os.RemoveAll(tmpDir)
	}

	/* Set up chroot in command */
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Chroot: chrootDir,
	}

	return nil
}

/* SetNetworkRules sets network egress rules */
func (s *EnhancedSandbox) SetNetworkRules(rules NetworkRules) {
	s.config.NetworkRules = rules
}

/* ApplyFileAllowlist applies file allowlist to command environment */
func (s *EnhancedSandbox) ApplyFileAllowlist(cmd *exec.Cmd) error {
	/* Set environment variable with allowed files */
	if len(s.config.AllowedFiles) > 0 {
		allowedFilesStr := strings.Join(s.config.AllowedFiles, ":")
		cmd.Env = append(os.Environ(), "SANDBOX_ALLOWED_FILES="+allowedFilesStr)
	}

	/* Set allowed directories */
	if len(s.config.AllowedDirs) > 0 {
		allowedDirsStr := strings.Join(s.config.AllowedDirs, ":")
		cmd.Env = append(cmd.Env, "SANDBOX_ALLOWED_DIRS="+allowedDirsStr)
	}

	return nil
}

/* executeInContainer executes command in Docker container */
func (s *EnhancedSandbox) executeInContainer(ctx context.Context, command string, args []string, workingDir string) ([]byte, error) {
	if s.dockerClient == nil {
		return nil, fmt.Errorf("Docker client not initialized")
	}

	/* Prepare container configuration */
	containerConfig := ContainerConfig{
		Image:       "alpine:latest", /* Default base image */
		Command:     append([]string{command}, args...),
		WorkingDir:  workingDir,
		MemoryLimit: s.config.MaxMemory,
		CPULimit:    s.config.MaxCPU,
		NetworkMode: "none", /* Isolated network by default */
		AutoRemove:  true,
	}

	/* Set network mode based on rules */
	if !s.config.NetworkRules.BlockAll {
		containerConfig.NetworkMode = "bridge" /* Allow network access */
	}

	/* Create container */
	containerID, err := s.dockerClient.CreateContainer(ctx, containerConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create container: %w", err)
	}

	/* Track container for cleanup */
	s.mu.Lock()
	s.containers[containerID] = containerID
	s.mu.Unlock()

	/* Ensure cleanup */
	defer func() {
		s.mu.Lock()
		delete(s.containers, containerID)
		s.mu.Unlock()
		_ = s.dockerClient.RemoveContainer(context.Background(), containerID)
	}()

	/* Start container */
	if err := s.dockerClient.StartContainer(ctx, containerID); err != nil {
		return nil, fmt.Errorf("failed to start container: %w", err)
	}

	/* Wait for container to complete */
	if err := s.dockerClient.WaitContainer(ctx, containerID); err != nil {
		return nil, fmt.Errorf("container execution failed: %w", err)
	}

	/* Get container logs */
	logs, err := s.dockerClient.GetContainerLogs(ctx, containerID)
	if err != nil {
		return nil, fmt.Errorf("failed to get container logs: %w", err)
	}

	return logs, nil
}

/* CleanupContainers removes all tracked containers */
func (s *EnhancedSandbox) CleanupContainers(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	var errors []string
	for containerID := range s.containers {
		if err := s.dockerClient.RemoveContainer(ctx, containerID); err != nil {
			errors = append(errors, fmt.Sprintf("failed to remove container %s: %v", containerID, err))
		}
		delete(s.containers, containerID)
	}

	if len(errors) > 0 {
		return fmt.Errorf("cleanup errors: %s", strings.Join(errors, "; "))
	}

	return nil
}

/* DefaultSandboxConfig returns default sandbox configuration */
func DefaultSandboxConfig() SandboxConfig {
	return SandboxConfig{
		MaxMemory: 512 * 1024 * 1024,      /* 512 MB */
		MaxCPU:    50.0,                   /* 50% CPU */
		MaxDisk:   1 * 1024 * 1024 * 1024, /* 1 GB */
		Timeout:   5 * time.Minute,
		Isolation: IsolationNone,
		NetworkRules: NetworkRules{
			BlockAll: true, /* Block all by default */
		},
	}
}
