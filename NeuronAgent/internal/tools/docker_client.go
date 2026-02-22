/*-------------------------------------------------------------------------
 *
 * docker_client.go
 *    Docker client for container isolation
 *
 * Provides Docker container operations for sandboxed execution.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/tools/docker_client.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"regexp"
	"strings"
	"time"
)

/* dockerCLIClient implements DockerClient using docker CLI */
type dockerCLIClient struct{}

/* Docker image name: alphanumeric, slash, period, hyphen, underscore, colon (for tag) */
var dockerImageRe = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9._/-]*([:][a-zA-Z0-9._-]+)?$`)

/* ValidateContainerConfig validates config to prevent injection and path escape */
func ValidateContainerConfig(config ContainerConfig) error {
	if config.Image == "" {
		return fmt.Errorf("container image is required")
	}
	if !dockerImageRe.MatchString(config.Image) {
		return fmt.Errorf("invalid container image name: only alphanumeric, ., _, -, /, : allowed")
	}
	for hostPath, containerPath := range config.Volumes {
		if strings.Contains(hostPath, "..") || strings.Contains(containerPath, "..") {
			return fmt.Errorf("volume paths must not contain ..")
		}
		if strings.HasPrefix(hostPath, "-") || strings.HasPrefix(containerPath, "-") {
			return fmt.Errorf("volume paths must not start with -")
		}
	}
	for _, env := range config.Env {
		if strings.Contains(env, "\x00") || strings.Contains(env, "\n") {
			return fmt.Errorf("environment variable must not contain null or newline")
		}
		parts := strings.SplitN(env, "=", 2)
		if len(parts) != 2 || parts[0] == "" {
			return fmt.Errorf("environment variable must be KEY=VALUE")
		}
		for _, r := range parts[0] {
			if r != '_' && r != '-' && (r < 'a' || r > 'z') && (r < 'A' || r > 'Z') && (r < '0' || r > '9') {
				return fmt.Errorf("environment variable name contains invalid character")
			}
		}
	}
	for _, c := range config.Command {
		if strings.Contains(c, "\x00") || strings.ContainsAny(c, ";&|$`\\<>()\n\r\t") {
			return fmt.Errorf("command argument contains invalid character")
		}
	}
	return nil
}

/* NewDockerClient creates a new Docker client */
func NewDockerClient() DockerClient {
	return &dockerCLIClient{}
}

/* Default timeouts for Docker operations when context has no deadline */
const (
	dockerCreateTimeout = 30 * time.Second
	dockerStartTimeout  = 10 * time.Second
	dockerWaitTimeout   = 5 * time.Minute
	dockerLogsTimeout   = 10 * time.Second
)

/* withTimeout ensures ctx has a deadline for Docker operations */
func withDockerTimeout(ctx context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
	if _, ok := ctx.Deadline(); ok {
		return ctx, func() {}
	}
	return context.WithTimeout(ctx, timeout)
}

/* CreateContainer creates a Docker container */
func (d *dockerCLIClient) CreateContainer(ctx context.Context, config ContainerConfig) (string, error) {
	if err := ValidateContainerConfig(config); err != nil {
		return "", fmt.Errorf("invalid container config: %w", err)
	}
	ctx, cancel := withDockerTimeout(ctx, dockerCreateTimeout)
	defer cancel()
	/* Build docker create command */
	args := []string{"create", "--rm"}

	/* Set memory limit */
	if config.MemoryLimit > 0 {
		args = append(args, fmt.Sprintf("--memory=%d", config.MemoryLimit))
	}

	/* Set CPU limit */
	if config.CPULimit > 0 {
		/* Convert percentage to CPU shares (1024 = 100%) */
		cpuShares := int(config.CPULimit * 10.24)
		args = append(args, fmt.Sprintf("--cpu-shares=%d", cpuShares))
	}

	/* Set network mode */
	if config.NetworkMode != "" {
		args = append(args, fmt.Sprintf("--network=%s", config.NetworkMode))
	}

	/* Set working directory */
	if config.WorkingDir != "" {
		args = append(args, fmt.Sprintf("--workdir=%s", config.WorkingDir))
	}

	/* Set environment variables */
	for _, env := range config.Env {
		args = append(args, fmt.Sprintf("--env=%s", env))
	}

	/* Mount volumes */
	for hostPath, containerPath := range config.Volumes {
		args = append(args, fmt.Sprintf("--volume=%s:%s", hostPath, containerPath))
	}

	/* Add image and command */
	args = append(args, config.Image)
	args = append(args, config.Command...)

	/* Execute docker create */
	cmd := exec.CommandContext(ctx, "docker", args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("docker create failed: %w, stderr: %s", err, stderr.String())
	}

	containerID := strings.TrimSpace(stdout.String())
	if containerID == "" {
		return "", fmt.Errorf("docker create returned empty container ID")
	}

	return containerID, nil
}

/* StartContainer starts a Docker container */
func (d *dockerCLIClient) StartContainer(ctx context.Context, containerID string) error {
	ctx, cancel := withDockerTimeout(ctx, dockerStartTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, "docker", "start", containerID)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("docker start failed: %w, stderr: %s", err, stderr.String())
	}

	return nil
}

/* WaitContainer waits for a container to complete */
func (d *dockerCLIClient) WaitContainer(ctx context.Context, containerID string) error {
	ctx, cancel := withDockerTimeout(ctx, dockerWaitTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, "docker", "wait", containerID)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	/* Get exit code */
	var exitCode int
	if err := cmd.Run(); err != nil {
		if exitError, ok := err.(*exec.ExitError); ok {
			exitCode = exitError.ExitCode()
		}
	}

	/* Check container status */
	statusCmd := exec.CommandContext(ctx, "docker", "inspect", "--format={{.State.Status}}", containerID)
	var statusOut bytes.Buffer
	statusCmd.Stdout = &statusOut
	if err := statusCmd.Run(); err == nil {
		status := strings.TrimSpace(statusOut.String())
		if status == "exited" && exitCode != 0 {
			return fmt.Errorf("container exited with code %d", exitCode)
		}
	}

	return nil
}

/* GetContainerLogs retrieves logs from a container */
func (d *dockerCLIClient) GetContainerLogs(ctx context.Context, containerID string) ([]byte, error) {
	ctx, cancel := withDockerTimeout(ctx, dockerLogsTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, "docker", "logs", containerID)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("docker logs failed: %w, stderr: %s", err, stderr.String())
	}

	return stdout.Bytes(), nil
}

/* RemoveContainer removes a Docker container */
func (d *dockerCLIClient) RemoveContainer(ctx context.Context, containerID string) error {
	/* Check if container exists */
	exists, err := d.ContainerExists(ctx, containerID)
	if err != nil || !exists {
		return nil /* Container doesn't exist, nothing to remove */
	}

	cmd := exec.CommandContext(ctx, "docker", "rm", "-f", containerID)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("docker rm failed: %w, stderr: %s", err, stderr.String())
	}

	return nil
}

/* ContainerExists checks if a container exists */
func (d *dockerCLIClient) ContainerExists(ctx context.Context, containerID string) (bool, error) {
	cmd := exec.CommandContext(ctx, "docker", "inspect", containerID)
	if err := cmd.Run(); err != nil {
		return false, nil /* Container doesn't exist */
	}
	return true, nil
}

/* dockerSDKClient implements DockerClient using Docker SDK (for future use) */
/* This would require adding github.com/docker/docker/client to go.mod */
type dockerSDKClient struct {
	/* client *client.Client */
}

/* Note: Full Docker SDK implementation would require:
 * - Adding github.com/docker/docker/client to go.mod
 * - Initializing client with client.NewClientWithOpts()
 * - Using container.Create(), container.Start(), container.Wait(), etc.
 * For now, we use the CLI-based implementation above.
 */
