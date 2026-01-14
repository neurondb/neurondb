package handlers

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"time"

	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
	"github.com/neurondb/NeuronDesktop/api/internal/utils"
)

/* FactoryHandlers handles factory/installation status endpoints */
type FactoryHandlers struct {
	queries *db.Queries
}

/* NewFactoryHandlers creates new factory handlers */
func NewFactoryHandlers(queries *db.Queries) *FactoryHandlers {
	return &FactoryHandlers{
		queries: queries,
	}
}

/* FactoryStatusResponse represents the factory status */
type FactoryStatusResponse struct {
	OS              OSInfo          `json:"os"`
	Docker          DockerInfo      `json:"docker"`
	NeuronDB        ComponentStatus `json:"neurondb"`
	NeuronAgent     ComponentStatus `json:"neuronagent"`
	NeuronMCP       ComponentStatus `json:"neuronmcp"`
	InstallCommands InstallCommands `json:"install_commands"`
}

/* OSInfo represents OS information */
type OSInfo struct {
	Type    string `json:"type"`    // "linux", "darwin", "windows"
	Distro  string `json:"distro"`  // "ubuntu", "debian", "rhel", "rocky", "macos"
	Version string `json:"version"` // OS version
	Arch    string `json:"arch"`    // "amd64", "arm64"
}

/* DockerInfo represents Docker availability */
type DockerInfo struct {
	Available bool   `json:"available"`
	Version   string `json:"version,omitempty"`
}

/* ComponentStatus represents the status of a component */
type ComponentStatus struct {
	Installed    bool                   `json:"installed"`
	Running      bool                   `json:"running"`
	Reachable    bool                   `json:"reachable"`
	Status       string                 `json:"status"` // "installed", "running", "reachable", "missing", "error"
	ErrorMessage string                 `json:"error_message,omitempty"`
	Details      map[string]interface{} `json:"details,omitempty"`
}

/* InstallCommands provides OS-specific install commands */
type InstallCommands struct {
	Docker []string `json:"docker,omitempty"`
	Deb    []string `json:"deb,omitempty"`
	Rpm    []string `json:"rpm,omitempty"`
	MacPkg []string `json:"macpkg,omitempty"`
}

/* GetSetupState returns the setup completion state */
func (h *FactoryHandlers) GetSetupState(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	setting, err := h.queries.GetSetting(ctx, "setup_complete")
	if err != nil {
		WriteSuccess(w, map[string]interface{}{
			"setup_complete": false,
		}, http.StatusOK)
		return
	}

	completed, _ := setting.Value["completed"].(bool)
	WriteSuccess(w, map[string]interface{}{
		"setup_complete": completed,
	}, http.StatusOK)
}

/* SetSetupState marks setup as complete */
func (h *FactoryHandlers) SetSetupState(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	var req struct {
		Completed bool `json:"completed"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid request body"), nil)
		return
	}

	if err := h.queries.UpsertSetting(ctx, "setup_complete", map[string]interface{}{
		"completed": req.Completed,
	}); err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	WriteSuccess(w, map[string]interface{}{
		"setup_complete": req.Completed,
	}, http.StatusOK)
}

/* GetFactoryStatus returns the factory status */
func (h *FactoryHandlers) GetFactoryStatus(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	status := FactoryStatusResponse{}

	/* Detect OS */
	osInfo := detectOS()
	status.OS = osInfo

	/* Check Docker */
	dockerInfo := checkDocker()
	status.Docker = dockerInfo

	/* Check NeuronDB */
	neurondbStatus := h.checkNeuronDB(ctx)
	status.NeuronDB = neurondbStatus

	/* Check NeuronAgent */
	agentStatus := h.checkNeuronAgent(ctx)
	status.NeuronAgent = agentStatus

	/* Check NeuronMCP */
	mcpStatus := h.checkNeuronMCP(ctx)
	status.NeuronMCP = mcpStatus

	/* Generate install commands */
	status.InstallCommands = generateInstallCommands(osInfo, dockerInfo)

	WriteSuccess(w, status, http.StatusOK)
}

/* detectOS detects the operating system */
func detectOS() OSInfo {
	info := OSInfo{
		Type: runtime.GOOS,
		Arch: runtime.GOARCH,
	}

	switch runtime.GOOS {
	case "linux":
		/* Try to detect distro */
		if data, err := os.ReadFile("/etc/os-release"); err == nil {
			lines := strings.Split(string(data), "\n")
			for _, line := range lines {
				if strings.HasPrefix(line, "ID=") {
					info.Distro = strings.Trim(strings.TrimPrefix(line, "ID="), "\"")
				} else if strings.HasPrefix(line, "VERSION_ID=") {
					info.Version = strings.Trim(strings.TrimPrefix(line, "VERSION_ID="), "\"")
				}
			}
		} else if data, err := os.ReadFile("/etc/redhat-release"); err == nil {
			info.Distro = "rhel"
			info.Version = strings.TrimSpace(string(data))
		}
	case "darwin":
		info.Distro = "macos"
		out, err := exec.Command("sw_vers", "-productVersion").Output()
		if err == nil {
			info.Version = strings.TrimSpace(string(out))
		}
	case "windows":
		info.Distro = "windows"
	}

	return info
}

/* checkDocker checks if Docker is available */
func checkDocker() DockerInfo {
	info := DockerInfo{Available: false}

	cmd := exec.Command("docker", "version", "--format", "{{.Server.Version}}")
	out, err := cmd.Output()
	if err == nil {
		info.Available = true
		info.Version = strings.TrimSpace(string(out))
	}

	return info
}

/* checkNeuronDB checks NeuronDB installation and connectivity */
func (h *FactoryHandlers) checkNeuronDB(ctx context.Context) ComponentStatus {
	status := ComponentStatus{
		Status:  "missing",
		Details: make(map[string]interface{}),
	}

	/* Try to get default profile to check DSN */
	profile, err := h.queries.GetDefaultProfile(ctx)
	if err != nil || profile == nil {
		status.ErrorMessage = "No default profile configured"
		status.Details["recommendation"] = "Create a profile with NeuronDB DSN in Settings"
		return status
	}

	dsn := profile.NeuronDBDSN
	if dsn == "" {
		status.ErrorMessage = "NeuronDB DSN not configured"
		return status
	}

	status.Details["dsn"] = maskDSN(dsn)

	/* Try to connect */
	db, err := sql.Open("pgx", dsn)
	if err != nil {
		status.ErrorMessage = fmt.Sprintf("Failed to open connection: %v", err)
		return status
	}
	defer db.Close()

	testCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	if err := db.PingContext(testCtx); err != nil {
		status.ErrorMessage = fmt.Sprintf("Failed to connect: %v", err)
		return status
	}

	status.Installed = true
	status.Reachable = true
	status.Details["connected"] = true

	/* Check PostgreSQL version */
	var pgVersion string
	if err := db.QueryRowContext(testCtx, "SELECT version()").Scan(&pgVersion); err == nil {
		status.Details["postgres_version"] = pgVersion
	}

	/* Check NeuronDB extension */
	var extVersion string
	if err := db.QueryRowContext(testCtx, "SELECT neurondb.version()").Scan(&extVersion); err == nil {
		status.Installed = true
		status.Running = true
		status.Status = "running"
		status.Details["extension_version"] = extVersion
		status.Details["extension_installed"] = true
	} else {
		status.Status = "reachable"
		status.ErrorMessage = "NeuronDB extension not installed. Run: CREATE EXTENSION neurondb;"
		status.Details["extension_installed"] = false
	}

	return status
}

/* checkNeuronAgent checks NeuronAgent installation and connectivity */
func (h *FactoryHandlers) checkNeuronAgent(ctx context.Context) ComponentStatus {
	status := ComponentStatus{
		Status:  "missing",
		Details: make(map[string]interface{}),
	}

	/* Try to get default profile */
	profile, err := h.queries.GetDefaultProfile(ctx)
	if err != nil || profile == nil {
		status.ErrorMessage = "No default profile configured"
		return status
	}

	endpoint := profile.AgentEndpoint
	if endpoint == "" {
		status.ErrorMessage = "NeuronAgent endpoint not configured"
		status.Details["recommendation"] = "Configure agent_endpoint in profile settings"
		return status
	}

	status.Details["endpoint"] = endpoint

	/* Try to reach /health endpoint */
	client := &http.Client{Timeout: 5 * time.Second}
	healthURL := strings.TrimSuffix(endpoint, "/") + "/health"

	req, err := http.NewRequestWithContext(ctx, "GET", healthURL, nil)
	if err != nil {
		status.ErrorMessage = fmt.Sprintf("Failed to create request: %v", err)
		return status
	}

	if profile.AgentAPIKey != "" {
		req.Header.Set("Authorization", "Bearer "+profile.AgentAPIKey)
	}

	resp, err := client.Do(req)
	if err != nil {
		status.ErrorMessage = fmt.Sprintf("Failed to connect: %v", err)
		return status
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		status.Installed = true
		status.Running = true
		status.Reachable = true
		status.Status = "running"
		status.Details["health_check"] = "passed"
	} else {
		status.ErrorMessage = fmt.Sprintf("Health check failed with status %d", resp.StatusCode)
		status.Status = "error"
	}

	return status
}

/* checkNeuronMCP checks if NeuronMCP binary is available and can connect */
func (h *FactoryHandlers) checkNeuronMCP(ctx context.Context) ComponentStatus {
	status := ComponentStatus{
		Status:  "missing",
		Details: make(map[string]interface{}),
	}

	/* Use existing detection utility */
	binaryPath := utils.FindNeuronMCPBinary()
	if binaryPath == "" {
		status.ErrorMessage = "NeuronMCP binary not found in PATH or common locations"
		status.Details["searched_locations"] = []string{
			"/usr/bin/neurondb-mcp",
			"/usr/local/bin/neurondb-mcp",
			"$HOME/.local/bin/neurondb-mcp",
		}
		return status
	}

	status.Installed = true
	status.Details["binary_path"] = binaryPath

	/* Verify binary is executable by checking file permissions */
	info, err := os.Stat(binaryPath)
	if err != nil {
		status.ErrorMessage = fmt.Sprintf("Cannot access binary: %v", err)
		status.Status = "error"
		return status
	}

	mode := info.Mode()
	if !mode.IsRegular() {
		status.ErrorMessage = "Binary path is not a regular file"
		status.Status = "error"
		return status
	}

	if mode&0111 == 0 {
		status.ErrorMessage = "Binary is not executable"
		status.Status = "error"
		return status
	}

	status.Details["executable"] = true
	status.Status = "installed"

	/* Try to get profile to check database connectivity */
	/* MCP needs database access to function, so we test connectivity */
	profile, err := h.queries.GetDefaultProfile(ctx)
	if err == nil && profile != nil {
		dsn := profile.NeuronDBDSN
		if dsn != "" {
			/* Test database connection using the same DSN that MCP would use */
			db, err := sql.Open("pgx", dsn)
			if err == nil {
				testDbCtx, cancelDb := context.WithTimeout(ctx, 3*time.Second)
				defer cancelDb()

				if err := db.PingContext(testDbCtx); err == nil {
					status.Reachable = true
					status.Details["database_reachable"] = true

					/* If binary is installed and database is reachable, mark as ready */
					if status.Status == "installed" {
						status.Status = "reachable"
						status.Details["ready"] = true
					}
				} else {
					status.Details["database_reachable"] = false
					if status.ErrorMessage == "" {
						status.ErrorMessage = fmt.Sprintf("Database connection failed: %v. MCP requires database access to function.", err)
					}
				}
				db.Close()
			} else {
				status.Details["database_reachable"] = false
				if status.ErrorMessage == "" {
					status.ErrorMessage = fmt.Sprintf("Cannot open database connection: %v", err)
				}
			}
		} else {
			status.Details["database_reachable"] = false
			if status.ErrorMessage == "" {
				status.ErrorMessage = "NeuronDB DSN not configured. MCP requires database access."
			}
		}
	} else {
		status.Details["database_reachable"] = false
		if status.ErrorMessage == "" {
			status.ErrorMessage = "No default profile configured. MCP requires database access."
		}
	}

	return status
}

/* generateInstallCommands generates OS-specific install commands */
func generateInstallCommands(osInfo OSInfo, dockerInfo DockerInfo) InstallCommands {
	cmds := InstallCommands{}

	/* Docker commands (always available if Docker is installed) */
	if dockerInfo.Available {
		cmds.Docker = []string{
			"# Install NeuronDB (CPU)",
			"./scripts/neurondb-docker.sh run --component neurondb --variant cpu",
			"",
			"# Install NeuronAgent",
			"./scripts/neurondb-docker.sh run --component neuronagent",
			"",
			"# Install NeuronMCP",
			"./scripts/neurondb-docker.sh run --component neuronmcp",
		}
	}

	/* OS-specific package commands */
	switch osInfo.Distro {
	case "ubuntu", "debian":
		cmds.Deb = []string{
			"# Download and install NeuronDB extension",
			"sudo dpkg -i neurondb_*.deb",
			"",
			"# Download and install NeuronAgent",
			"sudo dpkg -i neuronagent_*.deb",
			"sudo systemctl start neuronagent",
			"sudo systemctl enable neuronagent",
			"",
			"# Download and install NeuronMCP",
			"sudo dpkg -i neuronmcp_*.deb",
		}
	case "rhel", "rocky", "centos", "fedora":
		cmds.Rpm = []string{
			"# Download and install NeuronDB extension",
			"sudo rpm -ivh neurondb-*.rpm",
			"",
			"# Download and install NeuronAgent",
			"sudo rpm -ivh neuronagent-*.rpm",
			"sudo systemctl start neuronagent",
			"sudo systemctl enable neuronagent",
			"",
			"# Download and install NeuronMCP",
			"sudo rpm -ivh neuronmcp-*.rpm",
		}
	case "macos":
		cmds.MacPkg = []string{
			"# Install NeuronDB extension",
			"sudo installer -pkg neurondb-*.pkg -target /",
			"",
			"# Install NeuronAgent",
			"sudo installer -pkg neuronagent-*.pkg -target /",
			"sudo launchctl load /Library/LaunchDaemons/com.neurondb.neuronagent.plist",
			"sudo launchctl start com.neurondb.neuronagent",
			"",
			"# Install NeuronMCP",
			"sudo installer -pkg neuronmcp-*.pkg -target /",
		}
	}

	return cmds
}

/* maskDSN masks sensitive parts of DSN */
func maskDSN(dsn string) string {
	/* Simple masking: hide password if present */
	if strings.Contains(dsn, "@") {
		parts := strings.Split(dsn, "@")
		if len(parts) == 2 {
			auth := parts[0]
			if strings.Contains(auth, ":") {
				userParts := strings.Split(auth, ":")
				if len(userParts) >= 2 {
					return userParts[0] + ":****@" + parts[1]
				}
			}
		}
	}
	return dsn
}
