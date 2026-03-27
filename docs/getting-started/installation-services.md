# Service Management Guide

NeuronDB runs **inside** PostgreSQL. Manage it by managing the PostgreSQL server.

## Running PostgreSQL (and NeuronDB)

- **Linux (systemd):**  
  `sudo systemctl start postgresql` · `sudo systemctl enable postgresql`
- **macOS (Homebrew):**  
  `brew services start postgresql@17`

No separate NeuronDB service file is required. The extension is loaded via `shared_preload_libraries = 'neurondb'` in `postgresql.conf`.

## See also

- [Native installation](installation-native.md) — Building and installing the extension
- [Installation overview](installation.md) — All installation options
