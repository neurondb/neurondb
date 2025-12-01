# NeuronDB GPU Tools

Complete standalone GPU monitoring and diagnostics utility for NVIDIA, AMD, and Apple Silicon.

## Overview

`gpu.py` - Comprehensive GPU monitoring tool with no dependencies on PostgreSQL or databases.

### Supported Platforms

**Operating Systems:**
- ✅ **macOS** (Intel & Apple Silicon)
- ✅ **Rocky Linux** / RHEL / CentOS
- ✅ **Ubuntu** / Debian

**GPU Types:**
- ✅ **NVIDIA** (CUDA) - Full monitoring (all platforms)
- ✅ **AMD** (ROCm) - Basic monitoring (Linux only)
- ✅ **Apple Silicon** (Metal) - Integrated GPU (macOS only)
- ✅ **ARM Mali** - Basic detection (ARM Linux)

**Architectures:**
- ✅ **x86_64** / AMD64
- ✅ **ARM64** / AArch64

## Quick Start

```bash
# Make executable
chmod +x gpu.py

# Auto-detect platform and list GPUs
./gpu.py --list

# Show detailed system and GPU info
./gpu.py --info

# Real-time monitoring (like top)
./gpu.py --monitor

# Press Ctrl+C to exit monitoring
```

### Auto-Detection

The tool **automatically detects**:
- Your OS (macOS, Rocky Linux, Ubuntu, etc.)
- Your architecture (x86_64, ARM64)
- Your GPU type (NVIDIA, AMD, Apple Silicon, ARM Mali)
- Available backends (CUDA, ROCm, Metal)

**Example output:**
```
Platform: macOS (Darwin) | Arch: arm64
GPU 0: Apple Silicon GPU | Type: Apple (Metal)
```

**On Rocky Linux with NVIDIA:**
```
Platform: rocky (Linux) | Arch: x86_64
GPU 0: NVIDIA RTX 4090 | Type: NVIDIA (CUDA)
```

## Commands

### `--list`
List all detected GPUs in clean tabular format

```bash
./gpu.py --list
```

**Output:**
```
┌────┬─────────────────────┬──────┬─────────┬────────┬─────────┐
│ ID │ Name                │ Type │ Backend │ Memory │ Compute │
├────┼─────────────────────┼──────┼─────────┼────────┼─────────┤
│ 0  │ NVIDIA RTX 4090     │ NVIDIA│ CUDA   │ 24.0GB │ 8.9     │
│ 1  │ NVIDIA RTX 4090     │ NVIDIA│ CUDA   │ 24.0GB │ 8.9     │
└────┴─────────────────────┴──────┴─────────┴────────┴─────────┘
```

### `--info`
Show detailed GPU information

```bash
./gpu.py --info
```

### `--monitor`
Real-time GPU monitoring (like `top` for GPUs)

```bash
# Monitor with default 2s refresh
./gpu.py --monitor

# Monitor with 1s refresh
./gpu.py --monitor -i 1

# Monitor with verbose process listing
./gpu.py --monitor -v
```

**Press Ctrl+C to exit**

### `--stats`
Show current GPU statistics

```bash
# Basic stats
./gpu.py --stats

# Stats with processes
./gpu.py --stats -v
```

### `--memory`
Show GPU memory usage (NVIDIA only)

```bash
./gpu.py --memory
```

### `--processes`
List all GPU processes (NVIDIA only)

```bash
./gpu.py --processes
```

### `--watch`
Watch specific GPU metric in real-time

```bash
# Watch GPU utilization
./gpu.py --watch --metric gpu

# Watch memory utilization
./gpu.py --watch --metric memory

# Watch temperature
./gpu.py --watch --metric temp

# Watch power consumption
./gpu.py --watch --metric power

# Watch on specific GPU with custom interval
./gpu.py --watch --gpu-id 1 --metric temp -i 1
```

### `--diagnose`
Run comprehensive GPU diagnostics

```bash
# Run diagnostics
./gpu.py --diagnose

# Save to file
./gpu.py --diagnose -o gpu_diag.json

# Verbose diagnostics
./gpu.py --diagnose -v
```

### `--topology`
Show GPU topology and interconnects (NVIDIA only)

```bash
./gpu.py --topology
```

### `--version`
Show GPU driver and toolkit versions

```bash
./gpu.py --version
```

## Options

### Global Options

- `-i, --interval <seconds>` - Refresh interval for monitor/watch (default: 2)
- `-v, --verbose` - Verbose output (use -v, -vv, or -vvv for increasing verbosity)
- `--gpu-id <id>` - Specify GPU ID for watch operations (default: 0)
- `-o, --output <file>` - Output file for JSON export
- `--json` - Output in JSON format

### Verbose Levels

- Default (no `-v`) - Clean, minimal output
- `-v` - Include additional details (processes, extended metrics)
- `-vv` - Very verbose with all available information
- `-vvv` - Debug level output

## Examples

### Basic Monitoring Workflow

```bash
# 1. Check what GPUs are available
./gpu.py --list

# 2. Get detailed information
./gpu.py --info

# 3. Check current status
./gpu.py --stats

# 4. Start real-time monitoring
./gpu.py --monitor
```

### Advanced Usage

```bash
# Monitor with verbose output and fast refresh
./gpu.py --monitor -i 1 -v

# Watch GPU 0 temperature every second
./gpu.py --watch --gpu-id 0 --metric temp -i 1

# Export diagnostics for support
./gpu.py --diagnose -o gpu_report.json

# Check memory usage across all GPUs
./gpu.py --memory

# Monitor GPU processes
./gpu.py --processes -v
```

### Automation & Scripting

```bash
# Export current stats to JSON
./gpu.py --stats --json > gpu_stats.json

# Check if GPU utilization is high
./gpu.py --stats | grep "80" && echo "High GPU usage!"

# Log GPU stats every minute
while true; do
    ./gpu.py --stats >> gpu_log.txt
    sleep 60
done
```

## Platform-Specific Features

### NVIDIA (CUDA)
✅ Full monitoring support  
✅ Process listing  
✅ Memory details  
✅ Temperature and power  
✅ Clock speeds  
✅ Topology mapping  
✅ Watch mode  

### AMD (ROCm)
✅ Basic monitoring  
✅ Device detection  
⚠ Limited metrics (ROCm API constraints)  

### Apple Silicon (Metal)
✅ Device detection  
✅ Chip information  
✅ Overall GPU utilization (requires sudo)  
⚠ Per-process GPU usage not available (macOS limitation)  
Shows GPU-capable processes when GPU is active  

## Requirements

### Minimum Requirements
- Python 3.6+
- `numpy` and `psutil` (install via requirements.txt)

### Optional Requirements for Full Features

**System Tools:**
- **NVIDIA:** `nvidia-smi` (included with NVIDIA drivers), `nvcc` (CUDA Toolkit)
- **AMD:** `rocm-smi` (included with ROCm)
- **Apple Silicon:** macOS with Apple Silicon, `powermetrics` (requires sudo)

**Python Packages (for GPU stress testing):**
- **NVIDIA:** `cupy-cuda12x` (for CUDA 12.x) or `cupy-cuda11x` (for CUDA 11.x)
- **Apple Silicon:** `torch` (PyTorch with MPS support)
- **AMD:** `cupy-rocm-5-0` (for ROCm 5.0)

## Installation

### Basic Installation

```bash
# Clone or download
cd Neurondb/tools

# Install core dependencies
pip3 install -r requirements.txt

# Make executable
chmod +x gpu.py

# Test installation
./gpu.py --list
```

### With GPU Stress Testing Support

```bash
# Install core dependencies
pip3 install -r requirements.txt

# For NVIDIA GPU stress testing (CUDA 12.x)
pip3 install cupy-cuda12x

# For Apple Silicon GPU stress testing
pip3 install torch

# For AMD GPU stress testing (ROCm 5.0)
pip3 install cupy-rocm-5-0

# Test GPU stress
./gpu.py --use-gpu --intensity 50
```

### System-Wide Installation

```bash
# Optionally add to PATH
sudo ln -s $(pwd)/gpu.py /usr/local/bin/gpu

# Use system-wide
gpu --list
```

## Troubleshooting

### No GPUs Detected

```bash
# Check GPU drivers
./gpu.py --diagnose

# For NVIDIA, verify nvidia-smi works
nvidia-smi

# For AMD, verify rocm-smi works
rocm-smi

# For Apple Silicon, verify architecture
sysctl machdep.cpu.brand_string
```

### Command Not Found

```bash
# Make sure it's executable
chmod +x gpu.py

# Run with python3 explicitly
python3 gpu.py --list
```

### Permission Denied (Apple Silicon detailed stats)

```bash
# Some metrics require sudo
sudo ./gpu.py --monitor
```

## Integration

### Prometheus/Grafana

```bash
# Export metrics periodically
*/1 * * * * /path/to/gpu.py --stats --json > /var/lib/prometheus/gpu_metrics.json
```

### Alerting

```bash
# Check GPU temperature
TEMP=$(./gpu.py --stats | grep "Temp" | awk '{print $3}')
if [ $TEMP -gt 80 ]; then
    send_alert "High GPU temperature: ${TEMP}°C"
fi
```

### Logging

```bash
# Continuous logging
./gpu.py --monitor -i 5 >> /var/log/gpu_monitor.log 2>&1
```

## Performance Tips

- Use `--interval 5` or higher for less resource usage
- Default verbosity (no `-v`) is optimized for clean output
- `--monitor` mode is efficient and can run continuously
- `--watch` is ideal for tracking single metrics

## Support

- Documentation: https://neurondb.ai/docs
- Issues: https://github.com/neurondb/neurondb/issues
- NeuronDB Tools: Part of the neurondb suite

## License

MIT License - See LICENSE file

---

**Built by neurondb** - Making PostgreSQL AI better
