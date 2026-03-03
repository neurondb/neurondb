#!/usr/bin/env python3
"""
NeuronDB GPU Monitoring Tool
Complete standalone GPU utility for NVIDIA, AMD, and Apple Silicon
No database dependencies required
"""

import argparse
import sys
import json
import time
import os
import subprocess
import platform
import signal
from datetime import datetime
from typing import Dict, List, Optional


class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    NC = '\033[0m'


class Table:
    """Clean tabular output formatter"""
    
    @staticmethod
    def print_row(columns: List[str], widths: List[int], separator='│'):
        """Print a formatted table row"""
        parts = [f" {col.ljust(width)} " for col, width in zip(columns, widths)]
        print(f"{separator}{separator.join(parts)}{separator}")
    
    @staticmethod
    def print_separator(widths: List[int], left='├', mid='┼', right='┤', line='─'):
        """Print table separator"""
        parts = [line * (width + 2) for width in widths]
        print(f"{left}{mid.join(parts)}{right}")
    
    @staticmethod
    def print_header(widths: List[int], left='┌', mid='┬', right='┐', line='─'):
        """Print table header border"""
        parts = [line * (width + 2) for width in widths]
        print(f"{left}{mid.join(parts)}{right}")
    
    @staticmethod
    def print_footer(widths: List[int], left='└', mid='┴', right='┘', line='─'):
        """Print table footer border"""
        parts = [line * (width + 2) for width in widths]
        print(f"{left}{mid.join(parts)}{right}")


class GPUDetector:
    """Detect GPU hardware"""
    
    @staticmethod
    def detect_nvidia() -> Optional[List[Dict]]:
        """Detect NVIDIA GPUs"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.total,compute_cap,driver_version,pci.bus_id',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = [x.strip() for x in line.split(',')]
                        gpus.append({
                            'id': int(parts[0]),
                            'name': parts[1],
                            'memory_mb': int(parts[2]),
                            'compute_cap': parts[3],
                            'driver': parts[4],
                            'pci': parts[5],
                            'type': 'NVIDIA',
                            'backend': 'CUDA'
                        })
                return gpus
        except:
            pass
        return None
    
    @staticmethod
    def detect_amd() -> Optional[List[Dict]]:
        """Detect AMD GPUs"""
        try:
            result = subprocess.run(['rocm-smi', '--showid'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return [{'id': 0, 'name': 'AMD GPU', 'type': 'AMD', 'backend': 'ROCm'}]
        except:
            pass
        return None
    
    @staticmethod
    def detect_metal() -> Optional[List[Dict]]:
        """Detect Apple Silicon"""
        try:
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                   capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                chip = result.stdout.strip()
                if 'Apple' in chip or platform.processor() == 'arm':
                    return [{'id': 0, 'name': 'Apple Silicon GPU', 'chip': chip, 'type': 'Apple', 'backend': 'Metal'}]
        except:
            pass
        return None
    
    @staticmethod
    def get_os_info() -> Dict:
        """Detect OS and architecture"""
        os_name = platform.system()  # Darwin, Linux, Windows
        os_release = platform.release()
        machine = platform.machine()  # x86_64, arm64, aarch64
        
        # Detect Linux distribution
        distro = 'Unknown'
        if os_name == 'Linux':
            try:
                with open('/etc/os-release', 'r') as f:
                    for line in f:
                        if line.startswith('ID='):
                            distro = line.split('=')[1].strip().strip('"')
                            break
            except:
                # Fallback for older systems
                try:
                    if os.path.exists('/etc/redhat-release'):
                        with open('/etc/redhat-release', 'r') as f:
                            content = f.read().lower()
                            if 'rocky' in content:
                                distro = 'rocky'
                            elif 'centos' in content:
                                distro = 'centos'
                            elif 'rhel' in content:
                                distro = 'rhel'
                    elif os.path.exists('/etc/lsb-release'):
                        distro = 'ubuntu'
                except:
                    pass
        
        return {
            'os': os_name,
            'distro': distro if os_name == 'Linux' else os_name,
            'release': os_release,
            'arch': machine,
            'is_arm': machine in ['arm64', 'aarch64', 'armv7l', 'armv8'],
            'is_x86': machine in ['x86_64', 'amd64', 'AMD64'],
            'is_macos': os_name == 'Darwin',
            'is_linux': os_name == 'Linux',
            'is_rocky': distro in ['rocky', 'rhel', 'centos'],
            'is_ubuntu': distro in ['ubuntu', 'debian', 'pop']
        }
    
    @classmethod
    def detect_all(cls) -> List[Dict]:
        """Detect all GPUs with OS and architecture info"""
        os_info = cls.get_os_info()
        
        # Try NVIDIA first (works on all platforms)
        nvidia = cls.detect_nvidia()
        if nvidia:
            for gpu in nvidia:
                gpu.update({
                    'os': os_info['distro'],
                    'os_arch': os_info['arch'],
                    'platform': f"{os_info['distro']}/{os_info['arch']}"
                })
            return nvidia
        
        # Try AMD (Linux only)
        if os_info['is_linux']:
            amd = cls.detect_amd()
            if amd:
                for gpu in amd:
                    gpu.update({
                        'os': os_info['distro'],
                        'os_arch': os_info['arch'],
                        'platform': f"{os_info['distro']}/{os_info['arch']}"
                    })
                return amd
        
        # Try Apple Silicon (macOS only)
        if os_info['is_macos']:
            metal = cls.detect_metal()
            if metal:
                for gpu in metal:
                    gpu.update({
                        'os': 'macOS',
                        'os_arch': os_info['arch'],
                        'platform': f"macOS/{os_info['arch']}"
                    })
                return metal
        
        # ARM GPU on Linux (Mali, etc.)
        if os_info['is_linux'] and os_info['is_arm']:
            try:
                if os.path.exists('/dev/mali0'):
                    return [{
                        'id': 0,
                        'name': 'ARM Mali GPU',
                        'type': 'ARM',
                        'backend': 'Mali',
                        'os': os_info['distro'],
                        'os_arch': os_info['arch'],
                        'platform': f"{os_info['distro']}/{os_info['arch']}"
                    }]
            except:
                pass
        
        return []


class NVIDIAStats:
    """NVIDIA GPU statistics"""
    
    @staticmethod
    def get_stats() -> List[Dict]:
        """Get current statistics"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,utilization.memory,'
                 'memory.used,memory.free,memory.total,temperature.gpu,fan.speed,'
                 'power.draw,power.limit,clocks.current.graphics,clocks.current.memory',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            
            stats = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    p = [x.strip() for x in line.split(',')]
                    stats.append({
                        'id': int(p[0]),
                        'name': p[1],
                        'util_gpu': int(p[2]) if p[2] != '[N/A]' else 0,
                        'util_mem': int(p[3]) if p[3] != '[N/A]' else 0,
                        'mem_used': int(p[4]) if p[4] != '[N/A]' else 0,
                        'mem_free': int(p[5]) if p[5] != '[N/A]' else 0,
                        'mem_total': int(p[6]) if p[6] != '[N/A]' else 0,
                        'temp': int(p[7]) if p[7] != '[N/A]' else 0,
                        'fan': int(p[8]) if p[8] != '[N/A]' else 0,
                        'power': float(p[9]) if p[9] != '[N/A]' else 0,
                        'power_max': float(p[10]) if p[10] != '[N/A]' else 0,
                        'clock_gpu': int(p[11]) if p[11] != '[N/A]' else 0,
                        'clock_mem': int(p[12]) if p[12] != '[N/A]' else 0
                    })
            return stats
        except:
            return []
    
    @staticmethod
    def get_processes() -> List[Dict]:
        """Get GPU processes with detailed usage"""
        try:
            # Get process info
            result = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            
            processes = []
            for line in result.stdout.strip().split('\n'):
                if line and line.strip():
                    parts = [x.strip() for x in line.split(',')]
                    if len(parts) >= 3:
                        processes.append({
                            'pid': int(parts[0]),
                            'name': parts[1],
                            'memory_mb': int(parts[2])
                        })
            
            # Get per-process GPU utilization using pmon
            try:
                pmon_result = subprocess.run(
                    ['nvidia-smi', 'pmon', '-c', '1'],
                    capture_output=True, text=True, timeout=5
                )
                
                # Parse pmon output for GPU utilization per process
                pmon_data = {}
                for line in pmon_result.stdout.split('\n'):
                    if line.strip() and not line.startswith('#') and line.strip() != '':
                        parts = line.split()
                        if len(parts) >= 7 and parts[0].isdigit():
                            try:
                                gpu_id = int(parts[0])
                                pid = int(parts[1])
                                sm_util = parts[3] if parts[3] != '-' else '0'
                                mem_util = parts[4] if parts[4] != '-' else '0'
                                enc_util = parts[5] if parts[5] != '-' else '0'
                                dec_util = parts[6] if parts[6] != '-' else '0'
                                
                                pmon_data[pid] = {
                                    'gpu_id': gpu_id,
                                    'sm_util': int(sm_util),
                                    'mem_util': int(mem_util),
                                    'enc_util': int(enc_util),
                                    'dec_util': int(dec_util)
                                }
                            except (ValueError, IndexError):
                                pass
                
                # Merge pmon data with process data
                for proc in processes:
                    if proc['pid'] in pmon_data:
                        proc.update(pmon_data[proc['pid']])
                    else:
                        proc['sm_util'] = 0
                        proc['mem_util'] = 0
                        proc['gpu_id'] = 0
            except:
                # If pmon fails, set defaults
                for proc in processes:
                    proc['sm_util'] = 0
                    proc['mem_util'] = 0
                    proc['gpu_id'] = 0
            
            return processes
        except:
            return []


class AMDStats:
    """AMD GPU statistics"""
    
    @staticmethod
    def get_stats() -> List[Dict]:
        """Get current statistics"""
        try:
            result = subprocess.run(['rocm-smi', '--showuse'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return [{'id': 0, 'name': 'AMD GPU', 'util_gpu': 0}]
        except:
            pass
        return []


class MetalStats:
    """Apple Silicon statistics"""
    
    @staticmethod
    def get_gpu_stats_sudo() -> Dict:
        """Get GPU stats using powermetrics (requires sudo)"""
        try:
            result = subprocess.run(
                ['sudo', '-n', 'powermetrics', '--samplers', 'gpu_power', '-i', '500', '-n', '1'],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                output = result.stdout
                gpu_util = 0
                gpu_active = 0
                gpu_idle = 0
                
                for line in output.split('\n'):
                    if 'GPU active residency' in line or 'GPU Active residency' in line:
                        # Extract percentage: "GPU active residency: 12.34%"
                        parts = line.split(':')
                        if len(parts) > 1:
                            pct = parts[1].strip().replace('%', '')
                            try:
                                gpu_active = float(pct)
                                gpu_util = int(gpu_active)
                            except:
                                pass
                    elif 'GPU idle residency' in line or 'GPU Idle residency' in line:
                        parts = line.split(':')
                        if len(parts) > 1:
                            pct = parts[1].strip().replace('%', '')
                            try:
                                gpu_idle = float(pct)
                            except:
                                pass
                
                return {
                    'has_metrics': True,
                    'gpu_util': gpu_util,
                    'gpu_active': gpu_active,
                    'gpu_idle': gpu_idle
                }
        except:
            pass
        return {'has_metrics': False}
    
    @staticmethod
    def get_stats() -> List[Dict]:
        """Get current statistics"""
        stats = []
        try:
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                   capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                chip = result.stdout.strip()
                
                # Try to get GPU metrics with sudo (passwordless)
                gpu_metrics = MetalStats.get_gpu_stats_sudo()
                
                stats.append({
                    'id': 0,
                    'name': 'Apple Silicon GPU',
                    'chip': chip,
                    'util_gpu': gpu_metrics.get('gpu_util', 0),
                    'gpu_active': gpu_metrics.get('gpu_active', 0),
                    'gpu_idle': gpu_metrics.get('gpu_idle', 0),
                    'has_metrics': gpu_metrics.get('has_metrics', False),
                    'requires_sudo': not gpu_metrics.get('has_metrics', False)
                })
        except:
            pass
        return stats
    
    @staticmethod
    def get_processes() -> List[Dict]:
        """Get GPU-related processes with actual GPU usage"""
        processes = []
        
        # Note: Apple Silicon doesn't provide easy per-process GPU utilization
        # We show GPU-capable processes instead
        try:
            # Get list of all processes
            ps_result = subprocess.run(
                ['ps', 'aux'],
                capture_output=True, text=True, timeout=5
            )
            
            if ps_result.returncode != 0:
                return []
            
            # Filter for processes that are likely using Metal/GPU
            # On Apple Silicon, WindowServer, Chrome, Safari, Python (with ML libs) use GPU
            gpu_keywords = [
                'windowserver', 'chrome', 'safari', 'firefox', 'python',
                'torch', 'tensorflow', 'metal', 'gpu', 'neurondb',
                'blender', 'unity', 'unreal', 'cuda', 'opencl',
                'final cut', 'davinci', 'premiere', 'photoshop',
                'pytorch', 'keras', 'mps'
            ]
            
            for line in ps_result.stdout.split('\n')[1:]:
                parts = line.split()
                if len(parts) >= 11:
                    try:
                        cpu = float(parts[2])
                        mem = float(parts[3])
                        pid = int(parts[1])
                        cmd = ' '.join(parts[10:]).lower()
                        
                        # Check if process matches GPU keywords
                        if any(keyword in cmd for keyword in gpu_keywords):
                            # Get clean process name
                            cmd_original = ' '.join(parts[10:])
                            if '/' in cmd_original:
                                name = cmd_original.split('/')[-1].split()[0]
                            else:
                                name = parts[10]
                            
                            # Use CPU as proxy for GPU activity (higher CPU often means GPU use)
                            processes.append({
                                'pid': pid,
                                'name': name[:30],
                                'gpu_util': 0,  # Not available per-process on macOS
                                'cpu_util': cpu,
                                'mem_util': mem,
                                'cmd': cmd_original
                            })
                    except:
                        pass
            
            # Sort by CPU as proxy for GPU activity
            processes.sort(key=lambda x: x.get('cpu_util', 0), reverse=True)
            
        except:
            pass
        
        return processes


class GPUStress:
    """GPU stress testing"""
    
    @staticmethod
    def stress_nvidia(gpu_id: int = 0, intensity: int = 100):
        """Stress NVIDIA GPU"""
        try:
            import numpy as np
        except ImportError:
            print(f"{Colors.RED}numpy required: pip install numpy{Colors.NC}")
            return
        
        print(f"\n{Colors.YELLOW}Stressing GPU {gpu_id} at {intensity}% intensity{Colors.NC}")
        print(f"{Colors.CYAN}Press Ctrl+C to stop{Colors.NC}\n")
        
        running = True
        
        def signal_handler(sig, frame):
            nonlocal running
            running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        
        iteration = 0
        start_time = time.time()
        
        try:
            # Try CuPy for real GPU computation
            try:
                import cupy as cp
                has_cupy = True
                print(f"{Colors.GREEN}Using CuPy for GPU computation{Colors.NC}\n")
            except ImportError:
                has_cupy = False
                print(f"{Colors.YELLOW}CuPy not available - using numpy (CPU){Colors.NC}")
                print(f"{Colors.DIM}Install: pip install cupy-cuda11x or cupy-cuda12x{Colors.NC}\n")
            
            while running:
                iteration += 1
                
                if has_cupy:
                    size = 10000 if intensity >= 100 else 5000 if intensity >= 50 else 2000
                    a = cp.random.randn(size, size, dtype=cp.float32)
                    b = cp.random.randn(size, size, dtype=cp.float32)
                    c = cp.dot(a, b)
                    cp.cuda.Stream.null.synchronize()
                    del a, b, c
                else:
                    size = 1000
                    a = np.random.randn(size, size).astype(np.float32)
                    b = np.random.randn(size, size).astype(np.float32)
                    c = np.dot(a, b)
                    del a, b, c
                
                if iteration % 10 == 0:
                    elapsed = time.time() - start_time
                    stats = NVIDIAStats.get_stats()
                    if stats and len(stats) > gpu_id:
                        gpu = stats[gpu_id]
                        print(f"[{elapsed:5.0f}s] Iter {iteration:5d} | "
                              f"GPU: {gpu['util_gpu']:3d}% | "
                              f"Mem: {gpu['mem_used']:,}MB | "
                              f"Temp: {gpu['temp']:2d}°C | "
                              f"Power: {gpu['power']:.0f}W")
                
                time.sleep((100 - intensity) / 1000.0)
        
        except KeyboardInterrupt:
            pass
        
        elapsed = time.time() - start_time
        print(f"\n{Colors.GREEN}Stopped{Colors.NC} Duration: {elapsed:.1f}s | Iterations: {iteration}")
    
    @staticmethod
    def stress_metal(intensity: int = 100):
        """Stress Apple Silicon GPU"""
        try:
            import numpy as np
        except ImportError:
            print(f"{Colors.RED}numpy required: pip install numpy{Colors.NC}")
            return
        
        print(f"\n{Colors.YELLOW}Stressing Apple Silicon GPU at {intensity}% intensity{Colors.NC}")
        print(f"{Colors.CYAN}Press Ctrl+C to stop{Colors.NC}\n")
        
        running = True
        
        def signal_handler(sig, frame):
            nonlocal running
            running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        
        iteration = 0
        start_time = time.time()
        
        try:
            # Try PyTorch with Metal Performance Shaders (MPS)
            try:
                import torch
                if torch.backends.mps.is_available():
                    device = torch.device("mps")
                    has_mps = True
                    print(f"{Colors.GREEN}Using PyTorch MPS (Metal) backend{Colors.NC}\n")
                else:
                    has_mps = False
                    print(f"{Colors.YELLOW}MPS not available - using numpy (CPU){Colors.NC}")
                    print(f"{Colors.DIM}Install: pip install torch{Colors.NC}\n")
            except ImportError:
                has_mps = False
                print(f"{Colors.YELLOW}PyTorch not available - using numpy (CPU){Colors.NC}")
                print(f"{Colors.DIM}Install: pip install torch{Colors.NC}\n")
            
            while running:
                iteration += 1
                
                if has_mps:
                    # GPU computation using Metal backend
                    # Larger matrices and more operations to sustain GPU load
                    size = 8000 if intensity >= 100 else 6000 if intensity >= 75 else 4000 if intensity >= 50 else 2000
                    
                    # Batch multiple operations together for sustained GPU activity
                    # This keeps the GPU busy longer, making it visible to powermetrics
                    matrices = []
                    num_ops = 5 if intensity >= 75 else 3 if intensity >= 50 else 2
                    
                    for _ in range(num_ops):
                        a = torch.randn(size, size, dtype=torch.float32, device=device)
                        b = torch.randn(size, size, dtype=torch.float32, device=device)
                        c = torch.mm(a, b)
                        d = torch.mm(c, a)
                        e = torch.mm(d, b)
                        matrices.extend([a, b, c, d, e])
                    
                    # Synchronize to ensure all GPU operations complete
                    torch.mps.synchronize()
                    
                    # Clean up
                    del matrices
                else:
                    # CPU fallback
                    size = 1000
                    a = np.random.randn(size, size).astype(np.float32)
                    b = np.random.randn(size, size).astype(np.float32)
                    c = np.dot(a, b)
                    del a, b, c
                
                if iteration % 5 == 0:  # Report more frequently
                    elapsed = time.time() - start_time
                    ops_per_sec = iteration / elapsed
                    
                    # Get current GPU stats if available
                    gpu_info = ""
                    if has_mps:
                        stats = MetalStats.get_stats()
                        if stats and stats[0].get('has_metrics'):
                            gpu_active = stats[0].get('gpu_active', 0)
                            gpu_info = f"GPU: {gpu_active:.1f}% | "
                    
                    print(f"[{elapsed:5.0f}s] Iter {iteration:5d} | {gpu_info}"
                          f"Ops/sec: {ops_per_sec:6.1f} | "
                          f"Mode: {'GPU (MPS)' if has_mps else 'CPU'}")
                
                # No sleep for high intensity to sustain GPU load
                if intensity < 50:
                    time.sleep((100 - intensity) / 1000.0)
                # No sleep for >= 50% intensity - continuous GPU work!
        
        except KeyboardInterrupt:
            pass
        
        elapsed = time.time() - start_time
        print(f"\n{Colors.GREEN}Stopped{Colors.NC} Duration: {elapsed:.1f}s | Iterations: {iteration}")
        if has_mps:
            print(f"{Colors.GREEN}Apple Silicon GPU was actively used via Metal Performance Shaders{Colors.NC}")
    
    @staticmethod
    def stress_simple():
        """Simple CPU stress"""
        print(f"\n{Colors.YELLOW}Running CPU stress test{Colors.NC}")
        print(f"{Colors.CYAN}Press Ctrl+C to stop{Colors.NC}\n")
        
        running = True
        
        def signal_handler(sig, frame):
            nonlocal running
            running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        
        iteration = 0
        start_time = time.time()
        
        try:
            while running:
                iteration += 1
                _ = sum(i * i for i in range(1000000))
                
                if iteration % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"[{elapsed:5.0f}s] Iteration {iteration:5d} | Ops/sec: {iteration/elapsed:.0f}")
        
        except KeyboardInterrupt:
            pass
        
        elapsed = time.time() - start_time
        print(f"\n{Colors.GREEN}Stopped{Colors.NC} Duration: {elapsed:.1f}s | Iterations: {iteration}")


def cmd_list(args):
    """List all GPUs in clean table format"""
    gpus = GPUDetector.detect_all()
    
    if not gpus:
        print("No GPUs detected")
        return
    
    # Show OS info
    os_info = GPUDetector.get_os_info()
    print(f"{Colors.CYAN}Platform:{Colors.NC} {os_info['distro']} ({os_info['os']}) | {Colors.CYAN}Arch:{Colors.NC} {os_info['arch']}\n")
    
    # Table headers
    headers = ['ID', 'Name', 'Type', 'Backend', 'Memory', 'Compute']
    widths = [4, 30, 10, 8, 10, 10]
    
    # Print table
    Table.print_header(widths)
    Table.print_row(headers, widths)
    Table.print_separator(widths)
    
    for gpu in gpus:
        memory = f"{gpu.get('memory_mb', 0)/1024:.1f}GB" if gpu.get('memory_mb', 0) > 0 else 'N/A'
        compute = gpu.get('compute_cap', 'N/A')
        
        row = [
            str(gpu['id']),
            gpu['name'][:29],
            gpu['type'],
            gpu['backend'],
            memory,
            compute
        ]
        Table.print_row(row, widths)
    
    Table.print_footer(widths)
    
    if args.verbose > 0:
        print(f"\n{Colors.GREEN}Total:{Colors.NC} {len(gpus)} GPU(s)")
        print(f"{Colors.GREEN}Platform:{Colors.NC} {gpus[0].get('platform', 'Unknown')}")


def cmd_info(args):
    """Show detailed GPU information"""
    gpus = GPUDetector.detect_all()
    
    if not gpus:
        print("No GPUs detected")
        return
    
    # Show OS info
    os_info = GPUDetector.get_os_info()
    print(f"\n{Colors.CYAN}═══ System Information ═══{Colors.NC}")
    print(f"  OS: {os_info['distro']} ({os_info['os']} {os_info['release']})")
    print(f"  Architecture: {os_info['arch']}")
    if os_info['is_arm']:
        print(f"  {Colors.GREEN}ARM-based system{Colors.NC}")
    elif os_info['is_x86']:
        print(f"  {Colors.GREEN}x86_64 system{Colors.NC}")
    
    for gpu in gpus:
        print(f"\n{Colors.CYAN}═══ GPU {gpu['id']} ═══{Colors.NC}")
        print(f"{Colors.BOLD}{gpu['name']}{Colors.NC}")
        print(f"  Type: {gpu['type']} ({gpu['backend']})")
        print(f"  Platform: {gpu.get('platform', 'Unknown')}")
        
        if 'memory_mb' in gpu and gpu['memory_mb'] > 0:
            print(f"  Memory: {gpu['memory_mb']:,}MB ({gpu['memory_mb']/1024:.2f}GB)")
        if 'compute_cap' in gpu:
            print(f"  Compute: {gpu['compute_cap']}")
        if 'driver' in gpu:
            print(f"  Driver: {gpu['driver']}")
        if 'pci' in gpu:
            print(f"  PCI: {gpu['pci']}")
        if 'chip' in gpu:
            print(f"  Chip: {gpu['chip']}")


def cmd_monitor(args):
    """Real-time monitoring like top"""
    gpus = GPUDetector.detect_all()
    
    if not gpus:
        print("No GPUs detected")
        return
    
    gpu_type = gpus[0]['type']
    running = True
    
    def signal_handler(sig, frame):
        nonlocal running
        running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    
    if args.verbose == 0:
        print(f"Monitoring {len(gpus)} GPU(s) - Press Ctrl+C to quit")
        time.sleep(1)
    
    while running:
        # Clear screen
        print("\033[2J\033[H", end='', flush=True)  # Clear screen and move to top
        
        # Header
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"{Colors.CYAN}╔{'═' * 78}╗{Colors.NC}")
        print(f"{Colors.CYAN}║{'GPU Monitor'.center(78)}║{Colors.NC}")
        print(f"{Colors.CYAN}╚{'═' * 78}╝{Colors.NC}")
        print(f"Time: {timestamp} | GPUs: {len(gpus)} | Refresh: {args.interval}s\n")
        
        # Get and display stats
        if gpu_type == 'NVIDIA':
            stats = NVIDIAStats.get_stats()
            
            # Table for stats
            headers = ['ID', 'Name', 'GPU%', 'Mem%', 'Memory', 'Temp', 'Power', 'Clock']
            widths = [3, 20, 5, 5, 18, 6, 14, 12]
            
            Table.print_header(widths)
            Table.print_row(headers, widths)
            Table.print_separator(widths)
            
            for gpu in stats:
                # Color code utilization
                util = gpu['util_gpu']
                if util > 80:
                    util_str = f"{Colors.RED}{util}{Colors.NC}"
                elif util > 50:
                    util_str = f"{Colors.YELLOW}{util}{Colors.NC}"
                else:
                    util_str = f"{Colors.GREEN}{util}{Colors.NC}"
                
                mem_util = gpu['util_mem']
                mem_str = f"{mem_util}" if mem_util < 80 else f"{Colors.YELLOW}{mem_util}{Colors.NC}"
                
                memory = f"{gpu['mem_used']:,}/{gpu['mem_total']:,}MB"
                temp_str = f"{gpu['temp']}°C"
                if gpu['temp'] > 80:
                    temp_str = f"{Colors.RED}{gpu['temp']}°C{Colors.NC}"
                elif gpu['temp'] > 70:
                    temp_str = f"{Colors.YELLOW}{gpu['temp']}°C{Colors.NC}"
                
                power = f"{gpu['power']:.0f}/{gpu['power_max']:.0f}W"
                clock = f"{gpu['clock_gpu']}MHz"
                
                row = [
                    str(gpu['id']),
                    gpu['name'][:19],
                    util_str,
                    mem_str,
                    memory,
                    temp_str,
                    power,
                    clock
                ]
                Table.print_row(row, widths)
            
            Table.print_footer(widths)
            
            # Calculate total GPU usage from stats
            total_gpu_util = {gpu['id']: gpu['util_gpu'] for gpu in stats}
            
            # Always show processes in monitor mode
            processes = NVIDIAStats.get_processes()
            if processes:
                # Calculate per-GPU totals from processes
                gpu_process_totals = {}
                for proc in processes:
                    gpu_id = proc.get('gpu_id', 0)
                    proc_util = proc.get('sm_util', 0)
                    if gpu_id not in gpu_process_totals:
                        gpu_process_totals[gpu_id] = 0
                    gpu_process_totals[gpu_id] += proc_util
                
                print(f"\n{Colors.GREEN}GPU Processes ({len(processes)}):{Colors.NC}")
                
                # Compact table for monitor mode
                p_headers = ['GPU', 'PID', 'Process Name', 'GPU%', '% of Total', 'Memory']
                p_widths = [3, 7, 32, 5, 10, 12]
                
                if args.verbose > 0:
                    # Extended table with all metrics
                    p_headers = ['GPU', 'PID', 'Name', 'GPU%', 'Total%', 'Mem%', 'Memory', 'Enc', 'Dec']
                    p_widths = [3, 7, 24, 5, 6, 5, 10, 4, 4]
                
                Table.print_header(p_widths)
                Table.print_row(p_headers, p_widths)
                Table.print_separator(p_widths)
                
                total_mem = 0
                for proc in processes[:15]:
                    gpu_id = proc.get('gpu_id', 0)
                    gpu_util = proc.get('sm_util', 0)
                    mem_util = proc.get('mem_util', 0)
                    
                    # Calculate percentage of total GPU usage
                    total_util = total_gpu_util.get(gpu_id, 100)
                    pct_of_total = (gpu_util / total_util * 100) if total_util > 0 else 0
                    
                    # Color code high usage
                    gpu_str = f"{gpu_util}" if gpu_util < 80 else f"{Colors.YELLOW}{gpu_util}{Colors.NC}"
                    pct_str = f"{pct_of_total:.1f}" if pct_of_total < 80 else f"{Colors.YELLOW}{pct_of_total:.1f}{Colors.NC}"
                    
                    if args.verbose > 0:
                        mem_str = f"{mem_util}" if mem_util < 80 else f"{Colors.YELLOW}{mem_util}{Colors.NC}"
                        row = [
                            str(gpu_id),
                            str(proc['pid']),
                            proc['name'][:23],
                            gpu_str,
                            pct_str,
                            mem_str,
                            f"{proc['memory_mb']}MB",
                            str(proc.get('enc_util', 0)),
                            str(proc.get('dec_util', 0))
                        ]
                    else:
                        row = [
                            str(gpu_id),
                            str(proc['pid']),
                            proc['name'][:31],
                            gpu_str,
                            pct_str,
                            f"{proc['memory_mb']:,}MB"
                        ]
                    
                    Table.print_row(row, p_widths)
                    total_mem += proc['memory_mb']
                
                Table.print_footer(p_widths)
                
                # Summary with per-GPU breakdown
                summary_parts = []
                for gpu_id in sorted(total_gpu_util.keys()):
                    proc_count = len([p for p in processes if p.get('gpu_id', 0) == gpu_id])
                    if proc_count > 0:
                        summary_parts.append(f"GPU{gpu_id}: {total_gpu_util[gpu_id]}% ({proc_count} proc)")
                
                summary = " | ".join(summary_parts) if summary_parts else "No GPU usage"
                print(f"{summary} | Total Memory: {total_mem:,}MB")
            else:
                print(f"\n{Colors.DIM}No active GPU processes{Colors.NC}")
        
        elif gpu_type == 'AMD':
            stats = AMDStats.get_stats()
            print(f"AMD GPU monitoring - basic mode")
            for gpu in stats:
                print(f"GPU {gpu['id']}: {gpu['name']}")
        
        elif gpu_type == 'Apple':
            stats = MetalStats.get_stats()
            for gpu in stats:
                print(f"\n{Colors.BOLD}GPU {gpu['id']}: {gpu['name']}{Colors.NC}")
                if 'chip' in gpu:
                    print(f"  Chip: {gpu['chip']}")
                
                # Show GPU utilization if available
                if gpu.get('has_metrics'):
                    gpu_util = gpu.get('util_gpu', 0)
                    gpu_active = gpu.get('gpu_active', 0)
                    gpu_idle = gpu.get('gpu_idle', 0)
                    
                    # Color code GPU utilization
                    if gpu_util < 30:
                        util_color = Colors.GREEN
                    elif gpu_util < 70:
                        util_color = Colors.YELLOW
                    else:
                        util_color = Colors.RED
                    
                    print(f"  GPU Utilization: {util_color}{gpu_active:.1f}%{Colors.NC}")
                    print(f"  GPU Active: {gpu_active:.1f}% | Idle: {gpu_idle:.1f}%")
                else:
                    print(f"  {Colors.DIM}GPU metrics unavailable (requires sudo){Colors.NC}")
                
                # Only show processes when GPU is actually working
                if gpu.get('has_metrics'):
                    if gpu_active > 5:
                        # GPU is active - show likely processes
                        processes = MetalStats.get_processes()
                        
                        if processes:
                            # Show top active processes
                            active_procs = [p for p in processes if p.get('cpu_util', 0) > 10][:10]
                            
                            if active_procs:
                                print(f"\n{Colors.GREEN}GPU Active ({gpu_active:.1f}%) - Likely GPU Processes:{Colors.NC}")
                                print(f"{Colors.YELLOW}⚠  macOS limitation: Per-process GPU% not available{Colors.NC}")
                                print(f"{Colors.DIM}   Showing GPU-capable processes with high CPU activity{Colors.NC}\n")
                                
                                p_headers = ['PID', 'Process Name', 'CPU%', 'MEM%']
                                p_widths = [7, 45, 7, 7]
                                
                                Table.print_header(p_widths)
                                Table.print_row(p_headers, p_widths)
                                Table.print_separator(p_widths)
                                
                                for proc in active_procs:
                                    cpu_util = proc.get('cpu_util', 0)
                                    
                                    # Color code high usage
                                    cpu_str = f"{cpu_util:.1f}"
                                    if cpu_util > 100:
                                        cpu_str = f"{Colors.RED}{cpu_util:.1f}{Colors.NC}"
                                    elif cpu_util > 50:
                                        cpu_str = f"{Colors.YELLOW}{cpu_util:.1f}{Colors.NC}"
                                    elif cpu_util > 10:
                                        cpu_str = f"{Colors.GREEN}{cpu_util:.1f}{Colors.NC}"
                                    
                                    row = [
                                        str(proc['pid']),
                                        proc['name'][:44],
                                        cpu_str,
                                        f"{proc.get('mem_util', 0):.1f}"
                                        ]
                                    Table.print_row(row, p_widths)
                                
                                Table.print_footer(p_widths)
                            else:
                                print(f"\n{Colors.YELLOW}GPU is active but no high-CPU processes detected{Colors.NC}")
                    else:
                        # GPU idle
                        print(f"\n{Colors.DIM}GPU is idle (no active workload){Colors.NC}")
                else:
                    # No GPU metrics - show note
                    print(f"\n{Colors.YELLOW}Run with sudo to see GPU utilization:{Colors.NC}")
                    print(f"  {Colors.CYAN}sudo ./gpu.py --monitor{Colors.NC}")
        
        print(f"\n{Colors.CYAN}Press Ctrl+C to exit{Colors.NC}")
        time.sleep(args.interval)
    
    if args.verbose > 0:
        print(f"\n{Colors.YELLOW}Monitoring stopped{Colors.NC}")


def cmd_stats(args):
    """Show current GPU statistics"""
    gpus = GPUDetector.detect_all()
    
    if not gpus:
        print("No GPUs detected")
        return
    
    gpu_type = gpus[0]['type']
    
    if gpu_type == 'NVIDIA':
        stats = NVIDIAStats.get_stats()
        
        # Detailed stats table
        headers = ['ID', 'Name', 'GPU%', 'Mem%', 'Used', 'Free', 'Total', 'Temp', 'Fan', 'Power', 'PwrLmt', 'ClkGPU', 'ClkMem']
        widths = [3, 18, 5, 5, 7, 7, 7, 5, 4, 6, 6, 7, 7]
        
        Table.print_header(widths)
        Table.print_row(headers, widths)
        Table.print_separator(widths)
        
        for gpu in stats:
            row = [
                str(gpu['id']),
                gpu['name'][:17],
                str(gpu['util_gpu']),
                str(gpu['util_mem']),
                f"{gpu['mem_used']}M",
                f"{gpu['mem_free']}M",
                f"{gpu['mem_total']}M",
                f"{gpu['temp']}°",
                f"{gpu['fan']}%",
                f"{gpu['power']:.0f}W",
                f"{gpu['power_max']:.0f}W",
                f"{gpu['clock_gpu']}M",
                f"{gpu['clock_mem']}M"
            ]
            Table.print_row(row, widths)
        
        Table.print_footer(widths)
        
        if args.verbose > 0:
            processes = NVIDIAStats.get_processes()
            if processes:
                print(f"\n{Colors.GREEN}GPU Processes ({len(processes)}):{Colors.NC}")
                p_headers = ['GPU', 'PID', 'Process Name', 'GPU%', 'Mem%', 'Memory', 'Enc%', 'Dec%']
                p_widths = [3, 7, 30, 5, 5, 10, 5, 5]
                Table.print_header(p_widths)
                Table.print_row(p_headers, p_widths)
                Table.print_separator(p_widths)
                for proc in processes:
                    row = [
                        str(proc.get('gpu_id', 0)),
                        str(proc['pid']),
                        proc['name'][:29],
                        str(proc.get('sm_util', 0)),
                        str(proc.get('mem_util', 0)),
                        f"{proc['memory_mb']}MB",
                        str(proc.get('enc_util', 0)),
                        str(proc.get('dec_util', 0))
                    ]
                    Table.print_row(row, p_widths)
                Table.print_footer(p_widths)
    
    elif gpu_type == 'Apple':
        stats = MetalStats.get_stats()
        for gpu in stats:
            print(f"{Colors.BOLD}GPU {gpu['id']}: {gpu['name']}{Colors.NC}")
            if 'chip' in gpu:
                print(f"  Type: Apple (Metal)")
                print(f"  Chip: {gpu['chip']}")
            
            print(f"\n{Colors.YELLOW}Note:{Colors.NC} Apple Silicon GPU metrics require sudo:")
            print(f"  {Colors.CYAN}sudo powermetrics --samplers gpu_power -i 1000 -n 1{Colors.NC}")
            
            if args.verbose > 0:
                # Show system memory stats as fallback
                try:
                    vm_result = subprocess.run(['vm_stat'], capture_output=True, text=True, timeout=5)
                    if vm_result.returncode == 0:
                        print(f"\n{Colors.GREEN}System Memory:{Colors.NC}")
                        for line in vm_result.stdout.split('\n')[:5]:
                            if line.strip():
                                print(f"  {line.strip()}")
                except:
                    pass
    
    else:
        print(f"Stats not fully supported for {gpu_type} GPUs")


def cmd_diagnose(args):
    """Run comprehensive diagnostics"""
    print(f"{Colors.BOLD}GPU Diagnostics{Colors.NC}\n")
    
    # System info
    print(f"{Colors.GREEN}System:{Colors.NC}")
    print(f"  OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  Python: {platform.python_version()}")
    
    # CUDA
    print(f"\n{Colors.GREEN}CUDA:{Colors.NC}")
    try:
        nvcc = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
        if nvcc.returncode == 0:
            version_line = [l for l in nvcc.stdout.split('\n') if 'release' in l.lower()][0]
            print(f"  {Colors.GREEN}✓{Colors.NC} {version_line.strip()}")
        else:
            print(f"  {Colors.RED}✗{Colors.NC} Not installed")
    except:
        print(f"  {Colors.RED}✗{Colors.NC} Not installed")
    
    try:
        smi = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
        if smi.returncode == 0:
            print(f"  {Colors.GREEN}✓{Colors.NC} nvidia-smi available")
        else:
            print(f"  {Colors.RED}✗{Colors.NC} nvidia-smi not available")
    except:
        print(f"  {Colors.RED}✗{Colors.NC} nvidia-smi not available")
    
    # ROCm
    print(f"\n{Colors.GREEN}ROCm:{Colors.NC}")
    try:
        rocm = subprocess.run(['rocm-smi', '--version'], capture_output=True, text=True, timeout=5)
        if rocm.returncode == 0:
            print(f"  {Colors.GREEN}✓{Colors.NC} {rocm.stdout.strip().split(chr(10))[0]}")
        else:
            print(f"  {Colors.RED}✗{Colors.NC} Not installed")
    except:
        print(f"  {Colors.RED}✗{Colors.NC} Not installed")
    
    # Metal
    print(f"\n{Colors.GREEN}Metal:{Colors.NC}")
    try:
        chip = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                             capture_output=True, text=True, timeout=5)
        if chip.returncode == 0 and ('Apple' in chip.stdout or platform.processor() == 'arm'):
            print(f"  {Colors.GREEN}✓{Colors.NC} {chip.stdout.strip()}")
        else:
            print(f"  {Colors.RED}✗{Colors.NC} Not available")
    except:
        print(f"  {Colors.RED}✗{Colors.NC} Not available")
    
    # Detected GPUs
    print(f"\n{Colors.GREEN}Detected GPUs:{Colors.NC}")
    gpus = GPUDetector.detect_all()
    if gpus:
        headers = ['ID', 'Name', 'Type', 'Backend', 'Memory']
        widths = [4, 35, 10, 8, 12]
        Table.print_header(widths)
        Table.print_row(headers, widths)
        Table.print_separator(widths)
        for gpu in gpus:
            memory = f"{gpu.get('memory_mb', 0)/1024:.1f}GB" if gpu.get('memory_mb', 0) > 0 else 'N/A'
            Table.print_row([str(gpu['id']), gpu['name'][:34], gpu['type'], gpu['backend'], memory], widths)
        Table.print_footer(widths)
    else:
        print("  No GPUs detected")
    
    # Save to file if requested
    if args.output:
        diag_data = {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'os': platform.system(),
                'release': platform.release(),
                'machine': platform.machine()
            },
            'gpus': gpus
        }
        with open(args.output, 'w') as f:
            json.dump(diag_data, f, indent=2)
        if args.verbose > 0:
            print(f"\n{Colors.GREEN}Saved to: {args.output}{Colors.NC}")


def cmd_memory(args):
    """Show GPU memory usage"""
    gpus = GPUDetector.detect_all()
    
    if not gpus or gpus[0]['type'] != 'NVIDIA':
        print("Memory details only available for NVIDIA GPUs")
        return
    
    stats = NVIDIAStats.get_stats()
    
    headers = ['ID', 'Name', 'Used', 'Free', 'Total', 'Util%']
    widths = [3, 25, 12, 12, 12, 6]
    
    Table.print_header(widths)
    Table.print_row(headers, widths)
    Table.print_separator(widths)
    
    for gpu in stats:
        used = gpu['mem_used']
        free = gpu['mem_free']
        total = gpu['mem_total']
        util = gpu['util_mem']
        
        row = [
            str(gpu['id']),
            gpu['name'][:24],
            f"{used:,}MB",
            f"{free:,}MB",
            f"{total:,}MB",
            f"{util}%"
        ]
        Table.print_row(row, widths)
    
    Table.print_footer(widths)


def cmd_processes(args):
    """List GPU processes with utilization"""
    gpus = GPUDetector.detect_all()
    
    if not gpus or gpus[0]['type'] != 'NVIDIA':
        print("Process listing only available for NVIDIA GPUs")
        return
    
    processes = NVIDIAStats.get_processes()
    
    if not processes:
        print("No active GPU processes")
        return
    
    headers = ['GPU', 'PID', 'Process Name', 'GPU%', 'Mem%', 'Memory', 'Enc%', 'Dec%']
    widths = [3, 7, 35, 5, 5, 12, 5, 5]
    
    Table.print_header(widths)
    Table.print_row(headers, widths)
    Table.print_separator(widths)
    
    total_memory = 0
    for proc in processes:
        gpu_util = proc.get('sm_util', 0)
        mem_util = proc.get('mem_util', 0)
        
        # Color code high usage
        gpu_str = f"{gpu_util}" if gpu_util < 80 else f"{Colors.RED}{gpu_util}{Colors.NC}"
        mem_str = f"{mem_util}" if mem_util < 80 else f"{Colors.YELLOW}{mem_util}{Colors.NC}"
        
        row = [
            str(proc.get('gpu_id', 0)),
            str(proc['pid']),
            proc['name'][:34],
            gpu_str,
            mem_str,
            f"{proc['memory_mb']:,}MB",
            str(proc.get('enc_util', 0)),
            str(proc.get('dec_util', 0))
        ]
        Table.print_row(row, widths)
        total_memory += proc['memory_mb']
    
    Table.print_footer(widths)
    
    print(f"\nTotal: {len(processes)} process(es) | GPU Memory: {total_memory:,}MB")
    
    if args.verbose > 0:
        high_gpu = [p for p in processes if p.get('sm_util', 0) > 50]
        if high_gpu:
            print(f"{Colors.YELLOW}High GPU usage: {len(high_gpu)} process(es) > 50%{Colors.NC}")


def cmd_watch(args):
    """Watch specific metric"""
    gpus = GPUDetector.detect_all()
    
    if not gpus or gpus[0]['type'] != 'NVIDIA':
        print("Watch mode only available for NVIDIA GPUs")
        return
    
    running = True
    
    def signal_handler(sig, frame):
        nonlocal running
        running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"Watching GPU {args.gpu_id} - {args.metric} (Ctrl+C to stop)\n")
    
    try:
        while running:
            stats = NVIDIAStats.get_stats()
            if args.gpu_id < len(stats):
                gpu = stats[args.gpu_id]
                
                # Map metric name
                metric_map = {
                    'gpu': 'util_gpu',
                    'memory': 'util_mem',
                    'temp': 'temp',
                    'power': 'power',
                    'clock': 'clock_gpu'
                }
                
                metric_key = metric_map.get(args.metric, args.metric)
                value = gpu.get(metric_key, 'N/A')
                
                timestamp = datetime.now().strftime('%H:%M:%S')
                print(f"[{timestamp}] GPU {args.gpu_id} {args.metric}: {value}")
            
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    
    if args.verbose > 0:
        print(f"\n{Colors.YELLOW}Watch stopped{Colors.NC}")


def cmd_topology(args):
    """Show GPU topology"""
    gpus = GPUDetector.detect_all()
    
    if not gpus or gpus[0]['type'] != 'NVIDIA':
        print("Topology only available for NVIDIA GPUs")
        return
    
    try:
        result = subprocess.run(['nvidia-smi', 'topo', '-m'], capture_output=True, text=True, timeout=5)
        print(result.stdout)
    except Exception as e:
        print(f"Error getting topology: {e}")


def cmd_version(args):
    """Show GPU driver and toolkit versions"""
    print(f"{Colors.BOLD}GPU Versions{Colors.NC}\n")
    
    # NVIDIA
    try:
        smi = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                            capture_output=True, text=True, timeout=5)
        if smi.returncode == 0:
            driver = smi.stdout.strip().split('\n')[0]
            print(f"{Colors.GREEN}NVIDIA Driver: {driver}{Colors.NC}")
    except:
        pass
    
    try:
        nvcc = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
        if nvcc.returncode == 0:
            for line in nvcc.stdout.split('\n'):
                if 'release' in line.lower():
                    print(f"{Colors.GREEN}CUDA: {line.strip()}{Colors.NC}")
    except:
        pass
    
    # ROCm
    try:
        rocm = subprocess.run(['rocm-smi', '--version'], capture_output=True, text=True, timeout=5)
        if rocm.returncode == 0:
            print(f"{Colors.GREEN}ROCm: {rocm.stdout.strip().split(chr(10))[0]}{Colors.NC}")
    except:
        pass
    
    # Metal
    try:
        macos = subprocess.run(['sw_vers', '-productVersion'], capture_output=True, text=True, timeout=5)
        if macos.returncode == 0:
            print(f"{Colors.GREEN}macOS: {macos.stdout.strip()}{Colors.NC}")
    except:
        pass


def cmd_use_gpu(args):
    """Continuously use GPU for stress testing"""
    gpus = GPUDetector.detect_all()
    
    if not gpus:
        print(f"{Colors.YELLOW}No GPUs detected - running CPU stress test{Colors.NC}")
        GPUStress.stress_simple()
        return
    
    gpu_type = gpus[0]['type']
    
    if gpu_type == 'NVIDIA':
        GPUStress.stress_nvidia(args.gpu_id, args.intensity)
    elif gpu_type == 'Apple':
        print(f"{Colors.GREEN}Apple Silicon GPU detected{Colors.NC}")
        GPUStress.stress_metal(args.intensity)
    elif gpu_type == 'AMD':
        print(f"{Colors.YELLOW}AMD GPU stress support coming soon{Colors.NC}")
        print(f"{Colors.YELLOW}Running CPU simulation...{Colors.NC}")
        GPUStress.stress_simple()
    else:
        GPUStress.stress_simple()


def main():
    parser = argparse.ArgumentParser(
        description='NeuronDB GPU Monitoring Tool - Complete GPU utility',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gpu.py --list                    List all GPUs (clean table)
  gpu.py --info                    Detailed GPU information
  gpu.py --monitor                 Real-time monitoring (like top)
  gpu.py --monitor -i 1            Monitor with 1s refresh
  gpu.py --stats                   Current GPU statistics
  gpu.py --stats -v                Stats with processes
  gpu.py --memory                  GPU memory usage
  gpu.py --processes               List GPU processes
  gpu.py --watch --metric gpu      Watch GPU utilization
  gpu.py --diagnose                Full diagnostics
  gpu.py --diagnose -o diag.json   Save diagnostics
  gpu.py --topology                Show GPU topology
  gpu.py --version                 Show driver versions
        """
    )
    
    # Commands (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--list', action='store_true', help='List all GPUs')
    group.add_argument('--info', action='store_true', help='Detailed GPU info')
    group.add_argument('--monitor', action='store_true', help='Real-time monitor (Ctrl+C to quit)')
    group.add_argument('--stats', action='store_true', help='Current GPU stats')
    group.add_argument('--memory', action='store_true', help='Memory usage')
    group.add_argument('--processes', action='store_true', help='GPU processes')
    group.add_argument('--watch', action='store_true', help='Watch metric')
    group.add_argument('--diagnose', action='store_true', help='Run diagnostics')
    group.add_argument('--topology', action='store_true', help='GPU topology')
    group.add_argument('--version', action='store_true', help='Driver versions')
    group.add_argument('--use-gpu', action='store_true', help='Continuously use GPU (stress test)')
    
    # Options
    parser.add_argument('-i', '--interval', type=int, default=2,
                       help='Refresh interval in seconds (default: 2)')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                       help='Verbose output (default: 0, use -v, -vv, -vvv)')
    parser.add_argument('--gpu-id', type=int, default=0,
                       help='GPU ID for watch/stress (default: 0)')
    parser.add_argument('--metric', default='gpu',
                       choices=['gpu', 'memory', 'temp', 'power', 'clock'],
                       help='Metric to watch (default: gpu)')
    parser.add_argument('--intensity', type=int, default=100,
                       help='Stress test intensity 0-100 (default: 100)')
    parser.add_argument('-o', '--output', help='Output file (JSON)')
    parser.add_argument('--json', action='store_true', help='JSON output format')
    
    args = parser.parse_args()
    
    # Execute command
    if args.list:
        cmd_list(args)
    elif args.info:
        cmd_info(args)
    elif args.monitor:
        cmd_monitor(args)
    elif args.stats:
        cmd_stats(args)
    elif args.memory:
        cmd_memory(args)
    elif args.processes:
        cmd_processes(args)
    elif args.watch:
        cmd_watch(args)
    elif args.diagnose:
        cmd_diagnose(args)
    elif args.topology:
        cmd_topology(args)
    elif args.version:
        cmd_version(args)
    elif args.use_gpu:
        cmd_use_gpu(args)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted{Colors.NC}")
        sys.exit(0)
    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.NC}", file=sys.stderr)
        sys.exit(1)
