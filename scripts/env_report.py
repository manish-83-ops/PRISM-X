"""Environment and Hardware Information Reporter.

Gathers hardware, OS, Python version, git commit, and package versions,
saving the manifest to data/manifests/env_report.json and printing a summary.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
import psutil

def get_git_info() -> dict[str, str]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "uncommitted"
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    except Exception:
        branch = "unknown"
    return {"commit": commit, "branch": branch}

def get_installed_packages() -> dict[str, str]:
    try:
        from importlib.metadata import distributions
        return {dist.metadata["Name"]: dist.version for dist in distributions()}
    except Exception:
        return {}

def collect_env_report() -> dict:
    repo_root = Path(__file__).resolve().parent.parent
    git_info = get_git_info()
    
    # Check if repo is in OneDrive or Dropbox
    repo_str = str(repo_root)
    cloud_sync = []
    if "onedrive" in repo_str.lower():
        cloud_sync.append("OneDrive")
    if "dropbox" in repo_str.lower():
        cloud_sync.append("Dropbox")
        
    cpu_freq = psutil.cpu_freq()
    freq_mhz = cpu_freq.current if cpu_freq else None
    
    report = {
        "timestamp": psutil.boot_time(),
        "platform": {
            "os": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "architecture": platform.architecture()[0],
            "machine": platform.machine(),
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "hardware": {
            "processor": platform.processor(),
            "cpu_physical_cores": psutil.cpu_count(logical=False),
            "cpu_logical_cores": psutil.cpu_count(logical=True),
            "cpu_freq_mhz": freq_mhz,
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "disk_free_gb": round(shutil.disk_usage(repo_root).free / (1024**3), 2),
            "disk_total_gb": round(shutil.disk_usage(repo_root).total / (1024**3), 2),
        },
        "git": git_info,
        "repo_path": str(repo_root),
        "cloud_sync_detected": cloud_sync,
        "docker_available": shutil.which("docker") is not None,
    }
    
    # Save manifest
    manifest_dir = repo_root / "data" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    out_file = manifest_dir / "env_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    return report

if __name__ == "__main__":
    rep = collect_env_report()
    print("Environment report collected:")
    print(f"  OS: {rep['platform']['os']}")
    print(f"  Python: {rep['python']['version'].split()[0]}")
    print(f"  CPU: {rep['hardware']['processor']} ({rep['hardware']['cpu_physical_cores']}p / {rep['hardware']['cpu_logical_cores']}l cores)")
    print(f"  RAM: {rep['hardware']['ram_total_gb']} GB")
    print(f"  Disk Free: {rep['hardware']['disk_free_gb']} GB")
    print(f"  Docker: {'Available' if rep['docker_available'] else 'Not available'}")
    if rep['cloud_sync_detected']:
        print(f"  WARNING: Detected repo inside {', '.join(rep['cloud_sync_detected'])} sync folder!")
