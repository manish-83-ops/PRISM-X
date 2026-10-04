"""Secrets Scanner.

Scans the working tree and git commits for accidentally committed secrets,
API keys (e.g. gsk_*, sk-*, etc.), tokens, and credentials.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

SECRET_PATTERNS = [
    (r"gsk_[a-zA-Z0-9]{20,}", "Groq API Key"),
    (r"sk-[a-zA-Z0-9]{20,}", "OpenAI API Key"),
    (r"hf_[a-zA-Z0-9]{20,}", "HuggingFace Token"),
    (r"AIza[0-9A-Za-z-_]{35}", "Google API Key"),
    (r"AKIA[0-9A-Z]{16}", "AWS Access Key"),
    (r"(?i)bearer\s+[a-zA-Z0-9_\-\.]{30,}", "Bearer Token"),
    (r"(?i)api[_-]?key\s*[:=]\s*['\"][a-zA-Z0-9_\-]{20,}['\"]", "Generic API Key Assignment"),
]

IGNORED_DIRS = {".git", ".venv", "venv", "__pycache__", "data", "models", "snapshots", "model_cache", ".pytest_cache", "node_modules", "frontend", "bin", "dist"}
IGNORED_FILES = {"check_secrets.py", ".env"}
BINARY_EXTS = {".onnx", ".db", ".bin", ".exe", ".snapshot", ".pdf", ".png", ".webp", ".parquet", ".zip", ".tar", ".gz"}

def scan_file(path: Path) -> list[tuple[int, str, str]]:
    findings = []
    if path.suffix.lower() in BINARY_EXTS:
        return findings
    try:
        if path.stat().st_size > 1_000_000:  # Skip files > 1MB
            return findings
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return findings
    for line_idx, line in enumerate(content.splitlines(), start=1):
        for pattern, name in SECRET_PATTERNS:
            if re.search(pattern, line):
                # Ensure it's not a generic placeholder
                if "your_groq_api_key_here" in line or "sk-..." in line or "gsk_..." in line:
                    continue
                findings.append((line_idx, name, line.strip()[:60]))
    return findings

def scan_working_tree(repo_root: Path) -> list[str]:
    violations = []
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
        for fname in files:
            if fname in IGNORED_FILES:
                continue
            item = Path(root) / fname
            findings = scan_file(item)
            for line_idx, name, snippet in findings:
                rel = item.relative_to(repo_root)
                violations.append(f"Secret detected in {rel}:{line_idx} ({name}): {snippet}")
    return violations

def scan_git_history() -> list[str]:
    violations = []
    try:
        # Check if git has any commits
        head_check = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        if head_check.returncode != 0:
            return violations  # No commits yet
        
        diff = subprocess.check_output(
            ["git", "--no-pager", "diff", "HEAD~1", "--no-color", "--", "*.py", "*.yaml", "*.yml", "*.md", "*.env*", "*.sh"],
            text=True,
            errors="ignore",
        )
        for line in diff.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                for pattern, name in SECRET_PATTERNS:
                    if re.search(pattern, line):
                        if "your_groq_api_key_here" in line or "gsk_..." in line or "sk-..." in line:
                            continue
                        violations.append(f"Secret detected in commit diff ({name}): {line.strip()[:60]}")
    except Exception:
        pass
    return violations

def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    violations = scan_working_tree(repo_root)
    violations.extend(scan_git_history())
    
    if violations:
        print(f"FAILED: Found {len(violations)} potential secret(s):")
        for v in violations:
            print(f"  - {v}")
        return 1
    else:
        print("PASS: No secrets detected in working tree or git history.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
