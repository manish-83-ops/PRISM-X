"""PRISMX Configuration Loader and Canonical Hash Generator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "CONFIG.yaml"

def canonical_json(data: Any) -> str:
    """Produces deterministic canonical JSON with sorted keys and no unnecessary whitespace."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"))

def compute_config_hash(config_dict: dict[str, Any]) -> str:
    """Computes SHA-256 hash of canonical JSON config."""
    c_json = canonical_json(config_dict)
    return hashlib.sha256(c_json.encode("utf-8")).hexdigest()

def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Loads configuration dictionary from YAML and attaches its canonical config hash."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    cfg_hash = compute_config_hash(cfg)
    cfg["_config_hash"] = cfg_hash
    return cfg

def get_config_hash(config_path: Path | str | None = None) -> str:
    cfg = load_config(config_path)
    return cfg["_config_hash"]
