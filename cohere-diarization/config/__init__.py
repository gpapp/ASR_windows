import json
import os
from pathlib import Path
from typing import Any

_config = None

def get_config() -> dict:
    global _config
    if _config is None:
        config_path = Path(__file__).parent / "thresholds.json"
        with open(config_path, "r") as f:
            _config = json.load(f)
    return _config

def get(section: str, key: str = None, default: Any = None) -> Any:
    """Get config value. Usage: get('diarization', 'max_clusters') or get('debug')"""
    cfg = get_config()
    if key is None:
        return cfg.get(section, default)
    return cfg.get(section, {}).get(key, default)

def is_debug() -> bool:
    return get_config().get("debug", False)

def set_debug(enabled: bool):
    get_config()["debug"] = enabled