"""I/O helpers: config loading, seeding, JSON I/O."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file with `${env:VAR,default}` interpolation.

    Parameters
    ----------
    config_path
        Path to a YAML config file.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration as a nested dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = f.read()

    # Interpolate ${env:VAR,default} placeholders
    import re

    def _env_sub(match: re.Match) -> str:
        var = match.group(1)
        default = match.group(2) if match.group(2) is not None else ""
        return os.environ.get(var, default)

    pattern = re.compile(r"\$\{env:([^,}]+)(?:,([^}]*))?\}")
    raw = pattern.sub(_env_sub, raw)

    # Simple ${now:fmt} interpolation
    from datetime import datetime

    def _now_sub(match: re.Match) -> str:
        fmt = match.group(1) if match.group(1) else "%Y%m%d_%H%M%S"
        return datetime.now().strftime(fmt)

    raw = re.sub(r"\$\{now:([^}]+)\}", _now_sub, raw)

    cfg = yaml.safe_load(raw) or {}

    # Auto-merge sibling configs into default if loading default.yaml
    if config_path.stem == "default":
        parent = config_path.parent
        for sibling in sorted(parent.glob("*_config.yaml")):
            try:
                sub_raw = sibling.read_text(encoding="utf-8")
                sub_raw = pattern.sub(_env_sub, sub_raw)
                sub_raw = re.sub(r"\$\{now:([^}]+)\}", _now_sub, sub_raw)
                sub_cfg = yaml.safe_load(sub_raw)
                if isinstance(sub_cfg, dict):
                    for k, v in sub_cfg.items():
                        if k not in cfg:
                            cfg[k] = v
            except Exception:
                continue

    return cfg


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed all RNGs for reproducibility.

    Parameters
    ----------
    seed
        Integer seed.
    deterministic
        If True, force deterministic CuDNN (slower, but reproducible).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    """Write an object to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=_json_default)


def load_json(path: str | Path) -> Any:
    """Load a JSON file."""
    path = Path(path)
    with open(path) as f:
        return json.load(f)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def hash_str(s: str, length: int = 8) -> str:
    """Return a short MD5 hash of a string."""
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:length]
