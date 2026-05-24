from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict.

    Scalar and list values in *override* always win.  Nested dicts are merged
    depth-first so that a child config only needs to specify the keys it wants
    to change.
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str, _stack: Optional[List[Path]] = None) -> Dict[str, Any]:
    """Load a YAML config file, resolving ``extends`` inheritance chains.

    A config file may contain an optional top-level key::

        extends: "relative/path/to/base.yaml"

    The base file is loaded first (recursively, so chains are supported), then
    the child values are deep-merged on top — child always wins on conflict.
    The ``extends`` key is stripped from the returned dict.

    Cyclic ``extends`` chains raise ``ValueError``.
    """
    current = Path(config_path).resolve()
    _stack = _stack or []

    if current in _stack:
        chain = " -> ".join(str(p) for p in (_stack + [current]))
        raise ValueError(f"Cyclic config 'extends' detected: {chain}")

    with open(current, "r") as fh:
        cfg: Dict[str, Any] = yaml.safe_load(fh) or {}

    if not isinstance(cfg, dict):
        raise TypeError(f"Config file must be a YAML mapping, got {type(cfg).__name__}: {current}")

    parent_ref: Optional[str] = cfg.pop("extends", None)
    if not parent_ref:
        return cfg

    parent_path = (current.parent / parent_ref).resolve()
    parent_cfg = load_config(str(parent_path), _stack + [current])
    return _deep_merge(parent_cfg, cfg)

