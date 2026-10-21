"""Settings, and the two YAML files that fix every research choice before outcomes are seen."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("EDGELAB_HOME", Path(__file__).resolve().parents[2]))


def dsn() -> str:
    return os.environ.get("EDGELAB_DSN", "postgresql://edgelab:edgelab@localhost:5432/edgelab")


def load_research_config(path: Path | None = None) -> dict:
    return yaml.safe_load((path or ROOT / "config" / "research.yml").read_text(encoding="utf-8"))


def load_registry(path: Path | None = None) -> dict:
    registry = yaml.safe_load((path or ROOT / "config" / "hypotheses.yml").read_text(encoding="utf-8"))
    ids = [h["id"] for h in registry["hypotheses"]]
    if len(ids) != len(set(ids)):
        raise ValueError("hypothesis ids must be unique")
    return registry


def content_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]
