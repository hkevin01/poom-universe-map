import os
import yaml
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Config:
    raw: Dict[str, Any]

    def get(self, key: str, default=None):
        return self.raw.get(key, default)


def load_yaml(path: str) -> Config:
    """Load a YAML file from disk.

    Args:
        path: Path to YAML file.
    Returns:
        Config wrapper around the raw dict.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(data)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
