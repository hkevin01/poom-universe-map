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
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(data)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
