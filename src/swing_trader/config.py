"""Central configuration loader.

Loads config/settings.yaml (+ regimes.yaml) and config/api_keys.yaml (secrets,
git-ignored), resolving ${VAR:-default} placeholders against environment
variables / .env (NFR 4.4: no hardcoded secrets).
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"

_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-([^}]*))?\}")


def _resolve_env_vars(value: Any) -> Any:
    """Recursively resolve ${VAR:-default} placeholders in strings."""
    if isinstance(value, str):
        def repl(match: re.Match) -> str:
            var_name, _, default = match.groups()
            return os.environ.get(var_name, default if default is not None else "")

        return _VAR_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    return value


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


class Settings:
    """Typed-ish accessor over the merged settings/regimes/api_keys config."""

    def __init__(self, data: dict, regimes: dict, secrets: dict):
        self._data = data
        self._regimes = regimes
        self._secrets = secrets

    def get(self, dotted_path: str, default: Any = None) -> Any:
        """Fetch a nested value using 'a.b.c' dotted-path notation."""
        node: Any = self._data
        for part in dotted_path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def raw(self) -> dict:
        return self._data

    @property
    def regimes(self) -> dict:
        return self._regimes

    def secret(self, dotted_path: str, default: Any = None) -> Any:
        node: Any = self._secrets
        for part in dotted_path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def database_url(self) -> str:
        db = self._data.get("database", {})
        return (
            f"{db.get('driver', 'postgresql+psycopg2')}://"
            f"{db.get('user')}:{db.get('password')}@"
            f"{db.get('host')}:{db.get('port')}/{db.get('name')}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(REPO_ROOT / ".env", override=False)

    data = _resolve_env_vars(_load_yaml(CONFIG_DIR / "settings.yaml"))
    regimes = _resolve_env_vars(_load_yaml(CONFIG_DIR / "regimes.yaml"))
    secrets = _resolve_env_vars(_load_yaml(CONFIG_DIR / "api_keys.yaml"))

    return Settings(data=data, regimes=regimes, secrets=secrets)
