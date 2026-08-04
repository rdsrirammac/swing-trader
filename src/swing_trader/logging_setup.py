"""Logging configuration (NFR 4.5: rotated daily logs, 30-day retention)."""
from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import yaml

from swing_trader.config import REPO_ROOT


def setup_logging() -> None:
    cfg_path = REPO_ROOT / "config" / "logging.yaml"
    (REPO_ROOT / "logs").mkdir(exist_ok=True)

    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        # Make the file handler path absolute & rooted at the repo.
        file_handler = cfg.get("handlers", {}).get("file")
        if file_handler:
            file_handler["filename"] = str(REPO_ROOT / file_handler["filename"])
        logging.config.dictConfig(cfg)
    else:
        logging.basicConfig(level=logging.INFO)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"swing_trader.{name}")
