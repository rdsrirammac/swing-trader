"""macOS desktop notifications (SRS 3.10, AL-005).

Uses `pync` (a thin wrapper around `terminal-notifier`) if available and
the process is running on macOS. Must never crash the caller on
Linux/CI, or when `pync`/`terminal-notifier` isn't installed.
"""
from __future__ import annotations

import sys

from swing_trader.logging_setup import get_logger

logger = get_logger("notify.macos")


def send_macos_notification(title: str, message: str, subtitle: str | None = None) -> bool:
    """AL-005: send a native macOS notification.

    Returns True on success, False otherwise. Best-effort only — any
    failure (missing `pync`, non-darwin platform, `terminal-notifier`
    missing, notification permission denied, etc.) is caught, logged as a
    warning, and reported as False rather than raised.
    """
    if sys.platform != "darwin":
        logger.warning("send_macos_notification called on non-darwin platform (%s); skipping", sys.platform)
        return False

    try:
        import pync

        kwargs: dict = {"title": title}
        if subtitle:
            kwargs["subtitle"] = subtitle
        pync.notify(message, **kwargs)
        return True
    except ImportError:
        logger.warning("pync is not installed; cannot send macOS notification")
        return False
    except Exception as e:
        logger.warning("Failed to send macOS notification: %s", e)
        return False
