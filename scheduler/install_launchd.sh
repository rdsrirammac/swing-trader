#!/usr/bin/env bash
# Installs all swing-trader launchd jobs into ~/Library/LaunchAgents (macOS).
#
# Copies each scheduler/launchd/*.plist into ~/Library/LaunchAgents,
# substituting the __REPO_ROOT__ placeholder for the absolute repo path, then
# `launchctl load`s it. Idempotent: unloads any existing job with the same
# label first (ignoring errors if it wasn't loaded).
#
# Usage: bash scheduler/install_launchd.sh   (also: `make schedule-install`)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLIST_SRC_DIR="$REPO_ROOT/scheduler/launchd"
PLIST_DST_DIR="$HOME/Library/LaunchAgents"

mkdir -p "$PLIST_DST_DIR"
mkdir -p "$REPO_ROOT/logs"

echo "Installing swing-trader launchd jobs (repo: $REPO_ROOT)"

shopt -s nullglob
plists=("$PLIST_SRC_DIR"/*.plist)
shopt -u nullglob

if [ ${#plists[@]} -eq 0 ]; then
    echo "No .plist files found in $PLIST_SRC_DIR" >&2
    exit 1
fi

for src in "${plists[@]}"; do
    name="$(basename "$src")"
    label="${name%.plist}"
    dst="$PLIST_DST_DIR/$name"

    sed "s|__REPO_ROOT__|$REPO_ROOT|g" "$src" > "$dst"

    # Idempotent: unload first (ignore errors if not currently loaded).
    launchctl unload -w "$dst" >/dev/null 2>&1 || true

    if launchctl load -w "$dst" 2>/dev/null; then
        echo "  loaded: $label"
    else
        echo "  FAILED to load: $label (check launchctl error output)" >&2
    fi
done

echo "Done. Check status with: launchctl list | grep swingtrader"
