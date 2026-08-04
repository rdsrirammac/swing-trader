#!/usr/bin/env bash
# Uninstalls all swing-trader launchd jobs from ~/Library/LaunchAgents (macOS).
#
# Inverse of install_launchd.sh: unloads and removes each installed plist.
# Safe to run even if nothing is installed (each step ignores errors).
#
# Usage: bash scheduler/uninstall_launchd.sh   (also: `make schedule-uninstall`)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLIST_SRC_DIR="$REPO_ROOT/scheduler/launchd"
PLIST_DST_DIR="$HOME/Library/LaunchAgents"

echo "Uninstalling swing-trader launchd jobs"

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

    if [ -f "$dst" ]; then
        launchctl unload -w "$dst" >/dev/null 2>&1 || true
        rm -f "$dst"
        echo "  removed: $label"
    else
        echo "  not installed: $label"
    fi
done

echo "Done."
