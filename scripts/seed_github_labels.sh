#!/usr/bin/env bash
# Create/update all labels defined in .github/labels.yml via the GitHub CLI.
# Requires: `gh auth login` already run, and the repo already created & pushed
# (run from inside the repo, or set GH_REPO=owner/name).
#
# Usage: bash scripts/seed_github_labels.sh
set -euo pipefail

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI ('gh') not found. Install it (https://cli.github.com) and run 'gh auth login' first." >&2
  exit 1
fi

if ! command -v yq >/dev/null 2>&1 && ! python3 -c "import yaml" 2>/dev/null; then
  echo "Need either 'yq' or Python's PyYAML installed to parse .github/labels.yml." >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABELS_FILE="$REPO_ROOT/.github/labels.yml"

python3 - "$LABELS_FILE" <<'PYEOF'
import subprocess
import sys

import yaml

with open(sys.argv[1]) as f:
    labels = yaml.safe_load(f)

for label in labels:
    name, color, desc = label["name"], label["color"], label.get("description", "")
    # `gh label create` fails if it already exists; fall back to `edit`.
    create = subprocess.run(
        ["gh", "label", "create", name, "--color", color, "--description", desc, "--force"],
        capture_output=True, text=True,
    )
    if create.returncode == 0:
        print(f"OK: {name}")
    else:
        print(f"WARN: {name}: {create.stderr.strip()}")
PYEOF

echo "Done. Re-run any time you edit .github/labels.yml."
