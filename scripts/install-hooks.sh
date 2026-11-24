#!/usr/bin/env bash
# Installs the repo's git hooks into .git/hooks. Run once after cloning.
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
cp "$root/scripts/pre-push" "$root/.git/hooks/pre-push"
chmod +x "$root/.git/hooks/pre-push"
echo "installed .git/hooks/pre-push"
