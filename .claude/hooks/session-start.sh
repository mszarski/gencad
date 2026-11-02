#!/bin/bash
# Session startup hook for Claude Code on the web
set -euo pipefail

# Only run in remote Claude Code environment
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
    exit 0
fi

echo "Setting up development environment..."

# CPU torch is enough for the smoke tests; GPU work runs as HF Jobs
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r "$CLAUDE_PROJECT_DIR/gencad/jobs/requirements-job.txt"

# Clone upstream GenCAD at the pinned commit, apply patches/, symlink it in
"$CLAUDE_PROJECT_DIR/gencad/setup.sh" || echo "Warning: gencad setup failed"

echo "Session startup complete"
