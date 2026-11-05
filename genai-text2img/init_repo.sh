#!/usr/bin/env bash
# Initialise git and (optionally) publish to GitHub.
#   ./init_repo.sh                      -> local repo + first commit
#   ./init_repo.sh genai-text2img       -> also creates a GitHub repo via the `gh` CLI and pushes
set -euo pipefail
cd "$(dirname "$0")"
[ -d .git ] || git init -b main
git add -A
git commit -m "Initial commit: data prep, CLIP baseline, SD test-run pipeline, notebook, proposal" || true
if [ "${1:-}" != "" ]; then
  if command -v gh >/dev/null; then
    gh repo create "$1" --private --source=. --remote=origin --push
  else
    echo "Install GitHub CLI (gh) or run:  git remote add origin https://github.com/<you>/$1.git && git push -u origin main"
  fi
fi
