#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Editable configuration. Change values here instead of exporting env vars.
TARGET_REMOTE="origin"
TARGET_BRANCH="main"

echo "Repository root: ${PROJECT_ROOT}"
echo "Target: ${TARGET_REMOTE}/${TARGET_BRANCH}"
echo "Untracked files are preserved."

UNMERGED_FILES="$(git diff --name-only --diff-filter=U)"
if [[ -n "${UNMERGED_FILES}" ]]; then
    echo "Error: unresolved merge conflicts detected:"
    echo "${UNMERGED_FILES}"
    echo "Resolve the conflicts first, then rerun this script."
    exit 1
fi

TRACKED_CHANGES="$(git status --porcelain=v1 --untracked-files=no)"
if [[ -n "${TRACKED_CHANGES}" ]]; then
    echo "Error: tracked local changes detected:"
    echo "${TRACKED_CHANGES}"
    echo "Commit, stash, or discard tracked changes before pulling."
    exit 1
fi

git fetch origin main
git pull --ff-only origin main
git rev-parse HEAD
