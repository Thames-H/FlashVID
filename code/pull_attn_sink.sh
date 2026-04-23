#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-attn-sink}"

cd "${PROJECT_ROOT}"

echo "repo root: ${PROJECT_ROOT}"
echo "sync branch: ${REMOTE}/${BRANCH}"

git fetch "${REMOTE}" "${BRANCH}"

if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
    git checkout "${BRANCH}"
else
    git checkout -b "${BRANCH}" "${REMOTE}/${BRANCH}"
fi

git pull --ff-only "${REMOTE}" "${BRANCH}"
git submodule update --init --recursive
git rev-parse HEAD
