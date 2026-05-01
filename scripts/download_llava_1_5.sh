#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-liuhaotian/llava-v1.5-7b}"
REVISION="${REVISION:-main}"
MODEL_DIR="${MODEL_DIR:-${HOME}/autodl-tmp/llavav-1.5-7b}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HOME="${HF_HOME:-${HOME}/autodl-tmp/hf_cache}"

export HF_ENDPOINT
export HF_HOME

mkdir -p "$MODEL_DIR"

echo "Downloading ${MODEL_ID}@${REVISION}"
echo "Target directory: ${MODEL_DIR}"
echo "HF_ENDPOINT: ${HF_ENDPOINT}"
echo "HF_HOME: ${HF_HOME}"

if command -v hf >/dev/null 2>&1; then
    hf download "$MODEL_ID" \
        --revision "$REVISION" \
        --local-dir "$MODEL_DIR"
elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$MODEL_ID" \
        --revision "$REVISION" \
        --local-dir "$MODEL_DIR"
else
    python - "$MODEL_ID" "$REVISION" "$MODEL_DIR" <<'PY'
import importlib.util
import subprocess
import sys

repo_id, revision, local_dir = sys.argv[1:]

if importlib.util.find_spec("huggingface_hub") is None:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--user",
            "huggingface_hub",
        ]
    )

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=repo_id,
    revision=revision,
    local_dir=local_dir,
)
PY
fi

if find "$MODEL_DIR/.cache/huggingface/download" -name "*.incomplete" -print -quit 2>/dev/null | grep -q .; then
    echo "Download left incomplete files under ${MODEL_DIR}/.cache/huggingface/download" >&2
    exit 1
fi

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
    echo "Missing config.json in ${MODEL_DIR}; download may have failed." >&2
    exit 1
fi

echo "Download finished: ${MODEL_DIR}"
