import argparse
import os
from pathlib import Path

from huggingface_hub import (
    hf_hub_download,
    list_repo_files,
    snapshot_download,
)


DEFAULT_REPO_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_LOCAL_DIRNAME = "Qwen3-VL-8B-Instruct"
DEFAULT_AUTODL_PARENT = Path.home() / "autodl-tmp"


def default_local_dir() -> Path:
    override = os.environ.get("QWEN3_VL_MODEL_PATH")
    if override:
        return Path(override).expanduser()
    if os.name != "nt":
        return DEFAULT_AUTODL_PARENT / DEFAULT_LOCAL_DIRNAME
    return Path(__file__).resolve().parents[1] / DEFAULT_LOCAL_DIRNAME


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--local-dir",
        default=str(default_local_dir()),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    local_dir = Path(args.local_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

    print(f"Downloading {args.repo_id} to {local_dir}")
    try:
        snapshot_download(
            repo_id=args.repo_id,
            local_dir=str(local_dir),
        )
    except Exception as exc:
        print(f"Snapshot download interrupted: {exc}")
        print("Retrying any missing files individually...")
        for filename in list_repo_files(args.repo_id):
            target = local_dir / filename
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id=args.repo_id,
                filename=filename,
                local_dir=str(local_dir),
            )
    print(f"Model available at: {local_dir}")


if __name__ == "__main__":
    main()
