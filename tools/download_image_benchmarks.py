import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_ROOT = Path("/root/autodl-tmp/benchmark")

BENCHMARK_SOURCES = {
    "GQA": {
        "repo_id": "lmms-lab/GQA",
        "allow_patterns": [
            "testdev_balanced_instructions/*",
            "testdev_balanced_images/*",
        ],
    },
    "ScienceQA": {
        "repo_id": "lmms-lab/ScienceQA",
    },
    "MMBench": {
        "repo_id": "lmms-lab/MMBench",
    },
    "MME": {
        "repo_id": "lmms-lab/MME",
    },
    "POPE": {
        "repo_id": "lmms-lab/POPE",
        "allow_patterns": [
            "data/*",
        ],
    },
    "OCRBench": {
        "repo_id": "lmms-lab/OCRBench-v2",
        "allow_patterns": [
            "data/*",
        ],
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="Root directory for downloaded benchmarks.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["all"],
        help="Benchmarks to download. Use 'all' or any of: "
        + ", ".join(BENCHMARK_SOURCES.keys()),
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    return parser.parse_args()


def normalize_selection(raw_benchmarks):
    if len(raw_benchmarks) == 1 and raw_benchmarks[0].lower() == "all":
        return list(BENCHMARK_SOURCES.keys())

    normalized = []
    lookup = {name.lower(): name for name in BENCHMARK_SOURCES}
    for name in raw_benchmarks:
        key = name.lower()
        if key not in lookup:
            raise ValueError(
                f"Unknown benchmark '{name}'. Expected one of: "
                + ", ".join(BENCHMARK_SOURCES.keys())
            )
        normalized.append(lookup[key])
    return normalized


def download_benchmark(name: str, root: Path, force_download: bool):
    config = BENCHMARK_SOURCES[name]
    target_dir = root / name
    target_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    download_kwargs = {
        "repo_id": config["repo_id"],
        "repo_type": "dataset",
        "local_dir": str(target_dir),
        "force_download": force_download,
    }
    if token:
        download_kwargs["token"] = token
    if "allow_patterns" in config:
        download_kwargs["allow_patterns"] = config["allow_patterns"]

    print(f"[download] {name}: {config['repo_id']} -> {target_dir}")
    snapshot_download(**download_kwargs)
    print(f"[done] {name}: {target_dir}")


def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    for name in normalize_selection(args.benchmarks):
        download_benchmark(name, root, args.force_download)

    print(f"All requested benchmarks are available under: {root}")


if __name__ == "__main__":
    main()
